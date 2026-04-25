"""
Cross-lingual sentiment validation.

Tests whether using different LLMs / prompts in different languages biased the
original sentiment scores. For each sample post we:
  1. take the existing opinion (from the original analyzer)
  2. translate the text to the OTHER language via DeepL
  3. re-score with the OTHER analyzer (matching its original input format)
  4. compare original vs cross-lingual scores

Workflow:
    sample_tweets          - stratified-by-opinion sample of 200 tweets
    translate_weibo        - DeepL zh -> en for the weibo sample
    translate_tweets       - DeepL en -> zh for the tweet sample
    submit_weibo_gpt       - submit GPT-5-mini batch on translated weibo
    retrieve_weibo_gpt     - poll & download batch results
    parse_weibo_gpt        - parse jsonl results into parquet
    analyze_tweet_kimi     - Kimi live API on translated tweets (resumable)
    diff                   - agreement metrics + per-row CSV report

Required additions to config.py (see config.date.example.py for keys):
    DEEPL_API_KEY              (free vs pro endpoint auto-selected from key)
    KIMI_API_KEY
    KIMI_BASE_URL
    KIMI_MODEL
    CROSS_LINGUAL_DIR

The translated-weibo GPT reanalysis reuses the existing BATCH_PROMPT_ID,
BATCH_PROMPT_VERSION, and BATCH_MODEL — same prompt that scored the original
tweets, so any score shift can be attributed to the content (translation),
not to a different English prompt.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import deepl
import fire
import jsonlines
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from config import (
    OPENAI_API_KEY,
    SENTIMENT_OUTPUT_DIR,
    DEEPL_API_KEY,
    KIMI_API_KEY,
    KIMI_BASE_URL,
    KIMI_MODEL,
    CROSS_LINGUAL_DIR,
    BATCH_PROMPT_ID,
    BATCH_PROMPT_VERSION,
    BATCH_MODEL,
)
from batch_sentiment_analysis import load_parquet_files_filtered_by_date

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CROSS_LINGUAL_PATH = Path(CROSS_LINGUAL_DIR)
CROSS_LINGUAL_PATH.mkdir(parents=True, exist_ok=True)

DEFAULT_SAMPLE_SIZE = 200
DEFAULT_SEED = 42

WEIBO_SAMPLE_FILE = CROSS_LINGUAL_PATH / "weibo_translation_sample.parquet"
TWEET_SAMPLE_FILE = CROSS_LINGUAL_PATH / "tweet_translation_sample.parquet"
WEIBO_TRANSLATED_FILE = CROSS_LINGUAL_PATH / "weibo_translated.parquet"
TWEET_TRANSLATED_FILE = CROSS_LINGUAL_PATH / "tweet_translated.parquet"
WEIBO_REANALYZED_FILE = CROSS_LINGUAL_PATH / "weibo_reanalyzed_gpt.parquet"
TWEET_REANALYZED_FILE = CROSS_LINGUAL_PATH / "tweet_reanalyzed_kimi.parquet"
DEEPL_CACHE_FILE = CROSS_LINGUAL_PATH / "deepl_cache.json"
WEIBO_BATCH_REQUESTS = CROSS_LINGUAL_PATH / "weibo_en_requests.jsonl"
WEIBO_BATCH_RESULTS = CROSS_LINGUAL_PATH / "weibo_en_results.jsonl"
WEIBO_BATCH_STATE = CROSS_LINGUAL_PATH / "weibo_en_batch_state.json"

# Original Chinese system prompt copied verbatim from
# youth-analysis/ai_sentiment_analyzer.py so the Kimi reanalysis sees exactly
# the prompt used for the original weibo scoring.
CHINESE_SYSTEM_PROMPT = """
你将阅读一段用户在微博上发布的文本。请分析该文本对AI技术的观点，为其分配一个意见标签。你的输出必须是一个 JSON 对象，只包含字段 "opinion"，取值为 -2、-1、0、1、2 或 "cannot tell"。

标签定义：
2 = 对AI技术表达强烈的积极态度，明确主张AI技术利大于弊，并认为其对社会带来显著正面影响（如明显提升生活便利性、改善健康与医疗、带来经济机会、提高学习和工作效率、改善安全性、为研究和创新提供帮助等，或表达对使用AI的依赖或信任）。
1 = 整体倾向正面，但态度温和或带有一定保留（表达支持但语气不强；认为"总体有益"但同时提到风险或局限；表达期待、兴趣或积极看法，但无强烈赞美）。
0 = 客观中立或难以判断倾向（同时提到利弊，但无明确倾向）。
-1 = 整体倾向负面，但态度温和或带有一定保留（表达担忧或反对但未完全否定AI；认为"有风险或者有弊端"但承认某些益处；表达谨慎、不安或负面看法，但无强烈否定）。
-2 = 对AI技术表达强烈的消极态度，明确主张AI技术弊大于利，并认为其对社会带来显著负面影响（如加剧失业、经济增长泡沫、隐私风险、边缘群体偏见、错误信息或谣言、安全威胁等，或表达对AI的抗拒或不信任）。
"cannot tell" = 文本未表达对AI的任何态度（如内容与AI无关，或未反映观点）。
请仅返回一个JSON对象，例如：
{
"opinion": 1
}
""".strip()


# ============================================================
# Stratified sampling
# ============================================================


def _stratified_proportional_sample(
    df: pd.DataFrame, opinion_col: str, n: int, seed: int
) -> pd.DataFrame:
    """Proportionally stratified sample of n rows by opinion class."""
    df = df.copy()
    df["_op_int"] = pd.to_numeric(df[opinion_col], errors="coerce")
    df = df.dropna(subset=["_op_int"])
    df["_op_int"] = df["_op_int"].astype(int)
    df = df[df["_op_int"].isin([-2, -1, 0, 1, 2])]

    counts = df["_op_int"].value_counts().sort_index()
    total = counts.sum()
    raw = {op: cnt / total * n for op, cnt in counts.items()}
    allocations = {op: int(round(v)) for op, v in raw.items()}

    diff = n - sum(allocations.values())
    if diff != 0:
        fracs = sorted(
            raw.items(), key=lambda x: x[1] - int(x[1]), reverse=(diff > 0)
        )
        for op, _ in fracs[: abs(diff)]:
            allocations[op] += 1 if diff > 0 else -1

    logger.info(f"Stratified allocations (proportional): {dict(sorted(allocations.items()))}")

    samples = []
    for op, k in allocations.items():
        if k <= 0:
            continue
        cls_df = df[df["_op_int"] == op]
        if len(cls_df) <= k:
            logger.warning(f"Class {op} only has {len(cls_df)} rows, need {k}; using all")
            samples.append(cls_df)
        else:
            samples.append(cls_df.sample(n=k, random_state=seed))
    return pd.concat(samples, ignore_index=True).drop(columns=["_op_int"])


def sample_tweets(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    output_file: Union[str, Path] = TWEET_SAMPLE_FILE,
):
    """Sample tweets stratified by their existing GPT opinion score."""
    logger.info("=== Sampling tweets ===")

    batch_results_path = Path(SENTIMENT_OUTPUT_DIR) / "batch_results.parquet"
    if not batch_results_path.exists():
        raise FileNotFoundError(f"Missing {batch_results_path}; run analyze first")
    batch_df = pd.read_parquet(batch_results_path)
    batch_df["custom_id"] = batch_df["custom_id"].astype(str)
    logger.info(f"Loaded {len(batch_df)} batch results")

    tweets_df = load_parquet_files_filtered_by_date()
    tweets_df["id"] = tweets_df["id"].astype(str)

    merged = batch_df.merge(
        tweets_df[["id", "text", "createdAt"]],
        left_on="custom_id",
        right_on="id",
        how="inner",
    ).drop_duplicates(subset=["id"], keep="first")
    logger.info(f"Merged: {len(merged)} tweets with sentiment")

    sampled = _stratified_proportional_sample(
        merged, opinion_col="opinion", n=sample_size, seed=seed
    )
    sampled = sampled.rename(columns={"opinion": "original_opinion"})
    out = sampled[["id", "text", "original_opinion", "createdAt"]]

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    logger.info(f"Saved {len(out)} tweet samples to {output_path}")
    print("\n=== Sample distribution ===")
    print(out["original_opinion"].value_counts().sort_index())
    print(f"Saved to: {output_path}")


# ============================================================
# DeepL translation
# ============================================================


def _load_deepl_cache() -> dict:
    if DEEPL_CACHE_FILE.exists():
        with open(DEEPL_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_deepl_cache(cache: dict):
    with open(DEEPL_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


_deepl_client: Optional[deepl.DeepLClient] = None


def _get_deepl_client() -> deepl.DeepLClient:
    """Lazily build the DeepL client. Free vs Pro endpoint is auto-selected
    from the auth key suffix (`:fx` for free)."""
    global _deepl_client
    if _deepl_client is None:
        _deepl_client = deepl.DeepLClient(DEEPL_API_KEY)
    return _deepl_client


def _deepl_translate(text: str, target_lang: str, cache: dict) -> str:
    """Translate one piece of text via DeepL with on-disk cache."""
    if not text.strip():
        return ""
    key = f"{target_lang}::{text}"
    if key in cache:
        return cache[key]

    result = _get_deepl_client().translate_text(text, target_lang=target_lang)
    translated = result.text
    cache[key] = translated
    return translated


def _translate_column(
    df: pd.DataFrame, src_col: str, target_lang: str, save_every: int = 25
) -> pd.DataFrame:
    cache = _load_deepl_cache()
    out = []
    for i, text in enumerate(
        tqdm(df[src_col].fillna("").astype(str), desc=f"DeepL→{target_lang}")
    ):
        out.append(_deepl_translate(text, target_lang, cache))
        if (i + 1) % save_every == 0:
            _save_deepl_cache(cache)
    _save_deepl_cache(cache)
    df = df.copy()
    df["translated_text"] = out
    return df


def translate_weibo(
    weibo_sample_file: Union[str, Path] = WEIBO_SAMPLE_FILE,
    output_file: Union[str, Path] = WEIBO_TRANSLATED_FILE,
):
    """DeepL translate weibo sample (zh -> en)."""
    logger.info("=== Translating weibo zh -> en ===")
    df = pd.read_parquet(weibo_sample_file)
    df["weibo_id"] = df["weibo_id"].astype(str)
    df = _translate_column(df, src_col="weibo_content", target_lang="EN")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    logger.info(f"Saved {len(df)} translated weibo to {output_file}")


def translate_tweets(
    tweet_sample_file: Union[str, Path] = TWEET_SAMPLE_FILE,
    output_file: Union[str, Path] = TWEET_TRANSLATED_FILE,
):
    """DeepL translate tweet sample (en -> zh)."""
    logger.info("=== Translating tweets en -> zh ===")
    df = pd.read_parquet(tweet_sample_file)
    df["id"] = df["id"].astype(str)
    df = _translate_column(df, src_col="text", target_lang="ZH")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    logger.info(f"Saved {len(df)} translated tweets to {output_file}")


# ============================================================
# Re-analyze translated weibo with GPT-5-mini batch
# ============================================================


def submit_weibo_gpt(
    translated_file: Union[str, Path] = WEIBO_TRANSLATED_FILE,
):
    """Build batch requests for translated weibo and submit to OpenAI."""
    logger.info("=== Submitting GPT batch for translated weibo ===")
    df = pd.read_parquet(translated_file)
    df["weibo_id"] = df["weibo_id"].astype(str)

    # Reuse the same hosted prompt + model that scored the original tweets,
    # so any opinion shift on the translated weibo isolates the language/translation
    # effect rather than introducing a new prompt as a confound. We also keep the
    # exact "Tweet text:" framing the prompt was tuned against.
    with jsonlines.open(WEIBO_BATCH_REQUESTS, mode="w") as w:
        for _, row in df.iterrows():
            translated = str(row.get("translated_text", "") or "")
            w.write(
                {
                    "custom_id": str(row["weibo_id"]),
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": BATCH_MODEL,
                        "reasoning": {"effort": "medium"},
                        "text": {"format": {"type": "json_object"}},
                        "prompt": {
                            "id": BATCH_PROMPT_ID,
                            "version": BATCH_PROMPT_VERSION,
                        },
                        "input": f"Please output the results in JSON format. \nTweet text: {translated}",
                    },
                }
            )
    logger.info(f"Wrote {len(df)} requests to {WEIBO_BATCH_REQUESTS}")

    client = OpenAI(api_key=OPENAI_API_KEY)
    with open(WEIBO_BATCH_REQUESTS, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    state = {
        "file_id": uploaded.id,
        "batch_id": batch.id,
        "status": "submitted",
        "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "request_count": len(df),
    }
    with open(WEIBO_BATCH_STATE, "w") as f:
        json.dump(state, f, indent=2)
    logger.info(f"Submitted batch {batch.id} (status={batch.status})")
    print(f"Batch ID: {batch.id}")


def retrieve_weibo_gpt():
    """Poll the GPT batch and download results when ready."""
    logger.info("=== Retrieving GPT batch ===")
    if not WEIBO_BATCH_STATE.exists():
        raise FileNotFoundError(f"No batch state at {WEIBO_BATCH_STATE}")
    with open(WEIBO_BATCH_STATE) as f:
        state = json.load(f)

    client = OpenAI(api_key=OPENAI_API_KEY)
    batch = client.batches.retrieve(state["batch_id"])
    logger.info(f"Status: {batch.status}")

    if batch.status == "completed":
        content = client.files.content(batch.output_file_id)
        with open(WEIBO_BATCH_RESULTS, "wb") as f:
            f.write(content.read())
        state["status"] = "retrieved"
        state["retrieved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(WEIBO_BATCH_STATE, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved results to {WEIBO_BATCH_RESULTS}")
        print("Done. Run parse_weibo_gpt next.")
    elif batch.status == "failed":
        logger.error(f"Batch failed: {batch}")
    else:
        print(f"Not ready yet (status={batch.status}). Re-run later.")


def parse_weibo_gpt(
    output_file: Union[str, Path] = WEIBO_REANALYZED_FILE,
):
    """Parse jsonl batch results into a parquet of (weibo_id, cross_lingual_opinion)."""
    logger.info("=== Parsing GPT batch results ===")
    if not WEIBO_BATCH_RESULTS.exists():
        raise FileNotFoundError(f"Missing {WEIBO_BATCH_RESULTS}")

    rows = []
    with jsonlines.open(WEIBO_BATCH_RESULTS) as r:
        for item in r:
            weibo_id = item["custom_id"]
            opinion = None
            try:
                outputs = item["response"]["body"]["output"]
                for out in outputs:
                    if out.get("type") == "message":
                        text = out["content"][0]["text"]
                        start = text.find("{")
                        end = text.rfind("}") + 1
                        if start != -1 and end > start:
                            opinion = json.loads(text[start:end]).get("opinion")
                        break
            except Exception as e:
                logger.warning(f"Parse failed for {weibo_id}: {e}")
            rows.append({"weibo_id": weibo_id, "cross_lingual_opinion": opinion})

    out_df = pd.DataFrame(rows)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_file, index=False)
    logger.info(f"Parsed {len(out_df)} results -> {output_file}")
    print(out_df["cross_lingual_opinion"].value_counts(dropna=False))


# ============================================================
# Re-analyze translated tweets with Kimi live API
# ============================================================


def _build_kimi_user_prompt(translated_text: str, created_at: Optional[str]) -> str:
    """Match the original Kimi weibo-analysis user-prompt format."""
    parts = [f"微博内容: {translated_text}"]
    if created_at:
        try:
            dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y")
            parts.append(f"发布时间: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        except (ValueError, TypeError):
            pass
    parts.append("是否转发: 否")
    return "\n".join(parts)


def _kimi_score_one(
    client: OpenAI, translated_text: str, created_at: Optional[str], max_retries: int
) -> Optional[object]:
    user_prompt = _build_kimi_user_prompt(translated_text, created_at)
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=KIMI_MODEL,
                messages=[
                    {"role": "system", "content": CHINESE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            txt = resp.choices[0].message.content.strip()
            start = txt.find("{")
            end = txt.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(txt[start:end]).get("opinion")
            return None
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Kimi call failed after {max_retries} retries: {e}")
                return None
            time.sleep((attempt + 1) * 2)
    return None


def analyze_tweet_kimi(
    translated_file: Union[str, Path] = TWEET_TRANSLATED_FILE,
    output_file: Union[str, Path] = TWEET_REANALYZED_FILE,
    delay: float = 0.1,
    max_retries: int = 3,
):
    """Score translated tweets with Kimi live API. Resumable: re-running skips done IDs."""
    logger.info("=== Re-analyzing translated tweets with Kimi ===")
    df = pd.read_parquet(translated_file)
    df["id"] = df["id"].astype(str)

    output_path = Path(output_file)
    if output_path.exists():
        prev = pd.read_parquet(output_path)
        done_ids = set(prev["tweet_id"].astype(str))
        rows = prev.to_dict("records")
        logger.info(f"Resuming: {len(done_ids)} already analyzed")
    else:
        done_ids = set()
        rows = []

    client = OpenAI(api_key=KIMI_API_KEY, base_url=KIMI_BASE_URL)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Kimi analyze"):
        tweet_id = str(row["id"])
        if tweet_id in done_ids:
            continue
        translated = str(row.get("translated_text", "") or "")
        created_at = row.get("createdAt")
        opinion = _kimi_score_one(client, translated, created_at, max_retries)
        rows.append({"tweet_id": tweet_id, "cross_lingual_opinion": opinion})
        done_ids.add(tweet_id)

        if len(rows) % 20 == 0:
            pd.DataFrame(rows).to_parquet(output_path, index=False)
        time.sleep(delay)

    pd.DataFrame(rows).to_parquet(output_path, index=False)
    logger.info(f"Saved {len(rows)} results to {output_path}")


# ============================================================
# Diff analysis
# ============================================================


def _coerce_opinion(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    s = str(x).strip()
    if s in ("", "cannot tell", "None", "nan", "NaN"):
        return None
    try:
        v = int(float(s))
        return v if v in (-2, -1, 0, 1, 2) else None
    except ValueError:
        return None


def _quadratic_kappa(y1, y2, classes=(-2, -1, 0, 1, 2)) -> float:
    n_classes = len(classes)
    cls_index = {c: i for i, c in enumerate(classes)}
    O = np.zeros((n_classes, n_classes))
    for a, b in zip(y1, y2):
        O[cls_index[a], cls_index[b]] += 1
    if O.sum() == 0:
        return 0.0
    W = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = (i - j) ** 2 / (n_classes - 1) ** 2
    hist1 = O.sum(axis=1)
    hist2 = O.sum(axis=0)
    E = np.outer(hist1, hist2) / O.sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return float(1 - (W * O).sum() / den)


def _agreement_metrics(
    df: pd.DataFrame, original_col: str, cross_col: str, label: str
) -> dict:
    df = df.copy()
    df["o"] = df[original_col].apply(_coerce_opinion)
    df["c"] = df[cross_col].apply(_coerce_opinion)
    valid = df[df["o"].notna() & df["c"].notna()].copy()

    n_total = len(df)
    n_valid = len(valid)
    if n_valid == 0:
        return {"label": label, "n_total": n_total, "n_valid": 0, "note": "no valid pairs"}

    valid["o"] = valid["o"].astype(int)
    valid["c"] = valid["c"].astype(int)

    exact = (valid["o"] == valid["c"]).mean()
    within_one = ((valid["o"] - valid["c"]).abs() <= 1).mean()
    mean_abs = (valid["o"] - valid["c"]).abs().mean()
    mean_signed = (valid["c"] - valid["o"]).mean()
    kappa = _quadratic_kappa(valid["o"].tolist(), valid["c"].tolist())

    classes = [-2, -1, 0, 1, 2]
    cm = pd.crosstab(
        pd.Categorical(valid["o"], categories=classes),
        pd.Categorical(valid["c"], categories=classes),
        rownames=["original"],
        colnames=["cross_lingual"],
        dropna=False,
    )

    return {
        "label": label,
        "n_total": n_total,
        "n_valid": n_valid,
        "exact_match": float(exact),
        "within_one": float(within_one),
        "mean_abs_diff": float(mean_abs),
        "mean_signed_diff_cross_minus_orig": float(mean_signed),
        "quadratic_kappa": kappa,
        "confusion_matrix": cm.to_dict(),
    }


def diff(
    weibo_sample_file: Union[str, Path] = WEIBO_SAMPLE_FILE,
    tweet_sample_file: Union[str, Path] = TWEET_SAMPLE_FILE,
):
    """Compute agreement between original and cross-lingual opinions; write report."""
    logger.info("=== Diff analysis ===")

    weibo_orig = pd.read_parquet(weibo_sample_file)
    weibo_orig["weibo_id"] = weibo_orig["weibo_id"].astype(str)
    weibo_cross = pd.read_parquet(WEIBO_REANALYZED_FILE)
    weibo_cross["weibo_id"] = weibo_cross["weibo_id"].astype(str)
    weibo = weibo_orig.merge(weibo_cross, on="weibo_id", how="left")
    weibo_metrics = _agreement_metrics(
        weibo, "original_opinion", "cross_lingual_opinion", "weibo (zh→en, GPT)"
    )

    tweet_orig = pd.read_parquet(tweet_sample_file)
    tweet_orig["id"] = tweet_orig["id"].astype(str)
    tweet_cross = pd.read_parquet(TWEET_REANALYZED_FILE)
    tweet_cross["tweet_id"] = tweet_cross["tweet_id"].astype(str)
    tweet = tweet_orig.merge(
        tweet_cross, left_on="id", right_on="tweet_id", how="left"
    )
    tweet_metrics = _agreement_metrics(
        tweet, "original_opinion", "cross_lingual_opinion", "tweet (en→zh, Kimi)"
    )

    report = {"weibo": weibo_metrics, "tweet": tweet_metrics}
    report_file = CROSS_LINGUAL_PATH / "diff_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {report_file}")

    weibo_out = weibo[
        ["weibo_id", "weibo_content", "original_opinion", "cross_lingual_opinion"]
    ].copy()
    weibo_out["sample"] = "weibo"
    weibo_out = weibo_out.rename(
        columns={"weibo_id": "post_id", "weibo_content": "original_text"}
    )

    tweet_out = tweet[
        ["id", "text", "original_opinion", "cross_lingual_opinion"]
    ].copy()
    tweet_out["sample"] = "tweet"
    tweet_out = tweet_out.rename(columns={"id": "post_id", "text": "original_text"})

    combined = pd.concat([weibo_out, tweet_out], ignore_index=True)
    csv_file = CROSS_LINGUAL_PATH / "diff_per_row.csv"
    combined.to_csv(csv_file, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {csv_file}")

    print("\n=== Cross-lingual agreement ===")
    for m in (weibo_metrics, tweet_metrics):
        print(f"\n[{m['label']}]")
        if m.get("n_valid", 0) == 0:
            print("  no valid pairs")
            continue
        print(f"  n_valid           = {m['n_valid']} / {m['n_total']}")
        print(f"  exact_match       = {m['exact_match']:.3f}")
        print(f"  within ±1         = {m['within_one']:.3f}")
        print(f"  mean |diff|       = {m['mean_abs_diff']:.3f}")
        print(f"  mean (cross-orig) = {m['mean_signed_diff_cross_minus_orig']:+.3f}")
        print(f"  quadratic kappa   = {m['quadratic_kappa']:.3f}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    fire.Fire(
        {
            "sample_tweets": sample_tweets,
            "translate_weibo": translate_weibo,
            "translate_tweets": translate_tweets,
            "submit_weibo_gpt": submit_weibo_gpt,
            "retrieve_weibo_gpt": retrieve_weibo_gpt,
            "parse_weibo_gpt": parse_weibo_gpt,
            "analyze_tweet_kimi": analyze_tweet_kimi,
            "diff": diff,
        }
    )
