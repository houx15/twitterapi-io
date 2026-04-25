"""
Cross-lingual sentiment validation.

Tests whether using different LLMs / prompts in different languages biased the
original sentiment scores. For each sample post we:
  1. take the existing opinion (from the original analyzer)
  2. translate the text to the OTHER language via DeepL
  3. re-score with the OTHER analyzer (matching its original input format)
  4. compare original vs cross-lingual scores

Workflow:
    Single-shot (one sample, one translation):
        sample_tweets          - stratified-by-opinion sample of 200 tweets
        translate_weibo        - DeepL zh -> en for the weibo sample
        translate_tweets       - DeepL en -> zh for the tweet sample
        sample_translation     - spot-check 10 weibo + 10 tweet translations

    Per-round (5× by default — same translated text, repeat LLM scoring):
        submit_weibo_gpt   --round N    - submit GPT batch
        retrieve_weibo_gpt --round N    - poll & download batch results
        parse_weibo_gpt    --round N    - parse jsonl results into parquet
        analyze_tweet_kimi --round N    - Kimi live (resumable)
        diff               --round N    - per-round agreement metrics

    Orchestrators:
        run_scoring --rounds 5  - per-round submit_gpt + kimi (stops after submit)
        finalize    --rounds 5  - per-round retrieve/parse/diff, then aggregate
        aggregate   --rounds 5  - mean/std + per-post LLM stability across rounds

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
DEFAULT_ROUNDS = 5

# Shared across rounds — sampling and translation happen ONCE; only the LLM
# re-scoring + per-round diff are repeated, so we can measure LLM stability on
# the same translated text rather than mixing sampling variance into the result.
WEIBO_SAMPLE_FILE = CROSS_LINGUAL_PATH / "weibo_translation_sample.parquet"
TWEET_SAMPLE_FILE = CROSS_LINGUAL_PATH / "tweet_translation_sample.parquet"
WEIBO_TRANSLATED_FILE = CROSS_LINGUAL_PATH / "weibo_translated.parquet"
TWEET_TRANSLATED_FILE = CROSS_LINGUAL_PATH / "tweet_translated.parquet"
TRANSLATION_SAMPLE_CSV = CROSS_LINGUAL_PATH / "translation_sample.csv"
DEEPL_CACHE_FILE = CROSS_LINGUAL_PATH / "deepl_cache.json"
AGGREGATE_REPORT_FILE = CROSS_LINGUAL_PATH / "aggregate_report.json"
AGGREGATE_PER_ROW_FILE = CROSS_LINGUAL_PATH / "aggregate_per_row.csv"


def _round_paths(round: int) -> dict:
    """Per-round paths for LLM scoring + per-round diff (under round_<N>/)."""
    d = CROSS_LINGUAL_PATH / f"round_{round}"
    d.mkdir(parents=True, exist_ok=True)
    return {
        "weibo_reanalyzed": d / "weibo_reanalyzed_gpt.parquet",
        "tweet_reanalyzed": d / "tweet_reanalyzed_kimi.parquet",
        "weibo_batch_requests": d / "weibo_en_requests.jsonl",
        "weibo_batch_results": d / "weibo_en_results.jsonl",
        "weibo_batch_state": d / "weibo_en_batch_state.json",
        "diff_report": d / "diff_report.json",
        "diff_per_row": d / "diff_per_row.csv",
    }

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
):
    """Sample tweets stratified by their existing GPT opinion score (single shot)."""
    output_path = TWEET_SAMPLE_FILE
    logger.info(f"=== Sampling tweets [seed {seed}] ===")

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


def translate_weibo():
    """DeepL translate weibo sample (zh -> en)."""
    logger.info("=== Translating weibo zh -> en ===")
    df = pd.read_parquet(WEIBO_SAMPLE_FILE)
    df["weibo_id"] = df["weibo_id"].astype(str)
    df = _translate_column(df, src_col="weibo_content", target_lang="EN-US")
    df.to_parquet(WEIBO_TRANSLATED_FILE, index=False)
    logger.info(f"Saved {len(df)} translated weibo to {WEIBO_TRANSLATED_FILE}")


def translate_tweets():
    """DeepL translate tweet sample (en -> zh)."""
    logger.info("=== Translating tweets en -> zh ===")
    df = pd.read_parquet(TWEET_SAMPLE_FILE)
    df["id"] = df["id"].astype(str)
    df = _translate_column(df, src_col="text", target_lang="ZH")
    df.to_parquet(TWEET_TRANSLATED_FILE, index=False)
    logger.info(f"Saved {len(df)} translated tweets to {TWEET_TRANSLATED_FILE}")


def sample_translation(
    n: int = 10,
    seed: int = DEFAULT_SEED,
):
    """Spot-check translations: print n weibo (zh→en) and n tweet (en→zh) pairs,
    and save them as a CSV for review."""
    logger.info("=== Sampling translations for spot-check ===")
    weibo_df = pd.read_parquet(WEIBO_TRANSLATED_FILE)
    tweet_df = pd.read_parquet(TWEET_TRANSLATED_FILE)

    weibo_sample = weibo_df.sample(
        n=min(n, len(weibo_df)), random_state=seed
    ).reset_index(drop=True)
    tweet_sample = tweet_df.sample(
        n=min(n, len(tweet_df)), random_state=seed
    ).reset_index(drop=True)

    print("\n" + "=" * 80)
    print(f"WEIBO (zh → en)  — {len(weibo_sample)} samples")
    print("=" * 80)
    for i, row in weibo_sample.iterrows():
        print(
            f"\n--- {i + 1}/{len(weibo_sample)}  "
            f"weibo_id={row['weibo_id']}  opinion={row['original_opinion']} ---"
        )
        print(f"ZH: {row['weibo_content']}")
        print(f"EN: {row['translated_text']}")

    print("\n" + "=" * 80)
    print(f"TWEET (en → zh)  — {len(tweet_sample)} samples")
    print("=" * 80)
    for i, row in tweet_sample.iterrows():
        print(
            f"\n--- {i + 1}/{len(tweet_sample)}  "
            f"tweet_id={row['id']}  opinion={row['original_opinion']} ---"
        )
        print(f"EN: {row['text']}")
        print(f"ZH: {row['translated_text']}")

    weibo_out = weibo_sample[
        ["weibo_id", "weibo_content", "translated_text", "original_opinion"]
    ].copy()
    weibo_out["sample"] = "weibo (zh→en)"
    weibo_out = weibo_out.rename(
        columns={"weibo_id": "post_id", "weibo_content": "source_text"}
    )

    tweet_out = tweet_sample[
        ["id", "text", "translated_text", "original_opinion"]
    ].copy()
    tweet_out["sample"] = "tweet (en→zh)"
    tweet_out = tweet_out.rename(columns={"id": "post_id", "text": "source_text"})

    combined = pd.concat([weibo_out, tweet_out], ignore_index=True)[
        ["sample", "post_id", "source_text", "translated_text", "original_opinion"]
    ]

    combined.to_csv(TRANSLATION_SAMPLE_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved spot-check to: {TRANSLATION_SAMPLE_CSV}")


# ============================================================
# Re-analyze translated weibo with GPT-5-mini batch
# ============================================================


def submit_weibo_gpt(round: int = 1):
    """Build batch requests for translated weibo and submit to OpenAI."""
    paths = _round_paths(round)
    logger.info(f"=== Submitting GPT batch for translated weibo [round {round}] ===")
    df = pd.read_parquet(WEIBO_TRANSLATED_FILE)
    df["weibo_id"] = df["weibo_id"].astype(str)

    # Reuse the same hosted prompt + model that scored the original tweets,
    # so any opinion shift on the translated weibo isolates the language/translation
    # effect rather than introducing a new prompt as a confound. We also keep the
    # exact "Tweet text:" framing the prompt was tuned against.
    with jsonlines.open(paths["weibo_batch_requests"], mode="w") as w:
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
    logger.info(f"Wrote {len(df)} requests to {paths['weibo_batch_requests']}")

    client = OpenAI(api_key=OPENAI_API_KEY)
    with open(paths["weibo_batch_requests"], "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    state = {
        "round": round,
        "file_id": uploaded.id,
        "batch_id": batch.id,
        "status": "submitted",
        "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "request_count": len(df),
    }
    with open(paths["weibo_batch_state"], "w") as f:
        json.dump(state, f, indent=2)
    logger.info(f"Submitted batch {batch.id} (status={batch.status})")
    print(f"Round {round} — Batch ID: {batch.id}")


def retrieve_weibo_gpt(round: int = 1) -> bool:
    """Poll the GPT batch and download results when ready. Returns True iff completed."""
    paths = _round_paths(round)
    logger.info(f"=== Retrieving GPT batch [round {round}] ===")
    if not paths["weibo_batch_state"].exists():
        raise FileNotFoundError(f"No batch state at {paths['weibo_batch_state']}")
    with open(paths["weibo_batch_state"]) as f:
        state = json.load(f)

    client = OpenAI(api_key=OPENAI_API_KEY)
    batch = client.batches.retrieve(state["batch_id"])
    logger.info(f"Round {round} status: {batch.status}")

    if batch.status == "completed":
        content = client.files.content(batch.output_file_id)
        with open(paths["weibo_batch_results"], "wb") as f:
            f.write(content.read())
        state["status"] = "retrieved"
        state["retrieved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(paths["weibo_batch_state"], "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved results to {paths['weibo_batch_results']}")
        print(f"Round {round}: done. Run parse_weibo_gpt next.")
        return True
    elif batch.status == "failed":
        logger.error(f"Round {round} batch failed: {batch}")
        return False
    else:
        print(f"Round {round}: not ready yet (status={batch.status}). Re-run later.")
        return False


def _opinion_to_str(x) -> Optional[str]:
    """Stringify the opinion field uniformly. The prompt allows both integer
    labels (-2..2) and the literal "cannot tell", so writing a mixed-dtype
    column blows up pyarrow. Reading back through `_coerce_opinion` recovers
    the int / None for the metrics step.
    """
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    return str(x)


def parse_weibo_gpt(round: int = 1):
    """Parse jsonl batch results into a parquet of (weibo_id, cross_lingual_opinion)."""
    paths = _round_paths(round)
    logger.info(f"=== Parsing GPT batch results [round {round}] ===")
    if not paths["weibo_batch_results"].exists():
        raise FileNotFoundError(f"Missing {paths['weibo_batch_results']}")

    rows = []
    with jsonlines.open(paths["weibo_batch_results"]) as r:
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
            rows.append(
                {"weibo_id": weibo_id, "cross_lingual_opinion": _opinion_to_str(opinion)}
            )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(paths["weibo_reanalyzed"], index=False)
    logger.info(f"Parsed {len(out_df)} results -> {paths['weibo_reanalyzed']}")
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
    round: int = 1,
    delay: float = 0.1,
    max_retries: int = 3,
):
    """Score translated tweets with Kimi live API. Resumable: re-running skips done IDs."""
    paths = _round_paths(round)
    logger.info(f"=== Re-analyzing translated tweets with Kimi [round {round}] ===")
    df = pd.read_parquet(TWEET_TRANSLATED_FILE)
    df["id"] = df["id"].astype(str)

    output_path = paths["tweet_reanalyzed"]
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
        rows.append(
            {"tweet_id": tweet_id, "cross_lingual_opinion": _opinion_to_str(opinion)}
        )
        done_ids.add(tweet_id)

        if len(rows) % 20 == 0:
            pd.DataFrame(rows).to_parquet(output_path, index=False)
        time.sleep(delay)

    pd.DataFrame(rows).to_parquet(output_path, index=False)
    logger.info(f"Saved {len(rows)} results to {output_path}")


# ============================================================
# Diff analysis
# ============================================================


def _classify_opinion(x) -> str:
    """Classify a raw opinion value as 'valid_int', 'cannot_tell', or 'invalid'.

    Diff metrics are only computed on rows where both sides are 'valid_int'.
    The 'cannot_tell' bucket is reported separately as a proportion so we can
    distinguish "the analyzer punted" from "the analyzer disagreed".
    """
    if x is None:
        return "invalid"
    if isinstance(x, float) and pd.isna(x):
        return "invalid"
    s = str(x).strip().lower()
    if s == "cannot tell":
        return "cannot_tell"
    try:
        v = int(float(s))
        return "valid_int" if v in (-2, -1, 0, 1, 2) else "invalid"
    except ValueError:
        return "invalid"


def _coerce_opinion(x) -> Optional[int]:
    if _classify_opinion(x) != "valid_int":
        return None
    return int(float(str(x).strip()))


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


def _breakdown(df: pd.DataFrame, col: str, n_total: int) -> dict:
    counts = df[col].value_counts().to_dict()
    return {
        "valid_int": int(counts.get("valid_int", 0)),
        "cannot_tell": int(counts.get("cannot_tell", 0)),
        "invalid": int(counts.get("invalid", 0)),
        "cannot_tell_proportion": (
            float(counts.get("cannot_tell", 0) / n_total) if n_total else 0.0
        ),
    }


def _agreement_metrics(
    df: pd.DataFrame, original_col: str, cross_col: str, label: str
) -> dict:
    df = df.copy()
    df["o_cls"] = df[original_col].apply(_classify_opinion)
    df["c_cls"] = df[cross_col].apply(_classify_opinion)

    n_total = len(df)
    original_breakdown = _breakdown(df, "o_cls", n_total)
    cross_breakdown = _breakdown(df, "c_cls", n_total)

    valid = df[(df["o_cls"] == "valid_int") & (df["c_cls"] == "valid_int")].copy()
    n_valid_pairs = len(valid)

    base = {
        "label": label,
        "n_total": n_total,
        "original_breakdown": original_breakdown,
        "cross_breakdown": cross_breakdown,
        "n_valid_pairs": n_valid_pairs,
    }
    if n_valid_pairs == 0:
        return {**base, "note": "no valid pairs — cannot compute diff metrics"}

    valid["o"] = valid[original_col].apply(_coerce_opinion).astype(int)
    valid["c"] = valid[cross_col].apply(_coerce_opinion).astype(int)

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
        **base,
        "exact_match": float(exact),
        "within_one": float(within_one),
        "mean_abs_diff": float(mean_abs),
        "mean_signed_diff_cross_minus_orig": float(mean_signed),
        "quadratic_kappa": kappa,
        "confusion_matrix": cm.to_dict(),
    }


def diff(round: int = 1):
    """Compute agreement between original and cross-lingual opinions; write per-round report."""
    paths = _round_paths(round)
    logger.info(f"=== Diff analysis [round {round}] ===")

    weibo_orig = pd.read_parquet(WEIBO_SAMPLE_FILE)
    weibo_orig["weibo_id"] = weibo_orig["weibo_id"].astype(str)
    weibo_cross = pd.read_parquet(paths["weibo_reanalyzed"])
    weibo_cross["weibo_id"] = weibo_cross["weibo_id"].astype(str)
    weibo = weibo_orig.merge(weibo_cross, on="weibo_id", how="left")
    weibo_metrics = _agreement_metrics(
        weibo, "original_opinion", "cross_lingual_opinion", "weibo (zh→en, GPT)"
    )

    tweet_orig = pd.read_parquet(TWEET_SAMPLE_FILE)
    tweet_orig["id"] = tweet_orig["id"].astype(str)
    tweet_cross = pd.read_parquet(paths["tweet_reanalyzed"])
    tweet_cross["tweet_id"] = tweet_cross["tweet_id"].astype(str)
    tweet = tweet_orig.merge(
        tweet_cross, left_on="id", right_on="tweet_id", how="left"
    )
    tweet_metrics = _agreement_metrics(
        tweet, "original_opinion", "cross_lingual_opinion", "tweet (en→zh, Kimi)"
    )

    report = {"round": round, "weibo": weibo_metrics, "tweet": tweet_metrics}
    with open(paths["diff_report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {paths['diff_report']}")

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
    combined.to_csv(paths["diff_per_row"], index=False, encoding="utf-8-sig")
    logger.info(f"Saved {paths['diff_per_row']}")

    print(f"\n=== Cross-lingual agreement (round {round}) ===")
    for m in (weibo_metrics, tweet_metrics):
        print(f"\n[{m['label']}]")
        n_total = m["n_total"]
        ob = m["original_breakdown"]
        cb = m["cross_breakdown"]

        def _pct(n: int) -> str:
            return f"{n / n_total * 100:.1f}%" if n_total else "0.0%"

        print(f"  n_total                     = {n_total}")
        print(
            f"  original cannot_tell        = {ob['cannot_tell']:>4} ({_pct(ob['cannot_tell'])})"
        )
        print(
            f"  original other invalid      = {ob['invalid']:>4} ({_pct(ob['invalid'])})"
        )
        print(
            f"  cross cannot_tell           = {cb['cannot_tell']:>4} ({_pct(cb['cannot_tell'])})"
        )
        print(
            f"  cross other invalid         = {cb['invalid']:>4} ({_pct(cb['invalid'])})"
        )
        print(
            f"  pairs used for diff metrics = {m['n_valid_pairs']:>4} ({_pct(m['n_valid_pairs'])})"
        )
        if m["n_valid_pairs"] == 0:
            print("  no valid pairs — cannot compute diff metrics")
            continue
        print(f"  exact_match                 = {m['exact_match']:.3f}")
        print(f"  within ±1                   = {m['within_one']:.3f}")
        print(f"  mean |diff|                 = {m['mean_abs_diff']:.3f}")
        print(
            f"  mean (cross-orig)           = {m['mean_signed_diff_cross_minus_orig']:+.3f}"
        )
        print(f"  quadratic kappa             = {m['quadratic_kappa']:.3f}")


# ============================================================
# Multi-round aggregation + phase orchestrators
# ============================================================


def _summarize_metrics(metrics_list: list) -> dict:
    """Mean + std (ddof=1) across rounds for the numeric fields in _agreement_metrics."""
    out = {"n_rounds": len(metrics_list), "rounds": metrics_list, "summary": {}}
    if not metrics_list:
        return out

    for k in (
        "n_total",
        "n_valid_pairs",
        "exact_match",
        "within_one",
        "mean_abs_diff",
        "mean_signed_diff_cross_minus_orig",
        "quadratic_kappa",
    ):
        vals = [m[k] for m in metrics_list if k in m]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        out["summary"][f"mean_{k}"] = float(arr.mean())
        out["summary"][f"std_{k}"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    for side, key in [
        ("cross_breakdown", "cross_cannot_tell_proportion"),
        ("original_breakdown", "original_cannot_tell_proportion"),
    ]:
        vals = [m[side]["cannot_tell_proportion"] for m in metrics_list if side in m]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        out["summary"][f"mean_{key}"] = float(arr.mean())
        out["summary"][f"std_{key}"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    return out


def _per_post_stability(
    rounds_in: list[int],
    reanalyzed_key: str,
    id_col: str,
    sample_path: Path,
) -> dict:
    """For each post, gather the per-round cross-lingual opinions and summarize
    LLM stability (how often the analyzer gave the same label across rounds on
    the SAME translated text). Also returns a per-post wide DataFrame with one
    column per round, plus modal/all_agree/within_one_across_rounds columns.
    """
    sample_df = pd.read_parquet(sample_path)
    sample_df[id_col] = sample_df[id_col].astype(str)

    wide = sample_df[[id_col, "original_opinion"]].copy()
    for r in rounds_in:
        rdf = pd.read_parquet(_round_paths(r)[reanalyzed_key])
        # Reanalyzed parquets use weibo_id (GPT) or tweet_id (Kimi)
        join_col = "weibo_id" if reanalyzed_key == "weibo_reanalyzed" else "tweet_id"
        rdf[join_col] = rdf[join_col].astype(str)
        rdf = rdf.rename(columns={"cross_lingual_opinion": f"opinion_round_{r}"})
        wide = wide.merge(
            rdf[[join_col, f"opinion_round_{r}"]],
            left_on=id_col,
            right_on=join_col,
            how="left",
        )
        if join_col != id_col:
            wide = wide.drop(columns=[join_col])

    round_cols = [f"opinion_round_{r}" for r in rounds_in]

    def _row_stats(row):
        ints = [_coerce_opinion(row[c]) for c in round_cols]
        valid = [v for v in ints if v is not None]
        n_runs = len(rounds_in)
        n_valid = len(valid)
        if n_valid == 0:
            return pd.Series(
                {
                    "n_valid_runs": 0,
                    "modal_opinion": None,
                    "modal_count": 0,
                    "all_agree": False,
                    "within_one_across_rounds": False,
                }
            )
        from collections import Counter
        counter = Counter(valid)
        mode_val, mode_cnt = counter.most_common(1)[0]
        return pd.Series(
            {
                "n_valid_runs": n_valid,
                "modal_opinion": mode_val,
                "modal_count": mode_cnt,
                "all_agree": (n_valid == n_runs) and len(set(valid)) == 1,
                "within_one_across_rounds": (n_valid == n_runs)
                and (max(valid) - min(valid)) <= 1,
            }
        )

    stats = wide.apply(_row_stats, axis=1)
    wide = pd.concat([wide, stats], axis=1)

    n_total = len(wide)
    n_runs = len(rounds_in)
    n_all_valid = int((wide["n_valid_runs"] == n_runs).sum())
    n_unanimous = int(wide["all_agree"].sum())
    n_within_one = int(wide["within_one_across_rounds"].sum())

    modal_diff_metrics = None
    if n_all_valid > 0:
        eligible = wide[wide["n_valid_runs"] == n_runs].copy()
        eligible["cross_lingual_opinion"] = eligible["modal_opinion"].astype("Int64")
        modal_diff_metrics = _agreement_metrics(
            eligible,
            "original_opinion",
            "cross_lingual_opinion",
            "modal-vote diff",
        )

    return {
        "n_posts": int(n_total),
        "n_rounds": n_runs,
        "n_posts_all_runs_valid": n_all_valid,
        "proportion_all_runs_valid": float(n_all_valid / n_total) if n_total else 0.0,
        "n_posts_unanimous": n_unanimous,
        "proportion_unanimous": float(n_unanimous / n_total) if n_total else 0.0,
        "n_posts_within_one_across_rounds": n_within_one,
        "proportion_within_one_across_rounds": (
            float(n_within_one / n_total) if n_total else 0.0
        ),
        "mean_modal_count": float(wide["modal_count"].mean()),
        "modal_vote_diff": modal_diff_metrics,
        "_wide": wide,  # caller drops this before serializing
    }


def aggregate(rounds: int = DEFAULT_ROUNDS, start_round: int = 1):
    """Pool per-round diffs into a mean ± std report and per-post LLM stability."""
    logger.info(f"=== Aggregating rounds {start_round}..{rounds} ===")
    weibo_per_round, tweet_per_round, diff_rows, found = [], [], [], []
    for r in range(start_round, rounds + 1):
        paths = _round_paths(r)
        if not paths["diff_report"].exists():
            logger.warning(f"Round {r}: missing {paths['diff_report']}, skipping")
            continue
        found.append(r)
        with open(paths["diff_report"], encoding="utf-8") as f:
            data = json.load(f)
        weibo_per_round.append({"round": r, **data["weibo"]})
        tweet_per_round.append({"round": r, **data["tweet"]})
        if paths["diff_per_row"].exists():
            df = pd.read_csv(paths["diff_per_row"])
            df["round"] = r
            diff_rows.append(df)

    if not found:
        logger.error("No diff reports found; cannot aggregate")
        return

    weibo_stab = _per_post_stability(
        found, "weibo_reanalyzed", "weibo_id", WEIBO_SAMPLE_FILE
    )
    tweet_stab = _per_post_stability(
        found, "tweet_reanalyzed", "id", TWEET_SAMPLE_FILE
    )

    aggregated = {
        "n_rounds_requested": rounds - start_round + 1,
        "rounds_with_reports": found,
        "weibo": {
            **_summarize_metrics(weibo_per_round),
            "per_post_stability": {k: v for k, v in weibo_stab.items() if k != "_wide"},
        },
        "tweet": {
            **_summarize_metrics(tweet_per_round),
            "per_post_stability": {k: v for k, v in tweet_stab.items() if k != "_wide"},
        },
    }
    with open(AGGREGATE_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {AGGREGATE_REPORT_FILE}")

    weibo_wide = weibo_stab["_wide"].copy()
    weibo_wide["sample"] = "weibo"
    weibo_wide = weibo_wide.rename(columns={"weibo_id": "post_id"})
    tweet_wide = tweet_stab["_wide"].copy()
    tweet_wide["sample"] = "tweet"
    tweet_wide = tweet_wide.rename(columns={"id": "post_id"})
    pd.concat([weibo_wide, tweet_wide], ignore_index=True).to_csv(
        AGGREGATE_PER_ROW_FILE, index=False, encoding="utf-8-sig"
    )
    logger.info(f"Saved {AGGREGATE_PER_ROW_FILE}")

    print(f"\n=== Aggregate cross-lingual agreement ({len(found)} rounds: {found}) ===")
    for side in ("weibo", "tweet"):
        s = aggregated[side]
        sm = s["summary"]
        st = s["per_post_stability"]
        print(f"\n[{side}]")
        if not sm:
            print("  (no rounds)")
            continue

        def _line(label: str, mean_key: str, signed: bool = False):
            mean = sm.get(mean_key)
            std = sm.get("std_" + mean_key[5:])
            if mean is None:
                return
            std_str = f" ± {std:.3f}" if std is not None else ""
            fmt = f"{mean:+.3f}" if signed else f"{mean:.3f}"
            print(f"  {label:<37} = {fmt}{std_str}")

        print("  -- per-round diff (mean ± std across rounds) --")
        _line("n_valid_pairs (mean)", "mean_n_valid_pairs")
        _line("cross cannot_tell proportion", "mean_cross_cannot_tell_proportion")
        _line("original cannot_tell proportion", "mean_original_cannot_tell_proportion")
        _line("exact_match", "mean_exact_match")
        _line("within ±1", "mean_within_one")
        _line("mean |diff|", "mean_mean_abs_diff")
        _line(
            "mean (cross-orig)",
            "mean_mean_signed_diff_cross_minus_orig",
            signed=True,
        )
        _line("quadratic kappa", "mean_quadratic_kappa")

        print("  -- LLM stability across rounds (same translated text) --")
        print(
            f"  posts with all {st['n_rounds']} runs valid     "
            f"= {st['n_posts_all_runs_valid']:>4} / {st['n_posts']} "
            f"({st['proportion_all_runs_valid'] * 100:.1f}%)"
        )
        print(
            f"  posts with unanimous label across runs = "
            f"{st['n_posts_unanimous']:>4} / {st['n_posts']} "
            f"({st['proportion_unanimous'] * 100:.1f}%)"
        )
        print(
            f"  posts within ±1 across all runs        = "
            f"{st['n_posts_within_one_across_rounds']:>4} / {st['n_posts']} "
            f"({st['proportion_within_one_across_rounds'] * 100:.1f}%)"
        )
        print(
            f"  mean modal-count (out of {st['n_rounds']})            "
            f"= {st['mean_modal_count']:.2f}"
        )
        mvd = st.get("modal_vote_diff")
        if mvd and "exact_match" in mvd:
            print(
                f"  modal-vote vs original: exact={mvd['exact_match']:.3f}, "
                f"signed_diff={mvd['mean_signed_diff_cross_minus_orig']:+.3f}, "
                f"kappa={mvd['quadratic_kappa']:.3f}"
            )


def run_scoring(rounds: int = DEFAULT_ROUNDS, start_round: int = 1):
    """For each round in [start_round..rounds]: submit a GPT batch on the
    translated weibo, run Kimi on the translated tweets. Sample + translation
    must already exist (one-shot, top-level). Stops after Kimi finishes — wait
    for GPT batches to complete, then run `finalize`.
    """
    for f in (WEIBO_TRANSLATED_FILE, TWEET_TRANSLATED_FILE):
        if not f.exists():
            raise FileNotFoundError(
                f"Missing {f}. Run sample_tweets/translate_weibo/translate_tweets first."
            )

    for r in range(start_round, rounds + 1):
        logger.info(f"\n========== ROUND {r} (scoring) ==========")
        paths = _round_paths(r)

        if paths["weibo_batch_state"].exists():
            logger.info(f"Round {r}: GPT batch already submitted, skipping")
        else:
            submit_weibo_gpt(round=r)

        # analyze_tweet_kimi is internally resumable
        analyze_tweet_kimi(round=r)

    print("\n" + "=" * 60)
    print(f"Scoring done for rounds {start_round}..{rounds}.")
    print("GPT batches are queued; once they complete, run:")
    print(f"  python cross_lingual_validation.py finalize --rounds {rounds}")
    print("=" * 60)


def finalize(rounds: int = DEFAULT_ROUNDS, start_round: int = 1):
    """Retrieve GPT batches, parse, diff per round, then aggregate. Idempotent:
    rounds whose batches aren't ready yet are skipped — re-run later."""
    completed = []
    for r in range(start_round, rounds + 1):
        logger.info(f"\n========== ROUND {r} (finalize) ==========")
        paths = _round_paths(r)

        if not paths["weibo_batch_results"].exists():
            try:
                ready = retrieve_weibo_gpt(round=r)
            except Exception as e:
                logger.warning(f"Round {r}: retrieve_weibo_gpt error: {e}")
                ready = False
            if not ready:
                logger.warning(f"Round {r}: batch not ready, skipping")
                continue

        if not paths["weibo_reanalyzed"].exists():
            parse_weibo_gpt(round=r)

        if not paths["tweet_reanalyzed"].exists():
            logger.warning(
                f"Round {r}: tweet_reanalyzed missing (Kimi never finished?), "
                "skipping diff"
            )
            continue

        diff(round=r)
        completed.append(r)

    if not completed:
        print("\nNo rounds had completed batches; re-run `finalize` later.")
        return

    print(f"\nFinalized rounds: {completed}. Aggregating...")
    aggregate(rounds=rounds, start_round=start_round)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    fire.Fire(
        {
            "sample_tweets": sample_tweets,
            "translate_weibo": translate_weibo,
            "translate_tweets": translate_tweets,
            "sample_translation": sample_translation,
            "submit_weibo_gpt": submit_weibo_gpt,
            "retrieve_weibo_gpt": retrieve_weibo_gpt,
            "parse_weibo_gpt": parse_weibo_gpt,
            "analyze_tweet_kimi": analyze_tweet_kimi,
            "diff": diff,
            "aggregate": aggregate,
            "run_scoring": run_scoring,
            "finalize": finalize,
        }
    )
