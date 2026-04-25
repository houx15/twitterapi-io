# Cross-lingual sentiment validation — US side

Reviewer concern: original weibo sentiment used Kimi + Chinese prompt; tweet sentiment used GPT + English prompt. Different LLMs and prompts could bias the comparison.

This runbook covers the **US side**: sampling 200 tweets, translating both samples via DeepL, re-scoring each post in the *other* language with the *other* LLM, and computing agreement metrics. The China side (sampling 200 weibos) is in `youth-analysis/CROSS_LINGUAL_VALIDATION.md`; finish that first and `scp` the resulting parquet here.

## Experiment design

For each sample post we:

1. Take the existing opinion score (from the original analyzer)
2. Translate the text to the *other* language via DeepL
3. Re-score with the *other* analyzer, **using the same prompt that analyzer originally used**
4. Compare original vs cross-lingual scores

**Translated weibo → GPT reanalysis** reuses the existing `BATCH_PROMPT_ID` / `BATCH_PROMPT_VERSION` / `BATCH_MODEL` — i.e. the exact prompt that originally scored the tweets. Same `Tweet text:` framing, same `/v1/responses` endpoint, same `reasoning.effort=medium`. The only thing that differs from the original tweet run is the content (translated weibo).

**Translated tweet → Kimi reanalysis** reuses the verbatim Chinese system prompt from `youth-analysis/ai_sentiment_analyzer.py`, with the same `微博内容 / 发布时间 / 是否转发` user-message format. `是否转发` is hardcoded to `否` (tweets don't have weibo's `//` retweet semantic) and `发布时间` is parsed from Twitter's `createdAt`. These two metadata fields are constants/derived rather than per-row signals — minor caveat.

This holds the prompt content constant across languages so any score shift can be attributed to the language switch + translation, not to a different prompt.

## Setup

### 1. Add config keys

Open `config.py` and add the keys below (template lives in `config.date.example.py`):

```python
DEEPL_API_KEY = "..."   # free vs pro endpoint auto-selected from key suffix
KIMI_API_KEY = "..."
KIMI_BASE_URL = "https://api.moonshot.cn/v1"
KIMI_MODEL = "kimi-k2-0905-preview"
CROSS_LINGUAL_DIR = "sentiment_results/cross_lingual"
```

Translation goes through the official [`deepl`](https://pypi.org/project/deepl/) Python SDK (added to `requirements.txt`). The SDK auto-selects the free vs pro endpoint based on the key, so no URL config is needed.

The reanalysis of translated weibo reuses `BATCH_PROMPT_ID` / `BATCH_PROMPT_VERSION` / `BATCH_MODEL` already in `config.py` — no changes there.

### 2. Refresh dependencies

```bash
pip install -r requirements.txt
```

(`numpy` was added — it's used for the quadratic-weighted Cohen's κ.)

### 3. Drop the weibo sample in place

```bash
mkdir -p sentiment_results/cross_lingual
# copy from China server:
# scp <china-server>:<path>/ai_attitudes/weibo_translation_sample.parquet \
#     sentiment_results/cross_lingual/
ls sentiment_results/cross_lingual/weibo_translation_sample.parquet
```

## Pipeline

All commands run from the `twitterapi-io` directory. Outputs land in `CROSS_LINGUAL_DIR`.

```bash
# 1. Sample 200 tweets stratified by their existing GPT opinion
python cross_lingual_validation.py sample_tweets

# 2. DeepL translation (cached on disk → safe to re-run)
python cross_lingual_validation.py translate_weibo     # zh → en
python cross_lingual_validation.py translate_tweets    # en → zh

# 2b. Optional: spot-check 10 weibo + 10 tweet translations before re-scoring
python cross_lingual_validation.py sample_translation              # default n=10
python cross_lingual_validation.py sample_translation --n 20       # or larger

# 3. Re-score translated weibo with GPT batch (Responses API, hosted prompt)
python cross_lingual_validation.py submit_weibo_gpt
# poll until status=completed (re-run as needed):
python cross_lingual_validation.py retrieve_weibo_gpt
python cross_lingual_validation.py parse_weibo_gpt

# 4. Re-score translated tweets with Kimi live API (resumable)
python cross_lingual_validation.py analyze_tweet_kimi

# 5. Diff
python cross_lingual_validation.py diff
```

### Per-step notes

- **`sample_tweets`** — reads `sentiment_results/batch_results.parquet` and joins to the date-filtered tweet parquets via `load_parquet_files_filtered_by_date`. Proportional-stratified by class.
- **`translate_*`** — caches every (target_lang, text) → translation in `deepl_cache.json`, persisted every 25 calls. Safe to interrupt and re-run.
- **`sample_translation`** — prints `n` random pairs from each direction (default 10) and writes `translation_sample.csv` to `CROSS_LINGUAL_DIR`. Use this to eyeball whether DeepL handled AI keywords / sarcasm / quoted text reasonably before committing API spend on the re-scoring steps.
- **`submit_weibo_gpt`** — uploads requests, creates batch, persists `{file_id, batch_id, status}` to `weibo_en_batch_state.json`.
- **`retrieve_weibo_gpt`** — idempotent. Re-run until status=`completed`; results land in `weibo_en_results.jsonl`.
- **`parse_weibo_gpt`** — extracts `opinion` from each Responses API output and writes `weibo_reanalyzed_gpt.parquet`.
- **`analyze_tweet_kimi`** — Kimi live, ~0.1s delay between calls, 3 retries with exponential backoff, periodic save every 20 rows. Re-run after a crash and it skips IDs already in `tweet_reanalyzed_kimi.parquet`.
- **`diff`** — joins original vs cross-lingual opinions, writes `diff_report.json` (metrics + confusion matrix per sample) and `diff_per_row.csv` (per-post detail for both samples).

## Reading the report

`diff_report.json` has two top-level keys: `weibo` (zh→en, GPT) and `tweet` (en→zh, Kimi). Each contains:

- `n_total` / `n_valid` — pairs available vs pairs with valid integer opinion on both sides
- `exact_match` — fraction with identical labels
- `within_one` — fraction within ±1 on the ordinal scale
- `mean_abs_diff` — mean |original − cross|
- `mean_signed_diff_cross_minus_orig` — direction of bias. Positive = cross-lingual analyzer scored more positively than the original.
- `quadratic_kappa` — Cohen's κ adjusted for the ordinal nature of the labels
- `confusion_matrix` — 5×5 of original (rows) vs cross-lingual (cols)

Three numbers worth quoting in the response to the reviewer: `exact_match`, `mean_signed_diff_cross_minus_orig`, `quadratic_kappa`. If signed diff is small (|·| < 0.2) and κ > 0.6 on both samples, you have a clean defense.

## Spot-check before trusting the metrics

Open `diff_per_row.csv`, sort by `|original_opinion − cross_lingual_opinion|` desc, eyeball the top ~20 disagreements. You're looking for:

- DeepL translation errors (mistranslated AI keywords, dropped sarcasm, lost URL/quote context)
- Cases where the *original* score was wrong and the cross-lingual run is correcting it (these aren't bias, they're noise)
- Systematic patterns — e.g. Chinese sarcasm flattening to neutral when translated to English

The eyeballed pattern matters more than the kappa. The reviewer wants to know *what kind* of bias the language switch introduces.

## Outputs in `CROSS_LINGUAL_DIR`

```
weibo_translation_sample.parquet     # transferred from China server
tweet_translation_sample.parquet     # produced by sample_tweets
weibo_translated.parquet             # zh→en
tweet_translated.parquet             # en→zh
deepl_cache.json                     # DeepL cache
translation_sample.csv               # 10+10 spot-check pairs (sample_translation)
weibo_en_requests.jsonl              # GPT batch input
weibo_en_results.jsonl               # GPT batch raw output
weibo_en_batch_state.json            # {file_id, batch_id, status}
weibo_reanalyzed_gpt.parquet         # parsed GPT scores
tweet_reanalyzed_kimi.parquet        # Kimi scores
diff_report.json                     # metrics + confusion matrix
diff_per_row.csv                     # per-post original vs cross-lingual
```

## Troubleshooting

- **`retrieve_weibo_gpt` says "not ready"** — batches typically complete in minutes to ~1 hour for 200 items; just re-run later.
- **Kimi rate limits / network errors** — `analyze_tweet_kimi` is resumable; rerun and it picks up where it left off.
- **DeepL quota** — the free tier is 500k chars/month; 400 short posts is well under.
- **Empty `cross_lingual_opinion` rows** — `_coerce_opinion` filters them out of the metrics; the report shows `n_valid` vs `n_total` so you can see how many dropped.
- **Re-running `sample_tweets`** — overwrites the existing sample. If you've already run downstream steps against the prior sample, delete or move the downstream artifacts before resampling.
