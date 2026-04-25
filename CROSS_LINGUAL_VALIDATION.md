# Cross-lingual sentiment validation — US side

Reviewer concern: original weibo sentiment used Kimi + Chinese prompt; tweet sentiment used GPT + English prompt. Different LLMs and prompts could bias the comparison.

This runbook covers the **US side**: sample 200 tweets, translate both samples via DeepL **once**, then re-score the same translated text with the *other* LLM **5 times** to measure LLM stability rather than sampling variance. The China side (sampling weibos) is in `youth-analysis/CROSS_LINGUAL_VALIDATION.md`; finish that first and `scp` the resulting parquet here.

## Experiment design

For each sample post we:

1. Take the existing opinion score (from the original analyzer)
2. Translate the text to the *other* language via DeepL — **once**
3. Re-score with the *other* analyzer **5 times**, using the same prompt the analyzer originally used
4. Per round: diff cross-lingual vs original; aggregate: mean ± std + per-post LLM stability

Holding the sample and translation fixed isolates LLM noise: if the analyzer gives 5 different labels on the same translated text, no single-shot diff is trustworthy. The aggregate report tells you both how stable the diff is across runs and how often the analyzer agreed with itself across runs on each post.

**Translated weibo → GPT reanalysis** reuses the existing `BATCH_PROMPT_ID` / `BATCH_PROMPT_VERSION` / `BATCH_MODEL` — i.e. the exact prompt that originally scored the tweets. Same `Tweet text:` framing, same `/v1/responses` endpoint, same `reasoning.effort=medium`. The only thing that differs from the original tweet run is the content (translated weibo).

**Translated tweet → Kimi reanalysis** reuses the verbatim Chinese system prompt from `youth-analysis/ai_sentiment_analyzer.py`, with the same `微博内容 / 发布时间 / 是否转发` user-message format. `是否转发` is hardcoded to `否` and `发布时间` is parsed from Twitter's `createdAt`. Minor caveat: those two metadata fields are constants/derived rather than per-row signals.

## Layout

Single sample + single translation at the top level; only the LLM-scoring + per-round diff live under `round_<N>/`. The aggregate report sits at the top level.

```
sentiment_results/cross_lingual/
├── deepl_cache.json                # shared DeepL cache
├── weibo_translation_sample.parquet  # transferred from China server
├── tweet_translation_sample.parquet
├── weibo_translated.parquet         # zh→en (one-shot)
├── tweet_translated.parquet         # en→zh (one-shot)
├── translation_sample.csv           # 10+10 spot-check pairs
├── round_1/
│   ├── weibo_en_{requests,results}.jsonl
│   ├── weibo_en_batch_state.json
│   ├── weibo_reanalyzed_gpt.parquet
│   ├── tweet_reanalyzed_kimi.parquet
│   ├── diff_report.json
│   └── diff_per_row.csv
├── round_2/  ...
├── round_5/  ...
├── aggregate_report.json            # mean ± std + per-post stability
└── aggregate_per_row.csv            # per-post wide table: round-1..5 + modal
```

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

Translation goes through the official [`deepl`](https://pypi.org/project/deepl/) Python SDK. The reanalysis of translated weibo reuses `BATCH_PROMPT_ID` / `BATCH_PROMPT_VERSION` / `BATCH_MODEL` already in `config.py`.

### 2. Refresh dependencies

```bash
pip install -r requirements.txt
```

### 3. Receive the weibo sample

```bash
mkdir -p sentiment_results/cross_lingual
scp <user>@<china-server>:<path>/ai_attitudes/weibo_translation_sample.parquet \
    sentiment_results/cross_lingual/
```

### 4. (One-time) migrate any existing flat-layout LLM artifacts into round_1/

If you already finished one round under the flat layout, move just the LLM-scoring + diff files into `round_1/`. **Sample and translation files stay at the top level.**

```bash
cd sentiment_results/cross_lingual
mkdir -p round_1
mv weibo_en_requests.jsonl weibo_en_results.jsonl weibo_en_batch_state.json \
   weibo_reanalyzed_gpt.parquet tweet_reanalyzed_kimi.parquet \
   diff_report.json diff_per_row.csv \
   round_1/ 2>/dev/null
cd -
```

## Pipeline

All commands run from the `twitterapi-io` directory.

### One-shot prep (skip if already done)

```bash
python cross_lingual_validation.py sample_tweets
python cross_lingual_validation.py translate_weibo
python cross_lingual_validation.py translate_tweets
python cross_lingual_validation.py sample_translation        # optional spot-check
```

### Scoring — 5 rounds of GPT + Kimi on the same translated text

```bash
python cross_lingual_validation.py run_scoring --rounds 5
```

For each round 1..5 this runs:
1. `submit_weibo_gpt --round N` — submit a fresh GPT batch on the translated weibo
2. `analyze_tweet_kimi --round N` — Kimi live API on translated tweets (resumable)

Both subcommands skip rounds whose primary outputs already exist, so re-running after a partial pass is safe. When this finishes, **5 GPT batches are queued** and **5 Kimi runs are done**. Wait for OpenAI to complete the batches (minutes to ~1 hour each), then continue.

To resume from a partial round, e.g. round 3 onwards: `--start_round 3`.

### Finalize — retrieve, parse, diff per round, aggregate

```bash
python cross_lingual_validation.py finalize --rounds 5
```

For each round: retrieve the GPT batch, parse, diff. Rounds whose batches haven't completed yet are skipped — re-run `finalize` later. After at least one round finishes, an aggregate is written.

To run a single round manually:

```bash
python cross_lingual_validation.py retrieve_weibo_gpt --round 3
python cross_lingual_validation.py parse_weibo_gpt    --round 3
python cross_lingual_validation.py diff               --round 3
python cross_lingual_validation.py aggregate          --rounds 5
```

### Per-step notes

- **`sample_tweets`** — reads `sentiment_results/batch_results.parquet` and joins to date-filtered tweet parquets. Proportional-stratified by class; seed=42.
- **`translate_*`** — caches `(target_lang, text) → translation` in `deepl_cache.json`, persisted every 25 calls. Safe to interrupt and re-run.
- **`sample_translation`** — prints `n` random pairs from each direction (default 10) and writes `translation_sample.csv`. Eyeball whether DeepL handled AI keywords / sarcasm / quoted text reasonably.
- **`submit_weibo_gpt --round N`** — uploads requests, creates batch, persists `{round, file_id, batch_id, status}` to `round_N/weibo_en_batch_state.json`. Each round is a fresh batch on the same translated text — variance comes from GPT sampling, not different inputs.
- **`retrieve_weibo_gpt --round N`** — idempotent. Returns `True` iff completed (used by `finalize`).
- **`parse_weibo_gpt --round N`** — extracts `opinion` and writes `round_N/weibo_reanalyzed_gpt.parquet`.
- **`analyze_tweet_kimi --round N`** — Kimi live, ~0.1s delay, 3 retries with backoff, periodic save every 20 rows. Re-run resumes within a round (skips IDs already in `round_N/tweet_reanalyzed_kimi.parquet`).
- **`diff --round N`** — joins original vs round-N cross-lingual opinions; writes `round_N/diff_report.json` and `diff_per_row.csv`.
- **`aggregate --rounds 5`** — pools the 5 per-round diffs into mean ± std for each numeric metric; computes per-post LLM stability (unanimous / within-1 / modal-count); runs a modal-vote diff (per-post mode of 5 rounds vs original); writes `aggregate_report.json` and `aggregate_per_row.csv`.

## Reading the report

`round_N/diff_report.json` (per-round, structure unchanged from the single-round case):

- `n_total`, `original_breakdown`, `cross_breakdown` (incl. `cannot_tell_proportion`), `n_valid_pairs`
- `exact_match`, `within_one`, `mean_abs_diff`, `mean_signed_diff_cross_minus_orig`, `quadratic_kappa`, `confusion_matrix`

`aggregate_report.json` adds, for each side:

- `summary` — `mean_<metric>` / `std_<metric>` (ddof=1) across the rounds. Tight std → diff is stable across LLM runs.
- `per_post_stability` — denominator is `n_posts` (all 200) for every proportion below. A post counts as "unanimous" only if all 5 runs returned the **same valid integer**; a post that returned `cannot_tell` on any run is therefore *not* unanimous. This is the conservative reading: cannot_tell counts as instability, not as a separate refusal. To recover the conditional rate ("of posts where the analyzer always returned a valid label, how often did it agree with itself"), use `n_posts_unanimous / n_posts_all_runs_valid`.
  - `n_posts_all_runs_valid` / `proportion_all_runs_valid` — fraction of posts where every round returned a valid integer (no cannot_tell or parse failures)
  - `n_posts_unanimous` / `proportion_unanimous` — fraction with the same valid integer label on all rounds
  - `n_posts_within_one_across_rounds` / `proportion_within_one_across_rounds` — fraction within ±1 across all rounds (and all rounds valid)
  - `mean_modal_count` — average count of the most common label among the 5 runs (5.0 = always agreed; 1.0 = never agreed)
  - `modal_vote_diff` — apply the per-post mode as a "consensus" cross-lingual label and diff against the original. This is your most reliable single-number defense.

`aggregate_per_row.csv` — wide table: `post_id, sample, original_opinion, opinion_round_1, …, opinion_round_5, modal_opinion, modal_count, all_agree, within_one_across_rounds, n_valid_runs`. Sort by `modal_count` ascending to find the most LLM-unstable posts; sort by `|original_opinion - modal_opinion|` to find posts where the cross-lingual analyzer most disagrees with the original.

Three numbers worth quoting in the response to the reviewer:

1. **`proportion_unanimous`** — how reliable is the cross-lingual scoring on the same text? (high = the LLM choice doesn't matter much; reviewer concern is mooted)
2. **`modal_vote_diff.exact_match` / `quadratic_kappa`** — once we average out LLM noise, how well does the cross-lingual analyzer agree with the original?
3. **`modal_vote_diff.mean_signed_diff_cross_minus_orig`** — direction of any residual bias (positive = cross-lingual analyzer is more positive than original)

## Spot-check before trusting the metrics

Open `aggregate_per_row.csv`, find rows with low `modal_count` (LLM disagreement) and rows with large `|original_opinion - modal_opinion|` (cross-lingual disagreement with the original), eyeball ~20. Look for:

- DeepL translation errors (mistranslated AI keywords, dropped sarcasm, lost URL/quote context)
- Cases where the *original* score was wrong and the cross-lingual run is correcting it (noise, not bias)
- Systematic patterns — e.g. Chinese sarcasm flattening to neutral when translated to English

## Troubleshooting

- **`finalize` says "not ready"** — batches typically complete in minutes to ~1 hour each. Re-run later; rounds that aren't ready are skipped, completed ones still get diff'd.
- **Kimi rate limits / network errors** — `analyze_tweet_kimi` is resumable per round. Re-run `run_scoring` and it picks up from the last save.
- **DeepL quota** — single-shot translation of 200+200 short posts is well under the free 500k chars/month tier.
- **Mixed `int / "cannot tell"` errors** — fixed; `_opinion_to_str` writes a uniform string column. `_coerce_opinion` recovers ints for metrics.
- **Re-sampling** — delete the relevant top-level parquet (and the round_<N>/ subdirs) and re-run; cached translations come back fast from `deepl_cache.json`.
