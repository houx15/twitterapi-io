# Cross-lingual sentiment validation — results

## Background

Reviewers raised a concern about our China/US AI-attitude comparison: the original weibo sentiment used Kimi (Moonshot) with a Chinese prompt; the original tweet sentiment used GPT-5-mini batch with an English prompt. Differences in LLM and prompt could bias the country-level contrast.

To test for analyzer-driven bias we re-score each side with the *other* LLM, translating via DeepL. The sample and translation are held fixed, and the LLM scoring is repeated 5 times to isolate LLM noise from sampling noise.

## Method

1. Proportional-stratified sample of 200 weibos (with their existing Kimi-Chinese opinions) and 200 tweets (with their existing GPT-English opinions). Stratified by original opinion class `-2 / -1 / 0 / +1 / +2`.
2. DeepL translation: weibo `zh→en` once; tweet `en→zh` once.
3. **5 rounds** of GPT batch on the translated weibo + 5 rounds of Kimi live API on the translated tweets, all on the same translated text.
4. Per round: diff cross-lingual opinion against the original opinion. Aggregate: mean ± std of per-round metrics; per-post LLM stability; modal-vote diff (per-post mode of 5 runs vs original).

GPT reanalysis reuses the existing hosted prompt (`BATCH_PROMPT_ID` / `BATCH_PROMPT_VERSION` / `BATCH_MODEL`), so the only thing that differs from the original tweet run is the content. Kimi reanalysis reuses the verbatim Chinese system prompt from `youth-analysis/ai_sentiment_analyzer.py`. This holds the prompt content constant across languages.

## Results

### Per-round diff vs original (mean ± std across 5 rounds)

| Metric | Weibo (zh→en, GPT) | Tweet (en→zh, Kimi) |
|---|---|---|
| n_valid_pairs | 164.0 ± 3.7 | 151.2 ± 2.4 |
| cross cannot_tell proportion | 0.180 ± 0.019 | 0.244 ± 0.012 |
| original cannot_tell proportion | 0.000 | 0.000 |
| exact_match | 0.635 ± 0.027 | 0.700 ± 0.017 |
| within ±1 | 0.972 ± 0.004 | 0.981 ± 0.003 |
| mean \|diff\| | 0.403 ± 0.027 | 0.325 ± 0.017 |
| **mean signed (cross − orig)** | **−0.056 ± 0.021** | **+0.018 ± 0.026** |
| quadratic kappa | 0.793 ± 0.013 | 0.894 ± 0.005 |

### LLM stability across rounds (same translated text)

| Metric | Weibo | Tweet |
|---|---|---|
| posts with all 5 runs valid | 146 / 200 (73.0%) | 132 / 200 (66.0%) |
| posts unanimous across runs | 104 / 200 (52.0%) | 96 / 200 (48.0%) |
| posts within ±1 across runs | 144 / 200 (72.0%) | 132 / 200 (66.0%) |
| mean modal-count (out of 5) | 3.74 | 3.48 |

Denominator is the full 200-post sample (conservative — a post with any `cannot_tell` is "not unanimous"). Conditional on all 5 valid: weibo 104 / 146 = 71.2%, tweet 96 / 132 = 72.7% unanimous.

### Modal-vote diff vs original (per-post mode of 5 rounds)

| Metric | Weibo | Tweet |
|---|---|---|
| exact_match | 0.658 | 0.742 |
| **signed_diff (cross − orig)** | **−0.027** | **+0.030** |
| quadratic kappa | 0.804 | 0.920 |

## Interpretation

**Cross-lingual scores agree strongly with the original.** Modal-vote quadratic kappa is 0.80 (weibo) and 0.92 (tweet); 97–98% of posts agree within one ordinal level. The dominant signal is preserved when we swap LLM + language.

**Residual bias is small and directionally consistent across both samples.** Weibo: GPT-on-translated-weibo scored 0.056 *lower* than Kimi-on-original-weibo. Tweet: Kimi-on-translated-tweet scored 0.018 *higher* than GPT-on-original-tweet. Both directions imply **Kimi-Chinese rates ~0.02–0.06 more positively than GPT-English on the same content**, on a 5-point scale (`-2..+2`). The weibo effect is ~2.6× the round-to-round std (borderline-distinguishable from zero); the tweet effect is within noise.

**Content language and analyzer choice are confounded.** Because we swap both at once, the residual bias cannot be decomposed into "Kimi rates more positively" vs "Chinese-language framing elicits more positive scores". For the China/US comparison the practical implication is the same: the choice of LLM/prompt does not shift the cross-country contrast by more than ~0.06 on the opinion scale.

**Cannot_tell rate is non-trivial.** GPT punted on 18% of translated weibos; Kimi punted on 24% of translated tweets. The originals were already filtered to valid integer opinions, so this reflects a combination of translation loss and analyzer hesitation on borderline content. The agreement metrics are conditional on both analyzers committing to a label.

**LLM stability is moderate.** The modal label appears on average 3.74 / 3.48 of 5 runs; only ~50% of posts get the exact same label every time. The modal-vote diff is the most reliable single-number estimate (averages out per-run LLM noise).

## Headline numbers for the reviewer response

- Modal-vote quadratic kappa: **0.80** (weibo) / **0.92** (tweet)
- Within-±1 ordinal agreement: **97.2%** (weibo) / **98.1%** (tweet)
- Residual signed bias: **−0.06** (weibo) / **+0.02** (tweet) on a 5-point scale

The LLM/prompt choice contributes less than ~0.06 of bias on the opinion scale and preserves ordinal ranking with κ ≥ 0.80. The China/US contrast is unlikely to be driven by analyzer choice.

## Next steps

### 1. Adjust sampling strategy

Current sampling is proportional-stratified by original opinion class, which means most sampled posts come from the dominant class (typically class 0, mild stance). Rare extreme classes (`−2`, `+2`) are under-represented and contribute the noisiest part of the kappa estimate. Options:

- **Equal-stratification** (e.g., 40 per class with the current sample size, or 80 per class if we also enlarge): gives reliable per-class agreement numbers, useful if the reviewer wants to see whether cross-lingual agreement degrades at the extremes.
- **Length / clarity filter** before sampling — restrict to posts above a minimum length or with explicit AI keywords. Reduces the `cannot_tell` rate; tradeoff: less representative of the full distribution.
- **Drop sarcasm-heavy / quote-retweet rows** — these are the categories where DeepL most often loses the original sentiment. Practical filter: posts containing `//` (weibo retweet marker) or quote-tweet structure.

### 2. Adjust sample size

200 per side leaves `n_valid_pairs` around 150–165 after `cannot_tell` filtering, and the modal-vote diff uses `n_posts_all_runs_valid` of 132–146. Round-to-round std on signed_diff is 0.02–0.03 — comparable to the magnitude we are trying to resolve.

Doubling to 400 per side would roughly halve the standard error on signed_diff and tighten the kappa CI accordingly. Costs are linear: ~800 DeepL translations (still well under the free 500k chars/month quota), 5× GPT batch and 5× Kimi calls. A reasonable plan: resample to 400 per side with equal-stratification (80 per opinion class), re-run the 5-round LLM scoring, and report agreement broken down by original-opinion class.
