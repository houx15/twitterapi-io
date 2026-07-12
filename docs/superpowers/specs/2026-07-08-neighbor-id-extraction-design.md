# Neighbor-ID Extraction Pipeline — Design

**Date:** 2026-07-08
**Status:** Implemented. See revision note below.

> **Revision (2026-07-08, post-implementation):** The ego-subtraction no longer uses a
> separate external ego-ID file on Princeton. Because every `{user_id}-following.csv`
> member belongs to an ego, the ego IDs to exclude are derived from the **member
> filenames** while streaming: `extract_neighbors.py` takes `--ego_out=PATH` and writes
> each ego's `{user_id}` (one per member, ~0.4M total, streamed to a side file, never
> accumulated). Phase A produces a `.ego` part alongside each `.ids` part; Phase B merges
> the `.ego` parts and `comm -23` subtracts them — no `EGO_FILE` needed. The scripts are
> also self-contained (`NET_DIR`/paths and `#SBATCH --array=0-5` baked in) so they run via
> a bare `sbatch` with no exports. Any further exclusion against the crawl-completed ego
> list happens later on the crawl server. Sections below that reference a separate ego
> file / `EGO_FILE` / `ego_sorted.txt` are superseded by this note.
>
> **Revision 2 (2026-07-12):** The cluster limits concurrent jobs, so the slurm-array
> Phase A / Phase B split (`extract_neighbors_phaseA.slurm` + `..._phaseB.slurm`) was
> replaced by a **single job**, `slurm/extract_neighbors.slurm`, that processes all
> tarballs serially and then merges/subtracts/shards — one `sbatch`, one slot. It is
> **resumable**: re-submitting skips tarballs whose `.ids`/`.ego` parts already exist
> (atomic `.partial`→`mv`), so a time-limit kill just needs a resubmit. Sections below
> describing a slurm array or two-phase submission are superseded by this note.

## Goal

We have finished crawling tweets for ~0.4M **ego users**. We now want to expand the
crawl to the egos' **neighbors** — every account an ego user *follows* (their
following-lists), deduplicated across all egos, with the ego users themselves removed
(already crawled).

This spec covers **Step 1 only**: producing the deduplicated, ego-subtracted neighbor
ID list from the Princeton network data. Step 2 (crawling those IDs) reuses the existing
`user_tweet_crawler.py` and is out of scope here except for one flagged dependency
(see "Downstream dependency").

## Input data (Princeton server)

Location: `/scratch/network/COVID3/data-network`

```
data-twitter-network-6topics-{YYYY}{MMM}.tar.gz
└── {user_id}-following.csv      # following list (text, one int64 per line)
    {user_id}-following.pickle4  # same, python pickle (we ignore this)
    {user_id}-following.done     # empty completion marker (we ignore this)
```

- Every `{user_id}-following.csv` belongs to an **ego user** (confirmed).
- Each CSV line is one numeric followee user ID (int64). No header.
- The dataset is **extremely large**: ~0.4M ego users × their following-lists ⇒
  potentially hundreds of millions to ~2 billion total (non-unique) followee entries.

## Separate inputs

- **Ego-ID file**: a separate file containing the ego user IDs, one **numeric ID** per
  line (confirmed numeric, same namespace as the following-lists). Used only to subtract
  egos from the final neighbor set. We do **not** re-derive ego IDs by accumulating
  tarball filenames (that would cost memory).

## Constraints & principles

- **Memory-friendly above all.** Never load the full ID universe into RAM. No in-memory
  Python `set` of all followees. Dedup and subtraction happen on disk via external sort.
- **Do not fully unzip archives.** Stream members out of each `.tar.gz` sequentially;
  only read `*-following.csv` members; block-copy their bytes without materializing the
  archive.
- **Highest end-to-end efficiency.** `LC_ALL=C` byte-collation sort with `--parallel`;
  block-copy extraction (no per-line Python in the hot path); streaming merge & set-diff.
- **Fault isolation at scale.** A multi-hour serial job that dies loses everything;
  the design parallelizes per-tarball and isolates failures so a single bad archive is
  rerun in isolation.
- Follow existing repo style: standalone script exposed as a **Fire** CLI (matching
  `user_tweet_crawler.py`, `plot.py`); SLURM for execution (matching `slurm/analyze.slurm`).

## Architecture

Two cooperating pieces, each with one clear job:

### 1. `extract_neighbors.py` — streaming extractor (Fire CLI)

Single responsibility: walk tarball members and emit followee IDs to **stdout**. It does
**not** deduplicate, subtract, or hold any set in memory.

Command: `stream --tarball=<path>` (single archive; the unit of the slurm array).

Behavior:
- Open the archive with `tarfile.open(path, "r:gz")` and iterate **sequentially**
  (`for member in tar:`), never calling `getmembers()`/`extractall()`.
- Select only members whose name ends in `-following.csv` (this cleanly excludes
  `.pickle4` and `.done`).
- For each selected member, `tar.extractfile(member)` and `shutil.copyfileobj(f, stdout.buffer)`
  — block-level copy, O(block) memory, near-C speed.
- Wrap per-member reads in `try/except` and log+skip on error, so a corrupt member or
  archive never kills the job. Log progress (members streamed, bytes).
- Optional convenience command `stream_dir --dir=<path>` that loops all `*.tar.gz` in a
  directory to stdout — used only for a non-array fallback run.

Rationale for `tarfile` (pure Python) over `tar --wildcards -O` subprocess: no dependence
on GNU-tar wildcard flag portability, per-member fault isolation is natural, and
`copyfileobj` is already block-speed. `\r` normalization is handled downstream (`tr -d '\r'`).

### 2. SLURM orchestration — dedup, subtract, shard (coreutils, memory-bounded)

**Phase A — parallel per-tarball extract + local dedup (slurm array, one task per tarball):**

```bash
python extract_neighbors.py stream --tarball="$THIS_TARBALL" \
  | tr -d '\r' \
  | LC_ALL=C sort -u -S 50% --parallel="$CPUS" -T "$SCRATCH_TMP" \
  > "$PARTS/$(basename "$THIS_TARBALL").ids"
```
- One array index per tarball; the tarball for an index is chosen from a globbed list.
- Each task emits a **sorted-unique** shard. Tasks run concurrently; a failed tarball is
  rerun as a single array index without redoing the rest.
- `sort -S` caps memory; `-T "$SCRATCH_TMP"` spills to large scratch.

**Phase B — merge + ego-subtract + shard (one small job, runs after Phase A):**

```bash
# k-way streaming merge of already-sorted parts -> global unique neighbor set
LC_ALL=C sort -m -u "$PARTS"/*.ids > "$OUT/neighbors_unique.txt"

# ego file in the SAME byte collation
LC_ALL=C sort -u "$EGO_FILE" | tr -d '\r' > "$OUT/ego_sorted.txt"

# neighbors MINUS ego users (both identically sorted -> streaming set-difference)
LC_ALL=C comm -23 "$OUT/neighbors_unique.txt" "$OUT/ego_sorted.txt" > "$OUT/neighbors_final.txt"

# shard for step-2 crawling: 1,000,000 IDs per shard
split -l 1000000 -d --additional-suffix=.txt \
  "$OUT/neighbors_final.txt" "$OUT/neighbors_part_"

# report
wc -l "$OUT/neighbors_unique.txt" "$OUT/ego_sorted.txt" "$OUT/neighbors_final.txt"
```
`sort -m` is a cheap streaming merge because inputs are already sorted; `comm -23` is a
streaming two-file merge. Phase B stays light even at billion scale.

## Data flow

```
*.tar.gz  --(extract_neighbors.py stream, block-copy csv members)-->  raw ID stream (stdout)
raw stream  --(tr -d '\r' | sort -u -S -T --parallel)-->  per-tarball sorted-unique .ids   [Phase A, parallel array]
all .ids   --(sort -m -u)-->  neighbors_unique.txt
ego file   --(sort -u | tr -d '\r')-->  ego_sorted.txt
neighbors_unique.txt, ego_sorted.txt  --(comm -23)-->  neighbors_final.txt
neighbors_final.txt  --(split -l 1000000)-->  neighbors_part_00.txt, neighbors_part_01.txt, ...
```

## Output

- `neighbors_unique.txt` — all unique neighbor IDs (before ego removal). Intermediate.
- `ego_sorted.txt` — sorted-unique ego IDs. Intermediate.
- **`neighbors_final.txt`** — master result: sorted-unique neighbor IDs with egos removed,
  plain text, one numeric ID per line.
- **`neighbors_part_*.txt`** — 1,000,000 IDs per shard; direct input to the step-2 crawler.

Plain text one-ID-per-line is the highest-efficiency format end to end: trivially produced
by `sort`/`split`, and consumed directly by the crawler's `USERNAME_FILE` reader (`f.read().splitlines()`)
with zero conversion.

## Error handling

- Per-member and per-archive `try/except` in `extract_neighbors.py`; log and skip, never abort.
- Non-numeric / blank lines: `tr -d '\r'` normalizes line endings; malformed lines survive
  as-is through sort (dedup unaffected) — optionally the extractor can skip lines that are
  not all-digits, but default is trust-and-pass for speed. (Decision left to the plan; the
  data is documented as one int64 per line.)
- Phase B runs only after Phase A array completes (slurm dependency), so partial `.ids`
  files from a still-running array are never merged.

## SLURM resourcing (starting point, tune on first run)

- Phase A array: `--cpus-per-task` ~8–16, `--mem` sized so `sort -S 50%` has room
  (e.g. 32–64G), `--time` a few hours per tarball; `--array=0-(N_tarballs-1)`.
- `-T "$SCRATCH_TMP"` points at `/scratch` (needs ~archive-data-size of temp space).
- Phase B: single small job, modest CPU/mem, depends on Phase A (`--dependency=afterok:<arrayjobid>`).
- Mirror `slurm/analyze.slurm` conventions (conda env activation, `--mail-user`).

## Downstream dependency (Step 2 — flagged, not built here)

The existing crawler builds queries as `from:{username}` (Twitter's `from:` operator
expects a **handle**), but our neighbor list is numeric **user IDs**. Before Step 2 the IDs
likely need resolving to usernames (e.g. twitterapi.io batch user-info-by-ID endpoint), or
the crawler/query adapted to accept IDs. This is out of scope for Step 1 but must be
resolved before crawling begins.

## Assumptions

- Runs on Princeton Linux with **GNU coreutils** (`sort --parallel`, `comm`, `split -d --additional-suffix`).
- `/scratch` has enough free space for sort temp files and the per-tarball `.ids` shards.
- The ego-ID file is numeric IDs, one per line (confirmed).
- Ego users are exactly the accounts owning `-following.csv` files (confirmed); we still
  subtract using the separate ego file rather than accumulating filenames.

## Out of scope

- Resolving IDs → usernames (Step 2 prerequisite).
- The actual neighbor tweet crawl (reuses `user_tweet_crawler.py`).
- Any change to the following-network collection itself.
