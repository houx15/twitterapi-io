# Neighbor-ID Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a deduplicated, ego-subtracted list of neighbor (followee) user IDs from the Princeton Twitter following-network tarballs, sharded for the step-2 crawler.

**Architecture:** A thin streaming extractor (`extract_neighbors.py`, Fire CLI) block-copies `*-following.csv` member contents out of each `.tar.gz` to stdout — no dedup, no in-memory set. Two SLURM stages then do all the heavy lifting with memory-bounded coreutils: Phase A is a slurm array (one task per tarball) that dedups each tarball locally (`sort -u`); Phase B merges the sorted parts (`sort -m -u`), subtracts ego IDs (`comm -23`), and shards (`split`).

**Tech Stack:** Python 3 + `tarfile` + `fire` (extractor), pytest (tests), GNU coreutils `sort`/`comm`/`split`/`awk`, SLURM.

## Global Constraints

- Extractor writes **only** followee IDs to **stdout**; all logging/progress goes to **stderr** (stdout must stay a clean ID pipe). Verbatim: `stream=sys.stderr` on the logging handler.
- Never load the full ID universe into memory — no in-memory Python `set` of all followees. Dedup/subtract happen on disk.
- Do not fully unzip archives: sequential member access via `tar.next()`; select only members whose basename ends `-following.csv`.
- Every member's streamed content must end in exactly one newline (guard against last-ID/first-ID concatenation across files).
- All `sort`/`comm` steps use `LC_ALL=C` (identical byte collation) so `comm -23` set-difference is valid.
- Ego file is numeric IDs, one per line (confirmed). Shard size = **1,000,000** IDs per file (confirmed).
- Match repo conventions: standalone script at repo root exposed via `fire.Fire({...})` (like `user_tweet_crawler.py`, `plot.py`); SLURM scripts under `slurm/` mirroring `slurm/analyze.slurm` (conda `opinion` env, `--mail-user`).
- Spec: `docs/superpowers/specs/2026-07-08-neighbor-id-extraction-design.md`.

---

## File Structure

- Create: `extract_neighbors.py` — streaming extractor + Fire CLI (`stream`, `stream_dir`).
- Create: `tests/test_extract_neighbors.py` — pytest unit tests over synthetic tarballs.
- Create: `slurm/extract_neighbors_phaseA.slurm` — array job: per-tarball extract + local `sort -u`.
- Create: `slurm/extract_neighbors_phaseB.slurm` — merge + ego-subtract + shard.
- Create: `slurm/README_neighbors.md` — runbook (env vars, array sizing, job dependency, dry-run).

---

## Task 1: Streaming extractor `extract_neighbors.py` (TDD)

**Files:**
- Create: `extract_neighbors.py`
- Test: `tests/test_extract_neighbors.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `stream_tarball(tarball_path, out) -> dict` — streams every `*-following.csv` member's bytes from one `.tar.gz` into binary stream `out`; returns `{"tarball": str, "streamed": int, "skipped": int, "bytes": int}`.
  - `_is_following_csv(name: str) -> bool` — true iff `name.split("/")[-1].endswith("-following.csv")`.
  - `_copy_member(file_obj, out) -> int` — block-copy with guaranteed single trailing newline; returns bytes written.
  - CLI commands `stream(tarball: str)` and `stream_dir(dir: str)` writing to `sys.stdout.buffer`.

- [ ] **Step 1: Ensure pytest is available**

Run: `python -c "import pytest" 2>/dev/null || pip install pytest`
Expected: no error (pytest importable).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_extract_neighbors.py`:

```python
import io
import tarfile

import extract_neighbors as en


def _make_tarball(tmp_path, members):
    """members: list of (name, data_bytes_or_None). None => directory entry."""
    p = tmp_path / "test.tar.gz"
    with tarfile.open(p, "w:gz") as tar:
        for name, data in members:
            if data is None:
                info = tarfile.TarInfo(name)
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return p


def _run(p):
    out = io.BytesIO()
    stats = en.stream_tarball(p, out)
    lines = [ln for ln in out.getvalue().decode().split("\n") if ln != ""]
    return lines, stats


def test_only_following_csv_streamed(tmp_path):
    p = _make_tarball(tmp_path, [
        ("111-following.csv", b"1\n2\n3\n"),
        ("111-following.pickle4", b"\x80\x04garbage"),
        ("111-following.done", b""),
    ])
    lines, stats = _run(p)
    assert lines == ["1", "2", "3"]
    assert stats["streamed"] == 1


def test_path_prefix_stripped(tmp_path):
    p = _make_tarball(tmp_path, [("somedir/333-following.csv", b"7\n")])
    lines, _ = _run(p)
    assert lines == ["7"]


def test_no_cross_file_concatenation(tmp_path):
    # First file has NO trailing newline; must not merge with next file's first ID.
    p = _make_tarball(tmp_path, [
        ("aaa-following.csv", b"8\n9"),
        ("bbb-following.csv", b"10\n"),
    ])
    lines, _ = _run(p)
    assert "9" in lines
    assert "10" in lines
    assert "910" not in lines


def test_ignores_pickle_done_and_dirs(tmp_path):
    p = _make_tarball(tmp_path, [
        ("adir", None),
        ("adir/999-following.csv", b"42\n"),
        ("adir/999-following.done", b""),
    ])
    lines, stats = _run(p)
    assert lines == ["42"]
    assert stats["streamed"] == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_extract_neighbors.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'extract_neighbors'` (module not created yet).

- [ ] **Step 4: Write `extract_neighbors.py`**

Create `extract_neighbors.py`:

```python
"""Stream followee IDs out of Twitter following-network tar.gz archives.

Emits one numeric followee ID per line to stdout (block-speed, memory-friendly).
Does NOT deduplicate — dedup / ego-subtract / shard happen downstream via coreutils
(sort -u | sort -m -u | comm -23 | split). See
docs/superpowers/specs/2026-07-08-neighbor-id-extraction-design.md
"""

import gc
import glob
import logging
import sys
import tarfile
from pathlib import Path

import fire

logger = logging.getLogger("extract_neighbors")

CHUNK = 1 << 16   # 64 KiB block copy
GC_EVERY = 100_000


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # keep stdout a clean ID pipe
    )


def _is_following_csv(name: str) -> bool:
    return name.split("/")[-1].endswith("-following.csv")


def _copy_member(file_obj, out) -> int:
    """Block-copy member bytes to `out`, guaranteeing a single trailing newline.

    Prevents the last ID of one CSV concatenating with the first ID of the next.
    Returns bytes copied (excluding any added newline).
    """
    written = 0
    last = b""
    while True:
        chunk = file_obj.read(CHUNK)
        if not chunk:
            break
        out.write(chunk)
        written += len(chunk)
        last = chunk[-1:]
    if written and last != b"\n":
        out.write(b"\n")
    return written


def stream_tarball(tarball_path, out) -> dict:
    """Stream all *-following.csv member contents from one tar.gz into `out` (binary).

    Sequential access via tar.next(); per-member try/except so one bad member never
    kills the job; periodic gc. Returns a stats dict.
    """
    tarball_path = str(tarball_path)
    streamed = skipped = total_bytes = seen = 0
    with tarfile.open(tarball_path, "r:gz") as tar:
        member = tar.next()
        while member is not None:
            seen += 1
            if seen % GC_EVERY == 0:
                gc.collect()
            try:
                if not (member.isfile() and _is_following_csv(member.name)):
                    member = tar.next()
                    continue
                file_obj = tar.extractfile(member)
                if file_obj is None:
                    skipped += 1
                    member = tar.next()
                    continue
                with file_obj:
                    total_bytes += _copy_member(file_obj, out)
                streamed += 1
                if streamed % GC_EVERY == 0:
                    logger.info("%s: %d members, %d bytes", tarball_path, streamed, total_bytes)
            except Exception as e:  # one bad member must not abort a multi-hour job
                skipped += 1
                logger.warning("skip member in %s: %s", tarball_path, e)
            member = tar.next()
    stats = {"tarball": tarball_path, "streamed": streamed, "skipped": skipped, "bytes": total_bytes}
    logger.info("done %s: %s", tarball_path, stats)
    return stats


def stream(tarball: str):
    """CLI: stream one tarball's followee IDs to stdout."""
    _setup_logging()
    stream_tarball(tarball, sys.stdout.buffer)
    sys.stdout.buffer.flush()


def stream_dir(dir: str):
    """CLI fallback (non-array): stream every *.tar.gz in a directory (sorted) to stdout."""
    _setup_logging()
    tarballs = sorted(glob.glob(str(Path(dir) / "*.tar.gz")))
    logger.info("found %d tarballs in %s", len(tarballs), dir)
    for tb in tarballs:
        stream_tarball(tb, sys.stdout.buffer)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    fire.Fire({"stream": stream, "stream_dir": stream_dir})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_extract_neighbors.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Smoke-test the CLI end to end**

Run:
```bash
python - <<'PY'
import io, tarfile
with tarfile.open("/tmp/smoke.tar.gz", "w:gz") as t:
    for name, data in [("100-following.csv", b"5\n6\n"), ("100-following.done", b"")]:
        i = tarfile.TarInfo(name); i.size = len(data); t.addfile(i, io.BytesIO(data))
PY
python extract_neighbors.py stream --tarball=/tmp/smoke.tar.gz 2>/dev/null
```
Expected stdout:
```
5
6
```

- [ ] **Step 7: Commit**

```bash
git add extract_neighbors.py tests/test_extract_neighbors.py
git commit -m "feat: streaming followee-ID extractor for network tarballs"
```

---

## Task 2: Phase A SLURM array (per-tarball extract + local dedup)

**Files:**
- Create: `slurm/extract_neighbors_phaseA.slurm`

**Interfaces:**
- Consumes: `extract_neighbors.py stream --tarball=<path>` (Task 1).
- Produces: one sorted-unique `$PARTS/<tarball-basename>.ids` per array index, consumed by Task 3.

- [ ] **Step 1: Write the Phase A script**

Create `slurm/extract_neighbors_phaseA.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=nbr-extractA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/phaseA_%A_%a.out
# NOTE: pass --array=0-$((N-1))%20 on the sbatch command line (N = number of tarballs).

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

NET_DIR="${NET_DIR:-/scratch/network/COVID3/data-network}"
PARTS="${PARTS:-$SCRATCH/neighbor_extract/parts}"
SCRATCH_TMP="${SCRATCH_TMP:-$SCRATCH/neighbor_extract/tmp}"
CPUS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$PARTS" "$SCRATCH_TMP" logs

# Deterministic tarball list; pick this task's tarball by array index.
mapfile -t TARBALLS < <(ls -1 "$NET_DIR"/*.tar.gz | sort)
N=${#TARBALLS[@]}
echo "Total tarballs: $N ; array index: ${SLURM_ARRAY_TASK_ID}"
if [ "${SLURM_ARRAY_TASK_ID}" -ge "$N" ]; then
  echo "Index beyond tarball count; nothing to do."
  exit 0
fi

TB="${TARBALLS[$SLURM_ARRAY_TASK_ID]}"
OUTFILE="$PARTS/$(basename "$TB").ids"
echo "Processing $TB -> $OUTFILE"

# Extract -> strip CR + keep only all-digit lines (awk exits 0 even with no matches)
# -> memory-bounded dedup. Write to .partial, then atomically rename on success.
python extract_neighbors.py stream --tarball="$TB" \
  | awk '{ sub(/\r$/, "") } /^[0-9]+$/' \
  | LC_ALL=C sort -u -S 50% --parallel="$CPUS" -T "$SCRATCH_TMP" \
  > "$OUTFILE.partial"
mv -f "$OUTFILE.partial" "$OUTFILE"

echo "Done: $(wc -l < "$OUTFILE") unique IDs in $OUTFILE"
```

- [ ] **Step 2: Shellcheck / syntax check (if available)**

Run: `bash -n slurm/extract_neighbors_phaseA.slurm && (command -v shellcheck >/dev/null && shellcheck slurm/extract_neighbors_phaseA.slurm || echo "shellcheck not installed, syntax OK")`
Expected: no syntax errors. (Ignore shellcheck SC2154 for SLURM-provided vars if it runs.)

- [ ] **Step 3: Commit**

```bash
git add slurm/extract_neighbors_phaseA.slurm
git commit -m "feat: Phase A slurm array — per-tarball extract + local dedup"
```

---

## Task 3: Phase B SLURM job (merge + ego-subtract + shard)

**Files:**
- Create: `slurm/extract_neighbors_phaseB.slurm`

**Interfaces:**
- Consumes: `$PARTS/*.ids` (Task 2), `$EGO_FILE` (numeric ego IDs, one per line).
- Produces: `$OUT/neighbors_final.txt` (master) and `$OUT/neighbors_part_*.txt` (1M-ID shards).

- [ ] **Step 1: Write the Phase B script**

Create `slurm/extract_neighbors_phaseB.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=nbr-extractB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/phaseB_%j.out

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

PARTS="${PARTS:-$SCRATCH/neighbor_extract/parts}"
OUT="${OUT:-$SCRATCH/neighbor_extract/out}"
SCRATCH_TMP="${SCRATCH_TMP:-$SCRATCH/neighbor_extract/tmp}"
EGO_FILE="${EGO_FILE:?set EGO_FILE to the ego-ID file (one numeric ID per line)}"
CPUS="${SLURM_CPUS_PER_TASK:-4}"
SHARD_SIZE="${SHARD_SIZE:-1000000}"
mkdir -p "$OUT" "$SCRATCH_TMP" logs

NPARTS=$(ls -1 "$PARTS"/*.ids | wc -l)
echo "Merging $NPARTS sorted part files..."
# Streaming k-way merge of already-sorted parts -> global unique neighbor set.
LC_ALL=C sort -m -u "$PARTS"/*.ids > "$OUT/neighbors_unique.txt"

echo "Normalizing + sorting ego file..."
awk '{ sub(/\r$/, "") } /^[0-9]+$/' "$EGO_FILE" \
  | LC_ALL=C sort -u -S 50% --parallel="$CPUS" -T "$SCRATCH_TMP" \
  > "$OUT/ego_sorted.txt"

echo "Subtracting egos (neighbors - egos)..."
LC_ALL=C comm -23 "$OUT/neighbors_unique.txt" "$OUT/ego_sorted.txt" \
  > "$OUT/neighbors_final.txt"

echo "Sharding into ${SHARD_SIZE}-line files..."
rm -f "$OUT"/neighbors_part_*.txt
split -l "$SHARD_SIZE" -d -a 4 --additional-suffix=.txt \
  "$OUT/neighbors_final.txt" "$OUT/neighbors_part_"

echo "=== Counts ==="
wc -l "$OUT/neighbors_unique.txt" "$OUT/ego_sorted.txt" "$OUT/neighbors_final.txt"
echo "Shards: $(ls -1 "$OUT"/neighbors_part_*.txt | wc -l)"
```

- [ ] **Step 2: Syntax check**

Run: `bash -n slurm/extract_neighbors_phaseB.slurm && echo "syntax OK"`
Expected: `syntax OK`.

- [ ] **Step 3: Portable logic check on a laptop (optional, macOS/BSD-safe)**

This validates the merge → subtract → shard logic with plain coreutils (no GNU-only flags):
```bash
cd /tmp && rm -rf pb && mkdir -p pb/parts pb/out
printf '1\n3\n5\n' > pb/parts/a.ids
printf '3\n7\n9\n' > pb/parts/b.ids
printf '5\n9\n' > pb/ego.txt
LC_ALL=C sort -m -u pb/parts/*.ids > pb/out/neighbors_unique.txt   # 1 3 5 7 9
LC_ALL=C sort -u pb/ego.txt > pb/out/ego_sorted.txt               # 5 9
LC_ALL=C comm -23 pb/out/neighbors_unique.txt pb/out/ego_sorted.txt > pb/out/neighbors_final.txt
cat pb/out/neighbors_final.txt
```
Expected output (neighbors minus egos):
```
1
3
7
```

- [ ] **Step 4: Commit**

```bash
git add slurm/extract_neighbors_phaseB.slurm
git commit -m "feat: Phase B slurm — merge, ego-subtract, shard neighbor IDs"
```

---

## Task 4: Runbook + submission wiring

**Files:**
- Create: `slurm/README_neighbors.md`

**Interfaces:**
- Consumes: Tasks 2 & 3 scripts.
- Produces: operator instructions (no code artifact consumed downstream).

- [ ] **Step 1: Write the runbook**

Create `slurm/README_neighbors.md`:

````markdown
# Neighbor-ID extraction — run guide

Extracts the deduplicated, ego-subtracted neighbor (followee) ID list from the
Princeton following-network tarballs. See design:
`docs/superpowers/specs/2026-07-08-neighbor-id-extraction-design.md`.

## 0. One-time: sanity-check the data layout

```bash
export NET_DIR=/scratch/network/COVID3/data-network
ls -1 "$NET_DIR"/*.tar.gz | wc -l          # number of tarballs (= array size)
# Peek at member names in one archive (fast, no full unzip):
python - <<'PY'
import tarfile, itertools
p = __import__("glob").glob("/scratch/network/COVID3/data-network/*.tar.gz")[0]
with tarfile.open(p, "r:gz") as t:
    for m in itertools.islice(t, 5):
        print(m.name, m.size)
PY
```
Confirm names look like `.../{user_id}-following.csv`.

## 1. Set environment (exported so both phases + sbatch inherit them)

```bash
export NET_DIR=/scratch/network/COVID3/data-network
export EGO_FILE=/PATH/TO/ego_ids.txt        # one numeric ego ID per line — EDIT THIS
export PARTS=$SCRATCH/neighbor_extract/parts
export OUT=$SCRATCH/neighbor_extract/out
export SCRATCH_TMP=$SCRATCH/neighbor_extract/tmp
export SHARD_SIZE=1000000
mkdir -p logs
```

## 2. Submit Phase A (array) then Phase B (dependent on A)

```bash
N=$(ls -1 "$NET_DIR"/*.tar.gz | wc -l)
A=$(sbatch --parsable --array=0-$((N-1))%20 slurm/extract_neighbors_phaseA.slurm)
echo "Phase A job: $A"
sbatch --dependency=afterok:$A slurm/extract_neighbors_phaseB.slurm
```

`%20` caps concurrent array tasks at 20 — tune to your allocation.

## 3. Dry-run one tarball first (recommended before the full array)

```bash
TB=$(ls -1 "$NET_DIR"/*.tar.gz | head -1)
python extract_neighbors.py stream --tarball="$TB" 2>/tmp/extract.log \
  | awk '{ sub(/\r$/,"") } /^[0-9]+$/' \
  | LC_ALL=C sort -u | head
tail /tmp/extract.log     # streamed/skipped/bytes stats
```

## 4. Results

- `$OUT/neighbors_final.txt` — sorted-unique neighbor IDs, egos removed (master list).
- `$OUT/neighbors_part_0000.txt`, `...0001.txt`, ... — 1,000,000 IDs each; feed to the
  step-2 crawler (`USERNAME_FILE`).

## Troubleshooting

- **`comm` complains input not sorted** — some `.ids` part predates a code change; rerun
  the offending Phase A index. All sorts must be `LC_ALL=C`.
- **Phase A array index fails** — rerun just that index:
  `sbatch --array=<idx> slurm/extract_neighbors_phaseA.slurm`. Only complete files get the
  final `.ids` name (atomic `.partial` rename), so a killed task leaves no half-written part.
- **`sort -m` "too many open files"** — if there are hundreds of `.ids` parts, raise the
  limit (`ulimit -n 4096`) before Phase B, or merge in two rounds.

## Step 2 (separate work — flagged): IDs → usernames

The crawler queries `from:{username}` (a handle), but these are numeric IDs. Resolve IDs to
usernames (twitterapi.io batch user-info-by-ID) or adapt the crawler before crawling.
````

- [ ] **Step 2: Commit**

```bash
git add slurm/README_neighbors.md
git commit -m "docs: runbook for neighbor-ID extraction pipeline"
```

---

## Self-Review Notes

- **Spec coverage:** streaming extractor (Task 1) ✔; Phase A parallel per-tarball dedup (Task 2) ✔; Phase B merge+subtract+shard (Task 3) ✔; runbook/resourcing/dry-run/step-2 flag (Task 4) ✔; memory-friendliness (no in-memory set; external sort) ✔; `LC_ALL=C` consistency ✔; 1M shard size ✔; ego numeric subtraction via `comm -23` ✔.
- **Malformed-line decision (was open in spec):** resolved to strict — `awk '/^[0-9]+$/'` drops blanks/non-numeric in a single C-speed pass in both phases.
- **Cross-file concatenation risk (new, from streaming):** guarded by `_copy_member` trailing-newline logic + `test_no_cross_file_concatenation`.
- **Type consistency:** `stream_tarball(tarball_path, out) -> dict`, `_is_following_csv`, `_copy_member` names match across Task 1 code, tests, and Tasks 2/3 invocations of `extract_neighbors.py stream --tarball=`.
