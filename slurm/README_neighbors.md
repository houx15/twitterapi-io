# Neighbor-ID extraction — run guide

Extracts the deduplicated, ego-subtracted neighbor (followee) ID list from the
Princeton following-network tarballs. See design:
`docs/superpowers/specs/2026-07-08-neighbor-id-extraction-design.md`.

Everything runs in **one slurm job** — `slurm/extract_neighbors.slurm` — because the
cluster limits concurrent jobs. It processes all tarballs serially, then merges,
subtracts egos, and shards. It is **self-contained** (paths baked in, no `--array`) and
**resumable**. Egos are excluded using the IDs derived from the tarball member filenames
(`{user_id}-following.csv` → ego `{user_id}`), so **no external ego-ID file is needed**.

Defaults baked into the script (edit inside the file only if your setup differs):
- `NET_DIR=/scratch/network/COVID3/data-network`
- `PARTS=$SCRATCH/neighbor_extract/parts`, `OUT=$SCRATCH/neighbor_extract/out`,
  `SCRATCH_TMP=$SCRATCH/neighbor_extract/tmp`
- `SHARD_SIZE=1000000`, `SORT_BUF=32G` (per-sort buffer; only one sort runs at a time)
- Resources: `--cpus-per-task=16 --mem=64G --time=24:00:00` — tune to your allocation.

## 0. One-time: sanity-check the data layout

```bash
NET_DIR=/scratch/network/COVID3/data-network
ls -1 "$NET_DIR"/*.tar.gz | wc -l
# Peek at member names in one archive (fast, no full unzip):
NET_DIR="$NET_DIR" python - <<'PY'
import tarfile, itertools, os
p = __import__("glob").glob(os.environ["NET_DIR"] + "/*.tar.gz")[0]
with tarfile.open(p, "r:gz") as t:
    for m in itertools.islice(t, 5):
        print(m.name, m.size)
PY
```
Confirm names look like `.../{user_id}-following.csv`.

## 1. Submit the single job

Run from the repo root (so `python extract_neighbors.py` resolves):

```bash
sbatch slurm/extract_neighbors.slurm
```

That's it — one job, one slot. Watch it with `squeue -u $USER` and
`tail -f logs/extract_<jobid>.out`.

**If it hits the time limit** before finishing: just `sbatch` it again. Tarballs whose
`.ids` and `.ego` parts already exist under `$PARTS` are skipped, so it resumes where it
stopped, then runs the merge/subtract/shard.

## 2. Dry-run one tarball first (recommended)

```bash
NET_DIR=/scratch/network/COVID3/data-network
TB=$(ls -1 "$NET_DIR"/*.tar.gz | head -1)
python extract_neighbors.py stream --tarball="$TB" --ego_out=/tmp/egos.txt \
  | awk '{ sub(/\r$/,"") } /^[0-9]+$/' \
  | LC_ALL=C sort -u | head
sort -u /tmp/egos.txt | head     # ego IDs parsed from this tarball's filenames
```
(Streaming stats — streamed/skipped/bytes — are logged to stderr, once per tarball.)

## 3. Results

- `$OUT/neighbors_final.txt` — sorted-unique neighbor IDs, egos removed (master list).
- `$OUT/neighbors_part_0000.txt`, `...0001.txt`, ... — 1,000,000 IDs each; feed to the
  step-2 crawler (`USERNAME_FILE`).

## Troubleshooting

- **Re-run from scratch:** delete `$PARTS/*.ids` and `$PARTS/*.ego` (and `$OUT`) before
  resubmitting; otherwise existing parts are reused.
- **`comm` complains input not sorted** — a stale part predates a code change; delete that
  tarball's `.ids`/`.ego` and resubmit. All sorts must be `LC_ALL=C`.
- **OOM during sort** — lower `SORT_BUF` (e.g. `SORT_BUF=16G sbatch ...`) or raise `--mem`.
- **Too slow** — the tarballs are processed serially. If wall-clock matters more than job
  slots, we can add bounded internal parallelism (several tarballs at once in the one job);
  ask and we'll wire it up.

## Notes for later

- **Further ego exclusion:** this removes egos identified by the tarball filenames. If your
  crawl-completed ego list (on the crawling server) differs, do a second
  `comm -23 neighbors_final.txt <that_ego_list>` there before crawling.
- **Step 2 — IDs → usernames:** the crawler queries `from:{username}` (a handle), but these
  are numeric IDs. Resolve IDs to usernames (twitterapi.io batch user-info-by-ID) or adapt
  the crawler before crawling.
