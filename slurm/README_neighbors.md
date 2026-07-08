# Neighbor-ID extraction — run guide

Extracts the deduplicated, ego-subtracted neighbor (followee) ID list from the
Princeton following-network tarballs. See design:
`docs/superpowers/specs/2026-07-08-neighbor-id-extraction-design.md`.

Both scripts are **self-contained**: paths and the array range are baked in, so you
just `sbatch` them — no `export`s, no command-line `--array`. Egos are excluded using
the IDs derived from the tarball member filenames (`{user_id}-following.csv` → ego
`{user_id}`), so **no external ego-ID file is needed** on Princeton.

Defaults baked into the scripts (edit inside the files only if your setup differs):
- `NET_DIR=/scratch/network/COVID3/data-network`
- `PARTS=$SCRATCH/neighbor_extract/parts`, `OUT=$SCRATCH/neighbor_extract/out`,
  `SCRATCH_TMP=$SCRATCH/neighbor_extract/tmp`
- `SHARD_SIZE=1000000`
- Phase A `#SBATCH --array=0-5` (6 tarballs). If the tarball count changes, edit that
  one number in `slurm/extract_neighbors_phaseA.slurm`.

## 0. One-time: sanity-check the data layout

```bash
NET_DIR=/scratch/network/COVID3/data-network
ls -1 "$NET_DIR"/*.tar.gz | wc -l          # should be 6 (= array size 0-5)
# Peek at member names in one archive (fast, no full unzip):
NET_DIR="$NET_DIR" python - <<'PY'
import tarfile, itertools, os
p = __import__("glob").glob(os.environ["NET_DIR"] + "/*.tar.gz")[0]
with tarfile.open(p, "r:gz") as t:
    for m in itertools.islice(t, 5):
        print(m.name, m.size)
PY
```
Confirm names look like `.../{user_id}-following.csv`. If the count is not 6, update
`--array=0-(count-1)` in `slurm/extract_neighbors_phaseA.slurm`.

## 1. Submit Phase A, wait, then submit Phase B

Run these from the repo root (so `python extract_neighbors.py` resolves):

```bash
sbatch slurm/extract_neighbors_phaseA.slurm      # array: one task per tarball
# wait until all array tasks finish (squeue -u $USER), then:
sbatch slurm/extract_neighbors_phaseB.slurm      # merge + ego-subtract + shard
```

Phase A writes, per tarball: `$PARTS/<tarball>.ids` (sorted-unique followee IDs) and
`$PARTS/<tarball>.ego` (sorted-unique ego IDs from the filenames). Phase B merges all
of them and subtracts.

Optional — chain them in one go so Phase B auto-starts on success:
```bash
A=$(sbatch --parsable slurm/extract_neighbors_phaseA.slurm)
sbatch --dependency=afterok:$A slurm/extract_neighbors_phaseB.slurm
```

## 2. Dry-run one tarball first (recommended before the full array)

```bash
NET_DIR=/scratch/network/COVID3/data-network
TB=$(ls -1 "$NET_DIR"/*.tar.gz | head -1)
python extract_neighbors.py stream --tarball="$TB" --ego_out=/tmp/egos.txt \
  | awk '{ sub(/\r$/,"") } /^[0-9]+$/' \
  | LC_ALL=C sort -u | head
sort -u /tmp/egos.txt | head     # ego IDs parsed from this tarball's filenames
```
(Streaming stats — streamed/skipped/bytes — are logged to stderr, printed once per
task at the end.)

## 3. Results

- `$OUT/neighbors_final.txt` — sorted-unique neighbor IDs, egos removed (master list).
- `$OUT/neighbors_part_0000.txt`, `...0001.txt`, ... — 1,000,000 IDs each; feed to the
  step-2 crawler (`USERNAME_FILE`).

## Troubleshooting

- **`comm` complains input not sorted** — some `.ids`/`.ego` part predates a code change;
  rerun the offending Phase A index. All sorts must be `LC_ALL=C`.
- **Phase A array index fails** — rerun just that index:
  `sbatch --array=<idx> slurm/extract_neighbors_phaseA.slurm`. Only complete files get
  their final `.ids`/`.ego` name (atomic `.partial` rename), so a killed task leaves no
  half-written part.
- **`sort -m` "too many open files"** — only a concern with hundreds of parts; with 6
  tarballs (12 part files) it never triggers.

## Notes for later

- **Further ego exclusion:** this removes egos identified by the tarball filenames. If
  your crawl-completed ego list (on the crawling server) differs, do a second
  `comm -23 neighbors_final.txt <that_ego_list>` there before crawling.
- **Step 2 — IDs → usernames:** the crawler queries `from:{username}` (a handle), but
  these are numeric IDs. Resolve IDs to usernames (twitterapi.io batch user-info-by-ID)
  or adapt the crawler before crawling.
