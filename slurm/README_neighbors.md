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
