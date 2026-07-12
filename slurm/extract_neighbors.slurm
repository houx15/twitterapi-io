#!/bin/bash
#SBATCH --job-name=nbr-extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/extract_%j.out

# SINGLE-JOB extraction: does the whole pipeline (per-tarball extract+dedup, then
# merge + ego-subtract + shard) inside ONE slurm job — no array — because the cluster
# limits us to very few concurrent jobs. Tarballs are processed serially; each
# tarball's `python | awk | sort` pipeline already spreads across cores.
#
# RESUMABLE: re-submitting skips any tarball whose .ids AND .ego parts already exist
# (parts are written atomically via .partial -> mv), so a job killed by the time limit
# picks up where it left off.
#
# Just run:  sbatch slurm/extract_neighbors.slurm   (no exports, no --array)

module load anaconda3/2023.3
conda activate llm

NET_DIR="${NET_DIR:-/scratch/network/COVID3/data-network}"
# All outputs live under a per-user project folder (created by mkdir -p below).
PROJECT_DIR="${PROJECT_DIR:-/scratch/network/yh6580/twitterapi-io}"
PARTS="${PARTS:-$PROJECT_DIR/neighbor_extract/parts}"
OUT="${OUT:-$PROJECT_DIR/neighbor_extract/out}"
SCRATCH_TMP="${SCRATCH_TMP:-$PROJECT_DIR/neighbor_extract/tmp}"
CPUS="${SLURM_CPUS_PER_TASK:-4}"
SHARD_SIZE="${SHARD_SIZE:-1000000}"
# Absolute sort buffer (only ONE sort runs at a time, so this is safe well under --mem).
# Absolute value avoids `sort -S <pct>` reading node RAM instead of the cgroup limit.
SORT_BUF="${SORT_BUF:-32G}"
mkdir -p "$PARTS" "$OUT" "$SCRATCH_TMP" logs

mapfile -t TARBALLS < <(ls -1 "$NET_DIR"/*.tar.gz | sort)
N=${#TARBALLS[@]}
if [ "$N" -eq 0 ]; then
  echo "ERROR: no *.tar.gz found under NET_DIR=$NET_DIR" >&2
  exit 1
fi
echo "Found $N tarballs under $NET_DIR"

# ---- Per-tarball: followee IDs -> sorted-unique .ids ; ego IDs (from filenames) -> .ego ----
for TB in "${TARBALLS[@]}"; do
  BASE="$(basename "$TB")"
  IDS="$PARTS/$BASE.ids"
  EGO="$PARTS/$BASE.ego"
  if [[ -f "$IDS" && -f "$EGO" ]]; then
    echo "[skip]    $BASE (parts already present)"
    continue
  fi
  echo "[extract] $BASE"
  EGORAW="$PARTS/$BASE.ego.raw"

  # Followees stream to stdout; ego IDs go to --ego_out. Followees are CR-stripped,
  # digit-filtered (awk exits 0 even with no matches), then memory-bounded deduped.
  python extract_neighbors.py stream --tarball="$TB" --ego_out="$EGORAW" \
    | awk '{ sub(/\r$/, "") } /^[0-9]+$/' \
    | LC_ALL=C sort -u -S "$SORT_BUF" --parallel="$CPUS" -T "$SCRATCH_TMP" \
    > "$IDS.partial"
  mv -f "$IDS.partial" "$IDS"

  awk '{ sub(/\r$/, "") } /^[0-9]+$/' "$EGORAW" \
    | LC_ALL=C sort -u -S "$SORT_BUF" --parallel="$CPUS" -T "$SCRATCH_TMP" \
    > "$EGO.partial"
  mv -f "$EGO.partial" "$EGO"
  rm -f "$EGORAW"

  echo "[done]    $BASE: $(wc -l < "$IDS") followees, $(wc -l < "$EGO") egos"
done

# ---- Merge + ego-subtract + shard ----
echo "Merging followee parts..."
LC_ALL=C sort -m -u "$PARTS"/*.ids > "$OUT/neighbors_unique.txt"

echo "Merging ego parts (from filenames)..."
LC_ALL=C sort -m -u "$PARTS"/*.ego > "$OUT/ego_unique.txt"

echo "Subtracting egos (neighbors - egos)..."
LC_ALL=C comm -23 "$OUT/neighbors_unique.txt" "$OUT/ego_unique.txt" > "$OUT/neighbors_final.txt"

echo "Sharding into ${SHARD_SIZE}-line files..."
rm -f "$OUT"/neighbors_part_*.txt
split -l "$SHARD_SIZE" -d -a 4 --additional-suffix=.txt \
  "$OUT/neighbors_final.txt" "$OUT/neighbors_part_"

echo "=== Counts ==="
wc -l "$OUT/neighbors_unique.txt" "$OUT/ego_unique.txt" "$OUT/neighbors_final.txt"
echo "Shards: $(ls -1 "$OUT"/neighbors_part_*.txt | wc -l)"
echo "Output dir: $OUT"
