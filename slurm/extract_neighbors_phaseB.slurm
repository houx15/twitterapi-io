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
CPUS="${SLURM_CPUS_PER_TASK:-4}"
SHARD_SIZE="${SHARD_SIZE:-1000000}"
mkdir -p "$OUT" "$SCRATCH_TMP" logs

NPARTS=$(ls -1 "$PARTS"/*.ids | wc -l)
echo "Merging $NPARTS sorted part files..."
# Streaming k-way merge of already-sorted parts -> global unique neighbor set.
LC_ALL=C sort -m -u "$PARTS"/*.ids > "$OUT/neighbors_unique.txt"

# Ego IDs to exclude are derived from the tarball member filenames in Phase A
# (each {user_id}-following.csv belongs to an ego). Merge the per-tarball .ego parts.
echo "Merging ego IDs (from filenames)..."
LC_ALL=C sort -m -u "$PARTS"/*.ego > "$OUT/ego_unique.txt"

echo "Subtracting egos (neighbors - egos)..."
LC_ALL=C comm -23 "$OUT/neighbors_unique.txt" "$OUT/ego_unique.txt" \
  > "$OUT/neighbors_final.txt"

echo "Sharding into ${SHARD_SIZE}-line files..."
rm -f "$OUT"/neighbors_part_*.txt
split -l "$SHARD_SIZE" -d -a 4 --additional-suffix=.txt \
  "$OUT/neighbors_final.txt" "$OUT/neighbors_part_"

echo "=== Counts ==="
wc -l "$OUT/neighbors_unique.txt" "$OUT/ego_unique.txt" "$OUT/neighbors_final.txt"
echo "Shards: $(ls -1 "$OUT"/neighbors_part_*.txt | wc -l)"
