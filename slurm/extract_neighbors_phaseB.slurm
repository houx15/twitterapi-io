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
