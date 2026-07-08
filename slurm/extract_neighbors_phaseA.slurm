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
