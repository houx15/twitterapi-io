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
#SBATCH --array=0-5
# --array=0-5 -> one task per tarball (6 tarballs). If the tarball count changes,
# edit this single number (0-(N-1)); extra indices exit 0 via the bounds guard below.

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
if [ "$N" -eq 0 ]; then
  echo "ERROR: no *.tar.gz found under NET_DIR=$NET_DIR" >&2
  exit 1
fi
echo "Total tarballs: $N ; array index: ${SLURM_ARRAY_TASK_ID}"
if [ "${SLURM_ARRAY_TASK_ID}" -ge "$N" ]; then
  echo "Index beyond tarball count; nothing to do."
  exit 0
fi

TB="${TARBALLS[$SLURM_ARRAY_TASK_ID]}"
BASE="$(basename "$TB")"
OUTFILE="$PARTS/$BASE.ids"
EGOFILE="$PARTS/$BASE.ego"
EGORAW="$PARTS/$BASE.ego.raw"
echo "Processing $TB -> $OUTFILE (+ $EGOFILE)"

# Extract: followee IDs stream to stdout; ego IDs (from member filenames) go to
# --ego_out. Followees are CR-stripped, digit-filtered (awk exits 0 even with no
# matches), then memory-bounded deduped. Write to .partial, atomic rename on success.
python extract_neighbors.py stream --tarball="$TB" --ego_out="$EGORAW" \
  | awk '{ sub(/\r$/, "") } /^[0-9]+$/' \
  | LC_ALL=C sort -u -S 50% --parallel="$CPUS" -T "$SCRATCH_TMP" \
  > "$OUTFILE.partial"
mv -f "$OUTFILE.partial" "$OUTFILE"

# Dedup this tarball's ego IDs (the egos to be excluded in Phase B).
awk '{ sub(/\r$/, "") } /^[0-9]+$/' "$EGORAW" \
  | LC_ALL=C sort -u -S 50% --parallel="$CPUS" -T "$SCRATCH_TMP" \
  > "$EGOFILE.partial"
mv -f "$EGOFILE.partial" "$EGOFILE"
rm -f "$EGORAW"

echo "Done: $(wc -l < "$OUTFILE") followee IDs, $(wc -l < "$EGOFILE") ego IDs"
