#!/usr/bin/env bash
# Fetch the UPMC and Charville CODEX data (Zenodo record 13179600, CC-BY-4.0; cite Wu et al.,
# "Graph deep learning for the characterization of tumour microenvironments from spatial protein
# profiles in tissue specimens", Nat. Biomed. Eng. 2022) into data/. Raw data is never committed.
# Usage: scripts/fetch_data.sh            # download from Zenodo
#        LOCAL_SRC=/path/to/zip scripts/fetch_data.sh   # link local copies instead
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA_DIR:-data}
mkdir -p "$DATA"
printf '*\n' > "$DATA/.gitignore"   # keeps raw data out of git even though the repo .gitignore does not list data/
BASE=https://zenodo.org/records/13179600/files
for f in upmc_labels.csv charville_labels.csv upmc_raw_data.zip charville_raw_data.zip; do
  if [ -s "$DATA/$f" ]; then echo "have $DATA/$f"; continue; fi
  if [ -n "${LOCAL_SRC:-}" ]; then
    ln -s "$(realpath "$LOCAL_SRC/$f")" "$DATA/$f"; echo "linked $DATA/$f"
  else
    curl -fL --retry 3 -C - -o "$DATA/$f.part" "$BASE/$f?download=1" && mv "$DATA/$f.part" "$DATA/$f"
    echo "fetched $DATA/$f"
  fi
done
