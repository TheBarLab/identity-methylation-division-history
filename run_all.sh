#!/usr/bin/env bash
# Reproduce every table and figure from the downloaded data. ~20 minutes.
# Step 0 is separate: it needs its own environment (see requirements.txt).
# Step 8 runs before step 7, which plots its output (Supplementary Figure S1C).
# Step 9 (sensitivity checks quoted in the text) reruns steps 4 and 6 in a temporary folder.
set -euo pipefail
cd "$(dirname "$0")"
for s in src/step{1,2,3,4,5,6,8,9,7}_*.py; do
  echo "=== $s"
  python3 "$s"
done
echo "done: results/ and figures/ rebuilt"
