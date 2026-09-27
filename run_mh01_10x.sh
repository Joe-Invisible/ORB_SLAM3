#!/usr/bin/env bash

# Do not use "set -e":
# mono_euroc may segfault during keyframe-trajectory saving even though
# the frame trajectory was successfully written.

ORB_DIR="$HOME/slam/ORB_SLAM3"
DATASET="$HOME/slam/dataset/EuRoC/MH_01_easy"
EVO_ENV="$HOME/slam/evo-env"

VOCAB="$ORB_DIR/Vocabulary/ORBvoc.txt"
CONFIG="$ORB_DIR/Examples/Monocular/EuRoC.yaml"
TIMESTAMPS="$ORB_DIR/Examples/Monocular/EuRoC_TimeStamps/MH01.txt"
GROUNDTRUTH="$ORB_DIR/evaluation/Ground_truth/EuRoC_left_cam/MH01_GT.txt"

RESULT_DIR="$ORB_DIR/results/MH01_10runs"
CSV="$RESULT_DIR/ate_results.csv"

mkdir -p "$RESULT_DIR"

# Activate evo virtual environment
source "$EVO_ENV/bin/activate"

# Check required commands/files
if ! command -v evo_ape >/dev/null 2>&1; then
    echo "ERROR: evo_ape not found."
    exit 1
fi

for f in "$VOCAB" "$CONFIG" "$TIMESTAMPS" "$GROUNDTRUTH"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing file: $f"
        exit 1
    fi
done

if [ ! -d "$DATASET" ]; then
    echo "ERROR: Dataset not found: $DATASET"
    exit 1
fi

echo "run,rmse_m" > "$CSV"

cd "$ORB_DIR" || exit 1

START_ALL=$(date +%s)

for i in $(seq -w 1 10); do

    echo "Starting run ${i}/10"

    NAME="MH01_mono_run${i}"

    RAW_TRAJ="$ORB_DIR/f_${NAME}.txt"
    KF_TRAJ="$ORB_DIR/kf_${NAME}.txt"

    SEC_TRAJ="$RESULT_DIR/f_${NAME}_sec.txt"
    LOG="$RESULT_DIR/${NAME}.log"
    EVO_LOG="$RESULT_DIR/${NAME}_evo.txt"

    echo
    echo "========================================"
    echo "Starting run ${i}/10"
    echo "========================================"

    ./Examples/Monocular/mono_euroc \
        "$VOCAB" \
        "$CONFIG" \
        "$DATASET" \
        "$TIMESTAMPS" \
        "$NAME" \
        > "$LOG" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ORB-SLAM3 exited with code $EXIT_CODE."
        echo "Checking whether frame trajectory was saved..."
    fi

    if [ ! -s "$RAW_TRAJ" ]; then
        echo "ERROR: Run $i did not produce a valid trajectory."
        echo "${i},FAILED" >> "$CSV"
        continue
    fi

    # ORB-SLAM3 EuRoC output uses nanosecond timestamps.
    # evo expects timestamps in seconds.
    awk '{
        printf "%.9f", $1 / 1000000000.0;
        for (j=2; j<=NF; j++) printf " %s", $j;
        printf "\n";
    }' "$RAW_TRAJ" > "$SEC_TRAJ"

    evo_ape euroc \
        "$GROUNDTRUTH" \
        "$SEC_TRAJ" \
        -r trans_part \
        -as \
        --t_max_diff 0.02 \
        > "$EVO_LOG" 2>&1

    RMSE=$(awk '$1 == "rmse" {print $2}' "$EVO_LOG")

    if [ -z "$RMSE" ]; then
        echo "ERROR: Could not extract RMSE for run $i."
        echo "${i},FAILED" >> "$CSV"
    else
        echo "Run $i ATE RMSE: $RMSE m"
        echo "${i},${RMSE}" >> "$CSV"
    fi

    # Preserve the original ORB-SLAM outputs too
    mv "$RAW_TRAJ" "$RESULT_DIR/" 2>/dev/null
    mv "$KF_TRAJ" "$RESULT_DIR/" 2>/dev/null

    END_RUN=$(date +%s)
    RUN_SECONDS=$((END_RUN - START_RUN))

    RUN_NUM=$((10#$i))
    ELAPSED=$((END_RUN - START_ALL))
    AVG_SECONDS=$((ELAPSED / RUN_NUM))
    REMAINING_RUNS=$((10 - RUN_NUM))
    ETA_SECONDS=$((AVG_SECONDS * REMAINING_RUNS))

    printf "Run %02d completed in %dm %02ds\n" \
        "$RUN_NUM" \
        $((RUN_SECONDS / 60)) \
        $((RUN_SECONDS % 60))

    printf "Progress: %d/10 | Elapsed: %dm | Estimated remaining: %dm %02ds\n" \
        "$RUN_NUM" \
        $((ELAPSED / 60)) \
        $((ETA_SECONDS / 60)) \
        $((ETA_SECONDS % 60))

done

echo
echo "========================================"
echo "Ten runs completed"
echo "========================================"

python3 - "$CSV" <<'PY'
import csv
import statistics
import sys

filename = sys.argv[1]

values = []

with open(filename, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            values.append(float(row["rmse_m"]))
        except ValueError:
            pass

if not values:
    print("No successful ATE measurements.")
    raise SystemExit(1)

print()
print("ATE RMSE results:")
for i, value in enumerate(values, 1):
    print(f"  Run {i:02d}: {value:.6f} m")

print()
print(f"Successful runs : {len(values)}")
print(f"Mean RMSE       : {statistics.mean(values):.6f} m")
print(f"Median RMSE     : {statistics.median(values):.6f} m")
print(f"Min RMSE        : {min(values):.6f} m")
print(f"Max RMSE        : {max(values):.6f} m")
PY

echo
echo "Results saved in:"
echo "$RESULT_DIR"
