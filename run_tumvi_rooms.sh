#!/usr/bin/env bash

set -u
set -o pipefail

# ============================================================
# Configuration
# ============================================================

ORB_ROOT="${ORB_ROOT:-$HOME/slam/ORB_SLAM3}"
DATA_ROOT="${DATA_ROOT:-$HOME/slam/dataset/TUM_VI}"

RUNS="${RUNS:-3}"

# Override, for example:
#   ROOMS="1" RUNS=1 ./run_tumvi_rooms.sh
ROOMS="${ROOMS:-1 2 3 4 5 6}"

EXEC="$ORB_ROOT/Examples/Stereo-Inertial/stereo_inertial_tum_vi"
VOCAB="$ORB_ROOT/Vocabulary/ORBvoc.txt"
SETTINGS="$ORB_ROOT/Examples/Stereo-Inertial/TUM-VI.yaml"

TIMES_DIR="$ORB_ROOT/Examples/Stereo-Inertial/TUM_TimeStamps"
IMU_DIR="$ORB_ROOT/Examples/Stereo-Inertial/TUM_IMU"

RESULT_ROOT="$ORB_ROOT/results/tumvi_rooms"
LOG_DIR="$RESULT_ROOT/logs"
TRAJ_DIR="$RESULT_ROOT/trajectories"

mkdir -p "$LOG_DIR" "$TRAJ_DIR"

# ============================================================
# Helpers
# ============================================================

format_time()
{
    local total=$1
    local h=$((total / 3600))
    local m=$(((total % 3600) / 60))
    local s=$((total % 60))
    printf "%02dh:%02dm:%02ds" "$h" "$m" "$s"
}

fail()
{
    echo "ERROR: $*" >&2
    exit 1
}

# ============================================================
# Basic checks
# ============================================================

[[ -x "$EXEC" ]]     || fail "Executable not found: $EXEC"
[[ -f "$VOCAB" ]]    || fail "Vocabulary not found: $VOCAB"
[[ -f "$SETTINGS" ]] || fail "Settings not found: $SETTINGS"

read -r -a ROOM_LIST <<< "$ROOMS"

TOTAL_RUNS=$((${#ROOM_LIST[@]} * RUNS))

echo "============================================================"
echo "TUM-VI ORB-SLAM3 stereo-inertial benchmark"
echo "============================================================"
echo "ORB-SLAM3 : $ORB_ROOT"
echo "Dataset   : $DATA_ROOT"
echo "Settings  : $SETTINGS"
echo "Rooms     : ${ROOM_LIST[*]}"
echo "Runs/room : $RUNS"
echo "Total runs: $TOTAL_RUNS"
echo "Results   : $RESULT_ROOT"
echo "============================================================"

# ============================================================
# Check requested datasets before starting
# ============================================================

for room in "${ROOM_LIST[@]}"; do

    SEQ="dataset-room${room}_512_16"

    CAM0="$DATA_ROOT/$SEQ/mav0/cam0/data"
    CAM1="$DATA_ROOT/$SEQ/mav0/cam1/data"

    TIMES="$TIMES_DIR/dataset-room${room}_512.txt"
    IMU="$IMU_DIR/dataset-room${room}_512.txt"

    [[ -d "$CAM0" ]] || fail "Missing cam0 data: $CAM0"
    [[ -d "$CAM1" ]] || fail "Missing cam1 data: $CAM1"
    [[ -f "$TIMES" ]] || fail "Missing timestamp file: $TIMES"
    [[ -f "$IMU" ]]   || fail "Missing IMU file: $IMU"

done

# ============================================================
# Run benchmark
# ============================================================

COMPLETED=0
TOTAL_ELAPSED=0
BENCHMARK_START=$(date +%s)

for room in "${ROOM_LIST[@]}"; do

    SEQ="dataset-room${room}_512_16"

    CAM0="$DATA_ROOT/$SEQ/mav0/cam0/data"
    CAM1="$DATA_ROOT/$SEQ/mav0/cam1/data"

    TIMES="$TIMES_DIR/dataset-room${room}_512.txt"
    IMU="$IMU_DIR/dataset-room${room}_512.txt"

    for run in $(seq 1 "$RUNS"); do

        TRAJ_NAME="tumvi_room${room}_stereoi_run${run}"
        LOG_FILE="$LOG_DIR/${TRAJ_NAME}.log"

        CURRENT=$((COMPLETED + 1))

        echo
        echo "============================================================"
        echo "[$CURRENT/$TOTAL_RUNS] Room $room -- run $run/$RUNS"
        echo "Trajectory: $TRAJ_NAME"
        echo "Started:    $(date)"
        echo "============================================================"

        RUN_START=$(date +%s)

        (
        cd "$TRAJ_DIR" || exit 1

        "$EXEC" \
            "$VOCAB" \
            "$SETTINGS" \
            "$CAM0" \
            "$CAM1" \
            "$TIMES" \
            "$IMU" \
            "$TRAJ_NAME"
        ) 2>&1 | tee "$LOG_FILE"

        STATUS=${PIPESTATUS[0]}

        RUN_END=$(date +%s)
        RUN_TIME=$((RUN_END - RUN_START))
       
        COMPLETED=$((COMPLETED + 1))
        TOTAL_ELAPSED=$((TOTAL_ELAPSED + RUN_TIME))

        AVG_TIME=$((TOTAL_ELAPSED / COMPLETED))
        REMAINING=$((TOTAL_RUNS - COMPLETED))
        ETA=$((AVG_TIME * REMAINING))

        echo
        echo "Completed room $room run $run."
        echo "Run time:          $(format_time "$RUN_TIME")"
        echo "Completed:         $COMPLETED / $TOTAL_RUNS"
        echo "Average/run:       $(format_time "$AVG_TIME")"
        echo "Estimated remain:  $(format_time "$ETA")"
        echo "Trajectory output: $TRAJ_DIR"
        echo "Log:               $LOG_FILE"

    done
done

BENCHMARK_END=$(date +%s)
BENCHMARK_TIME=$((BENCHMARK_END - BENCHMARK_START))

echo
echo "============================================================"
echo "All TUM-VI room runs completed."
echo "Total time: $(format_time "$BENCHMARK_TIME")"
echo "Trajectories:"
echo "  $TRAJ_DIR"
echo "Logs:"
echo "  $LOG_DIR"
echo "============================================================"
