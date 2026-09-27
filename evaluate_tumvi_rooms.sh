#!/usr/bin/env bash
set -e

ORB_ROOT="$HOME/slam/ORB_SLAM3"
DATA_ROOT="$HOME/slam/dataset/TUM_VI"
TRAJ_DIR="$ORB_ROOT/results/tumvi_rooms/trajectories"
OUT="$ORB_ROOT/results/tumvi_rooms/ate_results.csv"

echo "room,run,rmse" > "$OUT"

for room in 1 2 3 4 5 6; do
    GT="$DATA_ROOT/dataset-room${room}_512_16/mav0/mocap0/data.csv"

    for run in 1 2 3; do
        TRAJ="$TRAJ_DIR/f_tumvi_room${room}_stereoi_run${run}.txt"
        TRAJ_SEC="$TRAJ_DIR/f_tumvi_room${room}_stereoi_run${run}_sec.txt"

        echo "Evaluating room${room}, run${run}..."

        # Convert ORB-SLAM3 timestamps from ns to seconds
        awk '{
            printf "%.9f", $1/1e9;
            for (i=2; i<=NF; i++) printf " %s", $i;
            printf "\n"
        }' "$TRAJ" > "$TRAJ_SEC"

        RMSE=$(
            evo_ape euroc \
                "$GT" \
                "$TRAJ_SEC" \
                -a 2>&1 |
            awk '$1 == "rmse" {print $2}'
        )

        if [[ -z "$RMSE" ]]; then
            echo "ERROR: failed to obtain RMSE for room${room} run${run}"
            exit 1
        fi

        echo "$room,$run,$RMSE" >> "$OUT"
        echo "  RMSE = $RMSE m"
    done
done

echo
echo "Individual results:"
column -s, -t "$OUT"

echo
echo "Median RMSE for each room:"
printf "%-8s %-12s\n" "Room" "Median ATE"

for room in 1 2 3 4 5 6; do
    MEDIAN=$(
        awk -F, -v r="$room" '$1 == r {print $3}' "$OUT" |
        sort -n |
        sed -n '2p'
    )

    printf "room%-3s  %.6f m\n" "$room" "$MEDIAN"
done
