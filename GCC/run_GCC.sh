#!/bin/bash

networks=("strike" "karate" "mexican" "dolphin" "polbooks" "football"
          "texas" "cornell" "wisconsin" "polblogs" "facebook"
          "brightkite" "gowalla" "amazon")

MAX_PARALLEL=2  # 한 번에 실행할 최대 프로세스 수
running_jobs=0

for net in "${networks[@]}"; do
    for b in {1..10}; do
        ./core $net $b &  # 백그라운드 실행
        ((running_jobs++))

        # 실행 중인 프로세스가 MAX_PARALLEL 개수 이상이면 대기
        if [[ $running_jobs -ge $MAX_PARALLEL ]]; then
            wait -n  # 하나의 프로세스가 끝날 때까지 대기
            ((running_jobs--))
        fi
    done
done

wait  # 모든 프로세스가 끝날 때까지 대기