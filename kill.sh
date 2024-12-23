#!/bin/bash

# 좀비 프로세스의 부모 프로세스 ID(PPID) 찾기
zombie_ppids=$(ps -eo stat,ppid | awk '/^Z/ {print $2}' | sort | uniq)

if [ -z "$zombie_ppids" ]; then
    echo "좀비 프로세스가 없습니다."
    exit 0
fi

# 부모 프로세스 강제 종료
echo "다음 부모 프로세스를 종료합니다: $zombie_ppids"
for ppid in $zombie_ppids; do
    echo "종료 중: $ppid"
    sudo kill -9 $ppid 2>/dev/null
done

echo "좀비 프로세스를 처리했습니다."

