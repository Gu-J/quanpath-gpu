#!/bin/bash

MAX_VAL=18446744073709551615  # 64位无符号最大值

prev_tx=0
prev_rx=0

echo "timestamp           TX_GB/s    RX_GB/s"

while true; do
    ts=$(date "+%Y-%m-%d %H:%M:%S")

    # 一次采样并提取最后GPU 0数据
    read tx rx <<< $(dcgmi dmon -e 1009,1010 -i 0 -c 1 | awk '/GPU 0/ {tx=$3; rx=$4} END{print tx, rx}')
    # 跳过无效数据
    if [[ -z "$tx" || -z "$rx" ]]; then
        echo "$ts   No data"
        sleep 1
        continue
    fi

    # 初始化
    if [[ $prev_tx -eq 0 && $prev_rx -eq 0 ]]; then
        prev_tx=$tx
        prev_rx=$rx
        sleep 1
        continue
    fi

    # 差值计算，考虑回绕
    diff_tx=$((tx - prev_tx))
    if (( diff_tx < 0 )); then
        diff_tx=$(( diff_tx + MAX_VAL + 1 ))
    fi

    diff_rx=$((rx - prev_rx))
    if (( diff_rx < 0 )); then
        diff_rx=$(( diff_rx + MAX_VAL + 1 ))
    fi

    # 转GB/s (1 GB = 1024^3 = 1073741824 bytes)
    tx_gbs=$(awk "BEGIN {printf \"%.6f\", $diff_tx/1073741824}")
    rx_gbs=$(awk "BEGIN {printf \"%.6f\", $diff_rx/1073741824}")

    echo "$ts   $tx_gbs   $rx_gbs"

    prev_tx=$tx
    prev_rx=$rx

    sleep 1
done
