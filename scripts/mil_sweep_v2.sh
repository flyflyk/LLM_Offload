#!/usr/bin/env bash
set -euo pipefail

# ===== 參數（可用環境變數覆寫） =====
MODEL="${1:-facebook/opt-30b}"
YAML="${YAML_PATH:-src/configs/accelerate.yaml}"
OFFLOAD_DIR="${OFFLOAD_DIR:-/mnt/ssd/offload_dir}"
BS="${BS:-2}"
GEN="${GEN:-20}"
START_IN="${START_IN:-256}"     # 倍增起點
STEP="${STEP:-64}"              # 二分粒度
MAX_POS="${MAX_POS:-2048}"      # OPT 通常 2048

LOGDIR="logs"
mkdir -p "$LOGDIR"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 備份 YAML，結束時還原
YAML_BAK="${YAML}.bak.$(date +%s)"
cp -f "$YAML" "$YAML_BAK"
restore_yaml() { cp -f "$YAML_BAK" "$YAML" || true; }
trap restore_yaml EXIT

# ---- 寫 YAML：enable_offload=true, enable_kv_offload=<true|false> ----
set_yaml_flags () {
  local kv="$1"   # "true" or "false"
  if command -v yq >/dev/null 2>&1; then
    # yq v4 正確用法：eval (e) + in-place (-i)
    yq e -i ".enable_offload = true | .enable_kv_offload = ${kv}" "$YAML"
  else
    # 後備：用 sed 修改
    sed -i -E 's/^[[:space:]]*enable_offload:[[:space:]]*.*/enable_offload: true/' "$YAML"
    sed -i -E "s/^[[:space:]]*enable_kv_offload:[[:space:]]*.*/enable_kv_offload: ${kv}/" "$YAML"
  fi
}


# ---- 單次執行；回傳 0=成功，1=失敗 ----
run_case () {
  local tag="$1"       # "layer" or "kv"
  local in_len="$2"
  local logfile="$LOGDIR/${MODEL//\//_}_${tag}_bs${BS}_in${in_len}_gen${GEN}.log"

  set +e
  python main.py --mode accelerate \
    --model "$MODEL" \
    --batch-size "$BS" \
    --input-len "$in_len" \
    --gen-len "$GEN" \
    --offload-dir "$OFFLOAD_DIR" > "$logfile" 2>&1
  local rc=$?
  set -e

  # === 判定失敗（避免用前瞻/後顧的 regex）===
  if [[ $rc -ne 0 ]] \
     || grep -qiE 'out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED' "$logfile" \
     || grep -qiE 'max_position_embeddings|maximum (sequence|context) length|sequence length is longer than|input_ids length .* > .* max_position_embeddings|past[_ ]key[_ ]values .* longer than' "$logfile"
  then
    echo "FAIL $tag in_len=$in_len" >&2
    echo "status,fail" > "$logfile.meta"
    return 1
  fi

  # === 擷取 Throughput 與 Total Inference Time（不使用 ?= 前瞻）===
  # 例： "Throughput: 233.65 tokens/sec"
  #      "Total Inference Time: 2.1913s"
  local tp tot
  tp=$(grep -E 'Throughput:' "$logfile" | tail -1 | \
        sed -n 's/.*Throughput:[[:space:]]*\([0-9.]\+\)[[:space:]]*tokens\/sec.*/\1/p')
  tot=$(grep -E 'Total Inference Time:' "$logfile" | tail -1 | \
        sed -n 's/.*Total Inference Time:[[:space:]]*\([0-9.]\+\)s.*/\1/p')

  # 如果抓不到就給 NA（避免空值）
  : "${tp:=NA}"
  : "${tot:=NA}"

  echo "OK   $tag in_len=$in_len (tp=${tp})" >&2
  {
    echo "status,ok"
    echo "throughput,${tp}"
    echo "total_time_s,${tot}"
  } > "$logfile.meta"
  return 0
}



# ---- 二分搜尋 ----
binary_search () {
  local tag="$1"; shift
  local lo="$1"; shift   # 已成功的最大值
  local hi="$1"; shift   # 已失敗的最小值
  local step="$1"; shift

  while (( hi - lo > step )); do
    local mid=$(( (lo + hi) / 2 ))
    if run_case "$tag" "$mid"; then
      lo=$mid
    else
      hi=$mid
    fi
  done
  echo "$lo"
}

# ---- 跑一個模式：回傳 MIL（只輸出數字） ----
sweep_mode () {
  local tag="$1"      # "layer" or "kv"
  local kv_flag="$2"  # "true" or "false"

  echo "==> Configure: enable_offload=true, enable_kv_offload=${kv_flag}" >&2
  set_yaml_flags "$kv_flag"

  local in=$START_IN
  local last_ok=0
  local failed_bound=$((MAX_POS + STEP))

  while (( in <= MAX_POS )); do
    if run_case "$tag" "$in"; then
      last_ok="$in"
      if (( in >= MAX_POS )); then break; fi
      in=$(( in * 2 ))
      if (( in > MAX_POS )); then in=$MAX_POS; fi
    else
      failed_bound="$in"
      break
    fi
  done

  local mil="$last_ok"
  if (( failed_bound <= MAX_POS )); then
    mil=$(binary_search "$tag" "$last_ok" "$failed_bound" "$STEP")
  fi
  echo "$mil"   # 只輸出數字到 stdout
}

echo ">>> Model: $MODEL" >&2
echo "Logs at: $LOGDIR" >&2
echo >&2

MIL_LAYER=$(sweep_mode "layer" "false")
MIL_KV=$(sweep_mode "kv" "true")

SUMMARY="$LOGDIR/summary_${MODEL//\//_}_bs${BS}_gen${GEN}.csv"
echo "model,mode,batch,gen_len,max_input_len" | tee "$SUMMARY" >/dev/null
echo "${MODEL},layer,$BS,$GEN,$MIL_LAYER" | tee -a "$SUMMARY" >/dev/null
echo "${MODEL},kv,$BS,$GEN,$MIL_KV" | tee -a "$SUMMARY" >/dev/null

echo >&2
echo "DONE. Summary: $SUMMARY" >&2