#!/usr/bin/env bash
# Run one arm with one persistent vLLM instance. Requires Linux, Bash, curl and
# setsid, plus a working vLLM/Mooncake installation and local Qwen weights.
# Usage: bash task_a_run.sh store /path/to/new-run --lengths 512 --repetitions 1
set -euo pipefail
umask 077
mode=${1:?usage: task_a_run.sh store|recompute NEW_RUN_DIRECTORY [benchmark options]}
case "$mode" in store|recompute) ;; *) echo "Unknown mode: $mode" >&2; exit 2;; esac
output=${2:?missing new run directory}
shift 2
scripts=$(cd -- "$(dirname -- "$0")" && pwd)
env_file=${TASK_A_ENV:-"$HOME/kv-cache-lab/.local/deployment/env.sh"}
if [[ -r "$env_file" ]]; then source "$env_file"; fi
model=${TASK_A_MODEL:-"${DATA:-}/models/Qwen2.5-1.5B-Instruct"}
test -r "$model/config.json" || { echo "Set TASK_A_MODEL to the checkpoint directory" >&2; exit 1; }
client="$scripts/task_a_benchmark.py"
test -r "$client" || { echo "Missing benchmark client: $client" >&2; exit 1; }
port=${TASK_A_PORT:-18102}
blocks=${TASK_A_GPU_BLOCKS:-160}
batch_tokens=${TASK_A_BATCH_TOKENS:-2048}
owner_bytes=${TASK_A_OWNER_BYTES:-4294967296}
master_port=52063
client_port=52064
owner_port=52065
master_metrics=52903
owner_health=52930

mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"   # Preserve any results from a previous invocation.
output=$(cd -- "$output" && pwd)
tag=$(python -c 'import uuid; print(uuid.uuid4())')
export TASK_A_SERVE_LOG="$output/serve.log"
export VLLM_RPC_BASE_PATH
VLLM_RPC_BASE_PATH=$(mktemp -d "${TMPDIR:-/tmp}/task-a.XXXXXXXX")
export VLLM_HOST_IP=127.0.0.1 MOONCAKE_REQUESTER_LOCAL_HOSTNAME=127.0.0.1
export MC_FORCE_TCP=1 MC_STORE_MEMCPY=0 PYTHONHASHSEED=0
unset MC_USE_TENT MC_USE_TEV1 MC_USE_NVLINK_IPC

# Each service starts in a new session. Only children bearing this run's random
# marker and belonging to a recorded session may receive shutdown signals.
declare -a pids=()
start_service() {
    local name=$1
    shift
    setsid env "TASK_A_SERVICE_RUN=$tag" "$@" > "$output/$name.log" 2>&1 </dev/null &
    service_pid=$!
    pids+=("$service_pid")
}
cleanup() {
    local status=$? clean_status=0
    trap - EXIT
    trap '' INT TERM HUP
    python - "$tag" "${pids[@]}" <<'PY' || clean_status=1
import os
import platform
from pathlib import Path
import signal
import sys
import time

marker = ("TASK_A_SERVICE_RUN=" + sys.argv[1]).encode()
sessions = {int(pid) for pid in sys.argv[2:]}

# Some Python distributions were built with headers predating Linux pidfds.
if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
    open_pid = os.pidfd_open
    send_pid = signal.pidfd_send_signal
elif platform.machine() == "x86_64":
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)

    def syscall(number, *args):
        value = libc.syscall(number, *args)
        if value == -1:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return value

    def open_pid(pid):
        return syscall(434, pid, 0)

    def send_pid(fd, sig):
        return syscall(424, fd, sig, 0, 0)
else:
    raise RuntimeError("This build needs Python with Linux pidfd support")

def members():
    found = []
    for p in Path("/proc").iterdir():
        if not p.name.isdecimal():
            continue
        try:
            fields = (p / "stat").read_text().rsplit(") ", 1)[1].split()
            if int(fields[3]) not in sessions or fields[0] == "Z":
                continue
            if p.stat().st_uid != os.getuid():
                raise RuntimeError(f"Foreign uid in recorded session: {p}")
            if marker not in (p / "environ").read_bytes().split(b"\0"):
                raise RuntimeError(f"Run marker mismatch: {p}")
            found.append(int(p.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # A terminating child can transiently deny access to environ.
            # Keep an unreadable session member pending. The signal path checks
            # its marker again; polling below handles the exit-to-zombie race.
            found.append(int(p.name))
    return found

for sig, seconds in [(signal.SIGTERM, 15), (signal.SIGKILL, 5)]:
    for pid in members():
        try:
            # Revalidate the tagged process after binding an unambiguous pidfd.
            fd = open_pid(pid)
            try:
                proc = Path("/proc", str(pid))
                if marker in (proc / "environ").read_bytes().split(b"\0"):
                    send_pid(fd, sig)
            finally:
                os.close(fd)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    until = time.monotonic() + seconds
    while members() and time.monotonic() < until:
        time.sleep(0.2)
    if not members():
        break
else:
    raise SystemExit("Tagged services still active")
PY
    if ((clean_status == 0)); then
        # The directory came from mktemp above and belongs to this arm.
        find "$VLLM_RPC_BASE_PATH" -depth -delete || clean_status=1
    fi
    printf '{"client_or_startup_exit":%d,"cleanup_exit":%d}\n' "$status" "$clean_status" > "$output/status.json"
    if ((status == 0)); then status=$clean_status; fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

wait_http() {
    local pid=$1 url=$2 deadline=$((SECONDS + $3))
    until curl -fsS --max-time 2 --connect-timeout 1 "$url" >/dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null || ((SECONDS >= deadline)); then
            echo "Service failed readiness: $url; logs in $output" >&2
            return 1
        fi
        sleep 1
    done
    kill -0 "$pid"
}

ports=("$port")
if [[ $mode == store ]]; then ports+=("$master_port" "$client_port" "$owner_port" "$master_metrics" "$owner_health"); fi
python - "${ports[@]}" <<'PY'
import socket, sys
for port in map(int, sys.argv[1:]):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
PY
nvidia-smi --query-gpu=uuid,name,driver_version,memory.total,memory.free,utilization.gpu --format=csv > "$output/gpu-before.csv"

args=(--model "$model" --served-model-name week1-qwen --dtype half --tensor-parallel-size 1
      --max-model-len 2048 --max-num-seqs 1 --max-num-batched-tokens "$batch_tokens" --block-size 16
      --num-gpu-blocks-override "$blocks" --gpu-memory-utilization 0.3 --enforce-eager --seed 0
      --enable-prefix-caching --prefix-caching-hash-algo sha256 --host 127.0.0.1 --port "$port")
if [[ ${TASK_A_DIAGNOSTICS:-0} == 1 ]]; then
    export TASK_A_DIAGNOSTIC_DIR="$output/diagnostics"
    export PYTHONPATH="$scripts${PYTHONPATH:+:$PYTHONPATH}"
    mkdir "$TASK_A_DIAGNOSTIC_DIR"
    args+=(--worker-extension-cls task_a_diagnostics.DiagnosticExtension)
fi
if [[ $mode == store ]]; then
    export MOONCAKE_CONFIG_PATH="$output/store.json"
    cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{"metadata_server":"P2PHANDSHAKE","master_server_address":"127.0.0.1:$master_port","mode":"standalone-store","protocol":"tcp","device_name":"","global_segment_size":0,"local_buffer_size":268435456,"enable_offload":false}
EOF
    export VLLM_MOONCAKE_STORE_TIER_LOG=1
    start_service master mooncake_master --rpc_address=127.0.0.1 --rpc_port="$master_port" \
        --rpc_thread_num=2 --handler_worker_thread_num=2 --metrics_host=127.0.0.1 \
        --metrics_port="$master_metrics" --enable_offload=false
    wait_http "$service_pid" "http://127.0.0.1:$master_metrics/metrics" 45
    start_service owner env MC_STORE_CLIENT_METRIC=1 mooncake_client \
        --host="127.0.0.1:$owner_port" --port="$client_port" --metadata_server=P2PHANDSHAKE \
        --master_server_address="127.0.0.1:$master_port" --protocol=tcp --device_names='' \
        --global_segment_size="$owner_bytes" --local_buffer_size=0 --enable_offload=false \
        --enable_http_server=true --http_port="$owner_health"
    wait_http "$service_pid" "http://127.0.0.1:$owner_health/health" 60
    if [[ ${TASK_A_ROUTING:-0} == 1 ]]; then
        export PYTHONPATH="$scripts${PYTHONPATH:+:$PYTHONPATH}"
        args+=(--kv-transfer-config '{"kv_connector":"LengthRoutedMooncakeStoreConnector","kv_connector_module_path":"task_a_routing","kv_role":"kv_both"}')
    else
        args+=(--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}')
    fi
fi
start_service serve python -m vllm.entrypoints.openai.api_server "${args[@]}"
api="http://127.0.0.1:$port"
wait_http "$service_pid" "$api/health" 360
python - "$output" "$mode" "$blocks" "$owner_bytes" "$tag" "$model" "$batch_tokens" "$@" <<'PY'
import json, os, re, sys
from pathlib import Path
out, mode, blocks, owner_bytes, tag, model, batch_tokens, *options = sys.argv[1:]
path = Path(out)
match = re.search(r"GPU KV cache size: ([\d,]+) tokens", (path / "serve.log").read_text(errors="replace"))
tokens = int(match[1].replace(",", "")) if match else None
if tokens != int(blocks) * 16:
    raise SystemExit(f"Expected {blocks} physical blocks of size 16, observed token capacity {tokens}")
config = dict(arm=mode, run_id=tag, model=model, dtype="float16", gpu_blocks=int(blocks),
              block_tokens=16, gpu_cache_tokens=tokens, max_model_len=2048, max_num_seqs=1,
              gpu_memory_utilization=0.3, enable_prefix_caching=True, hash_algorithm="sha256",
              max_num_batched_tokens=int(batch_tokens),
              diagnostics_enabled=os.environ.get("TASK_A_DIAGNOSTICS") == "1",
              batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1",
              store_owner_bytes=int(owner_bytes) if mode=="store" else 0, client_args=options)
(path / "run.json").write_text(json.dumps(config, indent=2) + "\n")
PY
python "$client" --mode "$mode" --output "$output/measurements.jsonl" --api "$api" --model week1-qwen "$@"
