#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

# Prevent stale user-site packages from shadowing the CUDA environment (for
# example, ~/.local/safetensors overriding the vLLM environment version).
export PYTHONNOUSERSITE=1

PROJECT_ROOT="/vast/projects/myatskar/design-documents/joseph/fine-tuning-or-retrieval"
VLLM_ENV="/vast/projects/myatskar/design-documents/conda_env/vllm"
CLIENT_ENV="/vast/projects/myatskar/design-documents/conda_env/finetuning"

MODEL="openai/gpt-oss-20b"
DOMAIN="arxiv"
ITEM=""
PARTS="all"
MODEL_SLUG=""
REASONING_EFFORT=""
MAX_WORKERS=2
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.90
READY_TIMEOUT_SECONDS=1800
REASONING_PARSER=""
MAX_NUM_SEQS=""
MAX_TOKENS=""
ENABLE_THINKING=""
TEMPERATURE=""
TOP_P=""
TOP_K=""
REPETITION_PENALTY=""
MIN_P=""
COMPACT_PROSE=""
QUANTIZATION=""
LIMIT_MM_PER_PROMPT=""
TRUST_REMOTE_CODE=""
ENABLE_EXPERT_PARALLEL=""
LANGUAGE_MODEL_ONLY=""
MAX_NUM_BATCHED_TOKENS=""
MOE_BACKEND=""
TOOL_CALL_PARSER=""
ENABLE_AUTO_TOOL_CHOICE=""
DISABLE_FLASHINFER_AUTOTUNE=""
ENABLE_CHUNKED_PREFILL=""
SMOKE_TEST_MODE="standard"

usage() {
    echo "Usage: $0 [options]"
    echo "  --model ID                    Hugging Face model id"
    echo "  --domain arxiv|medical|legal|all  Multiview pipeline(s)"
    echo "  --item NAME                   Paper or case basename"
    echo "  --parts CSV                   all or comma-separated view names"
    echo "  --model-slug NAME             Output model folder (derived when omitted)"
    echo "  --reasoning-effort LEVEL      low, medium, high, or omitted for custom"
    echo "  --max-workers N               Concurrent generation requests"
    echo "  --tensor-parallel-size N      vLLM tensor parallel workers"
    echo "  --max-model-len N             Served context limit"
    echo "  --max-tokens N                Per-generation output budget (clamped to fit)"
    echo "  --gpu-memory-utilization F    vLLM GPU memory fraction"
    echo "  --reasoning-parser NAME       vLLM reasoning parser (e.g. qwen3); omit to disable"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --domain) DOMAIN="$2"; shift 2 ;;
        --item) ITEM="$2"; shift 2 ;;
        --parts) PARTS="$2"; shift 2 ;;
        --model-slug) MODEL_SLUG="$2"; shift 2 ;;
        --reasoning-effort) REASONING_EFFORT="$2"; shift 2 ;;
        --max-workers) MAX_WORKERS="$2"; shift 2 ;;
        --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --reasoning-parser) REASONING_PARSER="$2"; shift 2 ;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
        --enable-thinking) ENABLE_THINKING="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --repetition-penalty) REPETITION_PENALTY="$2"; shift 2 ;;
        --min-p) MIN_P="$2"; shift 2 ;;
        --compact-prose) COMPACT_PROSE="$2"; shift 2 ;;
        --quantization) QUANTIZATION="$2"; shift 2 ;;
        --limit-mm-per-prompt) LIMIT_MM_PER_PROMPT="$2"; shift 2 ;;
        # Boolean serve flags take an explicit 1/0 value so they survive the submit
        # wrapper, which only forwards "--flag value" pairs (a bare flag would abort it).
        --trust-remote-code) TRUST_REMOTE_CODE="$2"; shift 2 ;;
        --enable-expert-parallel) ENABLE_EXPERT_PARALLEL="$2"; shift 2 ;;
        --language-model-only) LANGUAGE_MODEL_ONLY="$2"; shift 2 ;;
        --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
        --moe-backend) MOE_BACKEND="$2"; shift 2 ;;
        --tool-call-parser) TOOL_CALL_PARSER="$2"; shift 2 ;;
        --enable-auto-tool-choice) ENABLE_AUTO_TOOL_CHOICE="$2"; shift 2 ;;
        --disable-flashinfer-autotune) DISABLE_FLASHINFER_AUTOTUNE="$2"; shift 2 ;;
        --enable-chunked-prefill) ENABLE_CHUNKED_PREFILL="$2"; shift 2 ;;
        --smoke-test-mode) SMOKE_TEST_MODE="$2"; shift 2 ;;
        --ready-timeout-seconds) READY_TIMEOUT_SECONDS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$DOMAIN" in
    arxiv|medical|legal) DOMAINS=("$DOMAIN") ;;
    all) DOMAINS=(arxiv medical legal) ;;
    *) echo "Unsupported domain: $DOMAIN" >&2; exit 2 ;;
esac
case "$SMOKE_TEST_MODE" in
    standard|quick) ;;
    *) echo "Unsupported smoke-test mode: $SMOKE_TEST_MODE" >&2; exit 2 ;;
esac

for value in "$MAX_WORKERS" "$TENSOR_PARALLEL_SIZE"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Expected a positive integer, got: $value" >&2
        exit 2
    fi
done
# --max-model-len accepts a positive integer or the literal "auto" (vLLM derives the
# model's native context window — used for the MoE models whose context we don't pin).
if [[ "$MAX_MODEL_LEN" != "auto" && ! "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]]; then
    echo "Expected a positive integer or 'auto' for --max-model-len, got: $MAX_MODEL_LEN" >&2
    exit 2
fi

mkdir -p "$PROJECT_ROOT/logs/vllm"
JOB_TAG="${SLURM_JOB_ID:-manual_$$}"
SERVER_LOG="$PROJECT_ROOT/logs/vllm/${JOB_TAG}_server.log"
PIPELINE_LOG="$PROJECT_ROOT/logs/vllm/${JOB_TAG}_pipeline.log"
VLLM_PYTHON="$VLLM_ENV/bin/python"
VLLM_BIN="$VLLM_ENV/bin/vllm"
CLIENT_PYTHON="$CLIENT_ENV/bin/python"

# Triton and TorchInductor compile artifacts are node-specific scratch data.  A
# shared NFS cache can produce stale-file-handle failures when several Gemma jobs
# compile the same kernels concurrently, so keep these caches allocation-local.
LOCAL_COMPILE_ROOT="${SLURM_TMPDIR:-/tmp}/vllm_compile_${JOB_TAG}"
mkdir -p "$LOCAL_COMPILE_ROOT/triton" "$LOCAL_COMPILE_ROOT/torchinductor" "$LOCAL_COMPILE_ROOT/vllm"
export TRITON_CACHE_DIR="$LOCAL_COMPILE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_COMPILE_ROOT/torchinductor"
export VLLM_CACHE_ROOT="$LOCAL_COMPILE_ROOT/vllm"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$LOCAL_COMPILE_ROOT/flashinfer_autotune"

if [[ ! -x "$VLLM_PYTHON" || ! -x "$VLLM_BIN" ]]; then
    echo "Missing vLLM runtime under $VLLM_ENV" >&2
    exit 1
fi
if [[ ! -x "$CLIENT_PYTHON" ]]; then
    echo "Missing pipeline runtime under $CLIENT_ENV" >&2
    exit 1
fi

echo "Preflighting CUDA runtime on $(hostname)"
TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" "$VLLM_PYTHON" - <<'PY'
import os
import sys
import torch
import vllm

tp = int(os.environ["TENSOR_PARALLEL_SIZE"])
print(f"vLLM: {vllm.__version__}")
print(f"PyTorch: {torch.__version__} (CUDA build {torch.version.cuda})")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Visible CUDA devices: {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    sys.exit("CUDA is unavailable inside the SLURM allocation")
if torch.cuda.device_count() < tp:
    sys.exit(f"tensor parallel size {tp} exceeds {torch.cuda.device_count()} visible GPUs")
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
PY
nvidia-smi

PORT="$($VLLM_PYTHON - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

if [[ -z "$MODEL_SLUG" ]]; then
    BASE_SLUG="$(PYTHONPATH="$PROJECT_ROOT" "$CLIENT_PYTHON" -c 'import sys; from utils.utils import model_slug; print(model_slug(sys.argv[1]))' "$MODEL")"
    MODE="${REASONING_EFFORT:-custom}"
    MODEL_SLUG="${BASE_SLUG}_vllm_${MODE}"
fi

EXTRA_SERVE_ARGS=()
if [[ -n "$REASONING_PARSER" ]]; then
    EXTRA_SERVE_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ -n "$MAX_NUM_SEQS" ]]; then
    EXTRA_SERVE_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
    EXTRA_SERVE_ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi
if [[ -n "$MOE_BACKEND" ]]; then
    EXTRA_SERVE_ARGS+=(--moe-backend "$MOE_BACKEND")
fi
if [[ -n "$QUANTIZATION" ]]; then
    EXTRA_SERVE_ARGS+=(--quantization "$QUANTIZATION")
fi
if [[ -n "$LIMIT_MM_PER_PROMPT" ]]; then
    EXTRA_SERVE_ARGS+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT")
fi
if [[ -n "$TOOL_CALL_PARSER" ]]; then
    EXTRA_SERVE_ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]]; then
    EXTRA_SERVE_ARGS+=(--enable-auto-tool-choice)
fi
if [[ "$DISABLE_FLASHINFER_AUTOTUNE" == "1" ]]; then
    EXTRA_SERVE_ARGS+=(--kernel-config '{"enable_flashinfer_autotune":false}')
fi
if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
    EXTRA_SERVE_ARGS+=(--enable-chunked-prefill)
fi
# MoE serve toggles (value 1 enables the corresponding boolean vLLM flag).
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    EXTRA_SERVE_ARGS+=(--trust-remote-code)
fi
if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then
    EXTRA_SERVE_ARGS+=(--enable-expert-parallel)
fi
if [[ "$LANGUAGE_MODEL_ONLY" == "1" ]]; then
    EXTRA_SERVE_ARGS+=(--language-model-only)
fi

# vLLM's API server waits VLLM_ENGINE_READY_TIMEOUT_S (default 600s) for the engine
# core to start. Large models with a cold torch.compile cache (FXGraphCacheMiss) can
# exceed that and abort before the bash readiness loop below ever runs. Give the engine
# the same budget as READY_TIMEOUT_SECONDS unless the caller overrode it.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-$READY_TIMEOUT_SECONDS}"

echo "Launching $MODEL at $VLLM_BASE_URL (reasoning-parser=${REASONING_PARSER:-none})"
echo "Server log: $SERVER_LOG"
echo "Engine ready timeout: VLLM_ENGINE_READY_TIMEOUT_S=$VLLM_ENGINE_READY_TIMEOUT_S"
VLLM_COMMAND=("$VLLM_BIN")
if [[ "$QUANTIZATION" == "modelopt" ]]; then
    VLLM_COMMAND=("$VLLM_PYTHON" "$PROJECT_ROOT/scripts/vllm_modelopt_tied_lm_head.py")
    echo "ModelOpt tied-LM-head compatibility wrapper enabled"
fi
"${VLLM_COMMAND[@]}" serve "$MODEL" \
    --served-model-name "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-prefix-caching \
    "${EXTRA_SERVE_ARGS[@]}" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
until VLLM_API_KEY="$VLLM_API_KEY" "$VLLM_PYTHON" - "$VLLM_BASE_URL/models" <<'PY'
import json
import os
import sys
import urllib.request
request = urllib.request.Request(
    sys.argv[1],
    headers={"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"},
)
with urllib.request.urlopen(request, timeout=5) as response:
    payload = json.load(response)
if not payload.get("data"):
    raise SystemExit(1)
PY
do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "vLLM exited before becoming ready; see $SERVER_LOG" >&2
        tail -n 100 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for vLLM; see $SERVER_LOG" >&2
        exit 1
    fi
    sleep 15
done

if [[ "$MOE_BACKEND" == "flashinfer_trtllm_afp8" ]]; then
    if ! grep -q "FLASHINFER_TRTLLM_MXFP4_MXFP8" "$SERVER_LOG"; then
        echo "Requested FlashInfer AFP8 backend was not activated; refusing generation" >&2
        tail -n 100 "$SERVER_LOG" >&2 || true
        exit 1
    fi
fi

echo "Running endpoint smoke tests (mode=$SMOKE_TEST_MODE)"
MODEL="$MODEL" VLLM_BASE_URL="$VLLM_BASE_URL" VLLM_API_KEY="$VLLM_API_KEY" SMOKE_TEST_MODE="$SMOKE_TEST_MODE" "$VLLM_PYTHON" - <<'PY'
import json
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["VLLM_BASE_URL"],
    api_key=os.environ["VLLM_API_KEY"],
    timeout=600,
)
model = os.environ["MODEL"]
if os.environ["SMOKE_TEST_MODE"] == "quick":
    plain = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with one complete sentence confirming that the endpoint is ready."}],
        max_tokens=4096,
    )
    if not (plain.choices[0].message.content or "").strip():
        raise RuntimeError("quick prose smoke completion was empty")
    structured = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Return JSON with key ready and boolean value true."}],
        response_format={"type": "json_object"}, max_tokens=8192,
    )
    payload = json.loads(structured.choices[0].message.content or "")
    if payload.get("ready") is not True:
        raise RuntimeError(f"unexpected quick JSON smoke response: {payload}")
    print("vLLM quick smoke tests passed")
    raise SystemExit(0)
for effort in ("low", "high"):
    plain = client.chat.completions.create(
        model=model, reasoning_effort=effort,
        messages=[{"role": "user", "content": "Write at least 300 coherent words explaining why transactional file publication prevents stale partial outputs."}],
        max_tokens=12288 if effort == "low" else 24576,
    )
    if len((plain.choices[0].message.content or "").split()) < 250:
        raise RuntimeError(f"{effort} long-prose smoke completion was empty or short")
    structured = client.chat.completions.create(
        model=model, reasoning_effort=effort,
        messages=[{"role": "user", "content": "Return JSON with key ready and boolean value true."}],
        response_format={"type": "json_object"}, max_tokens=16384 if effort == "low" else 32768,
    )
    payload = json.loads(structured.choices[0].message.content or "")
    if payload.get("ready") is not True:
        raise RuntimeError(f"unexpected {effort} JSON smoke response: {payload}")
print("vLLM smoke tests passed")
PY

# Per-generation output budget: exported so the multiview pipeline (MULTIVIEW_MAX_TOKENS)
# and the vLLM truncation-growth cap (VLLM_MAX_MAX_TOKENS) both honor the requested size.
# utils clamps an over-large request down to the room left by the prompt.
if [[ -n "$MAX_TOKENS" ]]; then
    if ! [[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
        echo "Expected a positive integer for --max-tokens, got: $MAX_TOKENS" >&2
        exit 2
    fi
    export MULTIVIEW_MAX_TOKENS="$MAX_TOKENS"
    export VLLM_MAX_MAX_TOKENS="$MAX_TOKENS"
    echo "Per-generation output budget: MULTIVIEW_MAX_TOKENS=$MAX_TOKENS (clamped to fit context)"
fi

if [[ -n "$ENABLE_THINKING" ]]; then
    export QWEN_ENABLE_THINKING="$ENABLE_THINKING"
    echo "Qwen thinking: QWEN_ENABLE_THINKING=$ENABLE_THINKING"
fi
if [[ -n "$TEMPERATURE" ]]; then
    export QWEN_TEMPERATURE="$TEMPERATURE"
    export VLLM_TEMPERATURE="$TEMPERATURE"
    echo "vLLM sampling: temperature=$TEMPERATURE"
fi
if [[ -n "$TOP_P" ]]; then
    export QWEN_TOP_P="$TOP_P"
    export VLLM_TOP_P="$TOP_P"
    echo "vLLM sampling: top_p=$TOP_P"
fi
if [[ -n "$TOP_K" ]]; then
    export VLLM_TOP_K="$TOP_K"
    echo "vLLM sampling: VLLM_TOP_K=$TOP_K"
fi
if [[ -n "$REPETITION_PENALTY" ]]; then
    export QWEN_REPETITION_PENALTY="$REPETITION_PENALTY"
    echo "Qwen sampling: QWEN_REPETITION_PENALTY=$REPETITION_PENALTY"
fi
if [[ -n "$MIN_P" ]]; then
    export QWEN_MIN_P="$MIN_P"
    echo "Qwen sampling: QWEN_MIN_P=$MIN_P"
fi
if [[ -n "$COMPACT_PROSE" ]]; then
    export QWEN_COMPACT_PROSE="$COMPACT_PROSE"
    echo "Qwen compact prose: QWEN_COMPACT_PROSE=$COMPACT_PROSE"
fi

for CURRENT_DOMAIN in "${DOMAINS[@]}"; do
    case "$CURRENT_DOMAIN" in
        arxiv)
            PIPELINE="$PROJECT_ROOT/scripts/data-preparation/multiview/pipeline_multi_view_knowledge_arxiv.py"
            ITEM_FLAG="--papers"
            ;;
        medical)
            PIPELINE="$PROJECT_ROOT/scripts/data-preparation/multiview/pipeline_multi_view_knowledge_medical.py"
            ITEM_FLAG="--cases"
            ;;
        legal)
            PIPELINE="$PROJECT_ROOT/scripts/data-preparation/multiview/pipeline_multi_view_knowledge_legal.py"
            ITEM_FLAG="--cases"
            ;;
    esac
    IFS=',' read -r -a PART_ARGS <<< "$PARTS"
    PIPELINE_ARGS=(
        "$PIPELINE"
        --parts "${PART_ARGS[@]}"
        --model "$MODEL"
        --model_slug "$MODEL_SLUG"
        --provider vllm
        --base_url "$VLLM_BASE_URL"
        --max_workers "$MAX_WORKERS"
    )
    if [[ -n "$ITEM" ]]; then
        IFS=',' read -r -a ITEM_ARR <<< "$ITEM"
        PIPELINE_ARGS+=("$ITEM_FLAG" "${ITEM_ARR[@]}")
    fi
    if [[ -n "$REASONING_EFFORT" ]]; then
        PIPELINE_ARGS+=(--reasoning_effort "$REASONING_EFFORT")
    fi
    PIPELINE_LOG="$PROJECT_ROOT/logs/vllm/${JOB_TAG}_${CURRENT_DOMAIN}_pipeline.log"
    echo "Running $CURRENT_DOMAIN multiview pipeline for $ITEM (parts=$PARTS, slug=$MODEL_SLUG)"
    echo "Pipeline log: $PIPELINE_LOG"
    PYTHONPATH="$PROJECT_ROOT" "$CLIENT_PYTHON" "${PIPELINE_ARGS[@]}" \
        2>&1 | tee "$PIPELINE_LOG"
done

echo "Multiview generation completed successfully"
