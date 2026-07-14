#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly PYTHON="/data1/joseph/miniconda3/envs/vllm/bin/python"
readonly VLLM="/data1/joseph/miniconda3/envs/vllm/bin/vllm"
readonly EVALUATOR="${REPO_ROOT}/scripts/evaluate_generator_mcqa.py"

readonly WORK_ROOT="/local/joseph/generator_mcqa"
readonly STATE_ROOT="${WORK_ROOT}/state"
readonly SMOKE_STATE_ROOT="${WORK_ROOT}/smoke"
readonly LOG_DIR="${WORK_ROOT}/logs"
readonly PID_DIR="${WORK_ROOT}/pids"
readonly SUMMARY_PATH="${REPO_ROOT}/reports/generator_mcqa/accuracies.csv"
readonly HF_CACHE="/local/joseph/huggingface"
readonly RUNTIME_CACHE="${WORK_ROOT}/cache"

readonly VLLM_HOST="127.0.0.1"
readonly VLLM_PORT="8000"
readonly VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
readonly STARTUP_TIMEOUT_SECONDS="7200"

readonly -a API_MODELS=(
  gpt_5_mini_low
  gpt_5_mini_high
  gpt_5_4_mini_high
  gpt_5_4_mini_low
)

readonly -a LOCAL_MODELS=(
  gpt_oss_20b_low
  gpt_oss_120b_low
  gemma_4_12b
  gemma_4_31b_nvfp4
  glm_5_nvfp4
)

DRY_RUN=false
SMOKE_ONLY=false
SCOPE="all"
SERVER_PID=""
SERVER_PID_FILE=""

declare -a REQUESTED_MODELS=()
declare -a SELECTED_API_MODELS=()
declare -a SELECTED_LOCAL_MODELS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_generator_mcqa_benchmark.sh [OPTIONS]

Run the generator MCQA benchmark, resuming completed state automatically.
Every normal run performs a one-question-per-family answer-only smoke test before
the full evaluation. Local vLLM models are served and evaluated sequentially.

Options:
  --dry-run                 Print commands without running them.
  --smoke                   Run only the smoke evaluations.
  --models MODEL [...]      Run selected model keys. May be repeated; comma-
                            separated lists are also accepted.
  --api-only                Run only selected/default OpenAI models.
  --local-only              Run only selected/default local vLLM models.
  -h, --help                Show this help.

Runnable model keys:
  gpt_5_mini_low, gpt_5_mini_high
  gpt_5_4_mini_high, gpt_5_4_mini_low
  gpt_oss_20b_low, gpt_oss_120b_low
  gemma_4_12b, gemma_4_31b_nvfp4, glm_5_nvfp4

GLM-5.2 NVFP4 is external-only and is intentionally refused by this runner.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

print_command() {
  printf 'DRY-RUN:'
  printf ' %q' "$@"
  printf '\n'
}

run_command() {
  if [[ "${DRY_RUN}" == true ]]; then
    print_command "$@"
  else
    "$@"
  fi
}

append_model_spec() {
  local spec="$1"
  local -a pieces=()
  local piece

  [[ -n "${spec}" && "${spec}" != ,* && "${spec}" != *, && "${spec}" != *,,* ]] || \
    die "--models contains an empty model key"
  IFS=',' read -r -a pieces <<<"${spec}"
  for piece in "${pieces[@]}"; do
    [[ -n "${piece}" ]] || die "--models contains an empty model key"
    REQUESTED_MODELS+=("${piece}")
  done
}

is_api_model() {
  case "$1" in
    gpt_5_mini_low|gpt_5_mini_high|gpt_5_4_mini_high|gpt_5_4_mini_low)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_local_model() {
  case "$1" in
    gpt_oss_20b_low|gpt_oss_120b_low|gemma_4_12b|gemma_4_31b_nvfp4|glm_5_nvfp4)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

parse_args() {
  while (($#)); do
    case "$1" in
      --dry-run)
        DRY_RUN=true
        shift
        ;;
      --smoke)
        SMOKE_ONLY=true
        shift
        ;;
      --api-only)
        [[ "${SCOPE}" != "local" ]] || die "--api-only and --local-only are mutually exclusive"
        SCOPE="api"
        shift
        ;;
      --local-only)
        [[ "${SCOPE}" != "api" ]] || die "--api-only and --local-only are mutually exclusive"
        SCOPE="local"
        shift
        ;;
      --models)
        shift
        (($#)) || die "--models requires at least one model key"
        [[ "$1" != --* ]] || die "--models requires at least one model key"
        while (($#)) && [[ "$1" != --* ]]; do
          append_model_spec "$1"
          shift
        done
        ;;
      --models=*)
        append_model_spec "${1#--models=}"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --*)
        die "unknown option: $1"
        ;;
      *)
        die "unexpected positional argument: $1"
        ;;
    esac
  done
}

select_models() {
  local model
  local provider
  local -A seen=()

  if ((${#REQUESTED_MODELS[@]} == 0)); then
    if [[ "${SCOPE}" != "local" ]]; then
      SELECTED_API_MODELS=("${API_MODELS[@]}")
    fi
    if [[ "${SCOPE}" != "api" ]]; then
      SELECTED_LOCAL_MODELS=("${LOCAL_MODELS[@]}")
    fi
    return
  fi

  for model in "${REQUESTED_MODELS[@]}"; do
    if [[ "${model}" == "glm_5_2_nvfp4" ]]; then
      die "glm_5_2_nvfp4 is external-only; this runner will not invoke the unavailable LiteLLM server"
    fi
    if is_api_model "${model}"; then
      provider="api"
    elif is_local_model "${model}"; then
      provider="local"
    else
      die "unknown model key '${model}'; use --help to list runnable keys"
    fi
    if [[ -n "${seen[${model}]:-}" ]]; then
      continue
    fi
    seen["${model}"]=1

    if [[ "${provider}" == "api" ]]; then
      [[ "${SCOPE}" != "local" ]] || die "${model} is not selectable with --local-only"
      SELECTED_API_MODELS+=("${model}")
    else
      [[ "${SCOPE}" != "api" ]] || die "${model} is not selectable with --api-only"
      SELECTED_LOCAL_MODELS+=("${model}")
    fi
  done
}

model_settings() {
  local model="$1"

  case "${model}" in
    gpt_oss_20b_low)
      MODEL_ID="openai/gpt-oss-20b"
      DATA_PARALLEL_SIZE=8
      TENSOR_PARALLEL_SIZE=1
      MAX_WORKERS=32
      REASONING_PARSER="openai_gptoss"
      MAX_NUM_SEQS=32
      QUANTIZATION=""
      TRUST_REMOTE_CODE=false
      ENABLE_EXPERT_PARALLEL=false
      ;;
    gpt_oss_120b_low)
      MODEL_ID="openai/gpt-oss-120b"
      DATA_PARALLEL_SIZE=4
      TENSOR_PARALLEL_SIZE=2
      MAX_WORKERS=16
      REASONING_PARSER="openai_gptoss"
      MAX_NUM_SEQS=32
      QUANTIZATION=""
      TRUST_REMOTE_CODE=false
      ENABLE_EXPERT_PARALLEL=false
      ;;
    gemma_4_12b)
      MODEL_ID="google/gemma-4-12B-it"
      DATA_PARALLEL_SIZE=8
      TENSOR_PARALLEL_SIZE=1
      MAX_WORKERS=32
      REASONING_PARSER="gemma4"
      MAX_NUM_SEQS=32
      QUANTIZATION=""
      TRUST_REMOTE_CODE=true
      ENABLE_EXPERT_PARALLEL=false
      ;;
    gemma_4_31b_nvfp4)
      MODEL_ID="nvidia/Gemma-4-31B-IT-NVFP4"
      DATA_PARALLEL_SIZE=8
      TENSOR_PARALLEL_SIZE=1
      MAX_WORKERS=32
      REASONING_PARSER="gemma4"
      MAX_NUM_SEQS=32
      QUANTIZATION="modelopt"
      TRUST_REMOTE_CODE=true
      ENABLE_EXPERT_PARALLEL=false
      ;;
    glm_5_nvfp4)
      MODEL_ID="nvidia/GLM-5-NVFP4"
      DATA_PARALLEL_SIZE=1
      TENSOR_PARALLEL_SIZE=8
      MAX_WORKERS=8
      REASONING_PARSER="glm45"
      MAX_NUM_SEQS=16
      QUANTIZATION=""
      TRUST_REMOTE_CODE=true
      ENABLE_EXPERT_PARALLEL=true
      ;;
    *)
      die "internal error: no local settings for ${model}"
      ;;
  esac
}

stop_server() {
  local attempt

  if [[ -n "${SERVER_PID}" ]]; then
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill "${SERVER_PID}" 2>/dev/null || true
      for attempt in {1..30}; do
        kill -0 "${SERVER_PID}" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
      fi
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID_FILE}" ]]; then
    rm -f -- "${SERVER_PID_FILE}"
  fi
  SERVER_PID=""
  SERVER_PID_FILE=""
}

trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

prepare_directories() {
  run_command mkdir -p -- \
    "${STATE_ROOT}" \
    "${SMOKE_STATE_ROOT}" \
    "${LOG_DIR}" \
    "${PID_DIR}" \
    "$(dirname -- "${SUMMARY_PATH}")" \
    "${HF_CACHE}" \
    "${RUNTIME_CACHE}" \
    "${WORK_ROOT}/triton"
}

run_evaluation() {
  local model="$1"
  local workers="$2"
  local state_root="$3"
  local limit="$4"
  local base_url="${5:-}"
  local -a command=(
    "${PYTHON}" "${EVALUATOR}"
    --model-key "${model}"
    --protocols constrained
    --state-root "${state_root}"
    --summary-path "${SUMMARY_PATH}"
    --max-workers "${workers}"
  )

  if [[ -n "${base_url}" ]]; then
    command+=(--base-url "${base_url}")
  fi
  if [[ -n "${limit}" ]]; then
    command+=(--limit "${limit}" --no-summary)
  fi
  run_command "${command[@]}"
}

run_api_model() {
  local model="$1"

  printf '==> OpenAI: %s smoke test\n' "${model}"
  run_evaluation "${model}" 32 "${SMOKE_STATE_ROOT}" 1
  if [[ "${SMOKE_ONLY}" == false ]]; then
    printf '==> OpenAI: %s full benchmark\n' "${model}"
    run_evaluation "${model}" 32 "${STATE_ROOT}" ""
  fi
}

launch_server() {
  local model="$1"
  local log_path="${LOG_DIR}/${model}.log"
  local pid_path="${PID_DIR}/${model}.pid"
  local -a command=(
    "${VLLM}" serve "${MODEL_ID}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --data-parallel-size "${DATA_PARALLEL_SIZE}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --max-model-len 36864
    --gpu-memory-utilization 0.95
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens 8192
    --reasoning-parser "${REASONING_PARSER}"
  )

  if [[ -n "${QUANTIZATION}" ]]; then
    command+=(--quantization "${QUANTIZATION}")
  fi
  if [[ "${TRUST_REMOTE_CODE}" == true ]]; then
    command+=(--trust-remote-code)
  fi
  if [[ "${ENABLE_EXPERT_PARALLEL}" == true ]]; then
    command+=(--enable-expert-parallel)
  fi

  if [[ "${DRY_RUN}" == true ]]; then
    printf 'DRY-RUN:'
    printf ' %q' "${command[@]}"
    printf ' > %q 2>&1 &\n' "${log_path}"
    printf 'DRY-RUN: write server PID to %q\n' "${pid_path}"
    return
  fi

  if [[ -s "${pid_path}" ]]; then
    local stale_pid
    read -r stale_pid <"${pid_path}" || true
    if [[ "${stale_pid}" =~ ^[0-9]+$ ]] && kill -0 "${stale_pid}" 2>/dev/null; then
      die "${model} already appears to be running as PID ${stale_pid} (${pid_path})"
    fi
    rm -f -- "${pid_path}"
  fi

  : >"${log_path}"
  "${command[@]}" >"${log_path}" 2>&1 &
  SERVER_PID=$!
  SERVER_PID_FILE="${pid_path}"
  printf '%s\n' "${SERVER_PID}" >"${SERVER_PID_FILE}"
}

wait_for_server() {
  local model="$1"
  local log_path="${LOG_DIR}/${model}.log"
  local health_url="http://${VLLM_HOST}:${VLLM_PORT}/health"
  local models_url="${VLLM_BASE_URL}/models"
  local started_at="${SECONDS}"
  local models_response

  if [[ "${DRY_RUN}" == true ]]; then
    print_command curl --fail --silent --show-error "${health_url}"
    print_command curl --fail --silent --show-error "${models_url}"
    return
  fi

  printf '==> Waiting for %s on %s (log: %s)\n' "${model}" "${VLLM_BASE_URL}" "${log_path}"
  while true; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      printf 'vLLM exited before becoming ready; last log lines:\n' >&2
      tail -n 100 -- "${log_path}" >&2 || true
      return 1
    fi
    if curl --fail --silent --show-error --max-time 5 "${health_url}" >/dev/null 2>&1; then
      models_response="$(curl --fail --silent --show-error --max-time 5 "${models_url}" 2>/dev/null || true)"
      if [[ "${models_response}" == *"${MODEL_ID}"* ]]; then
        printf '==> vLLM ready: %s\n' "${model}"
        return 0
      fi
    fi
    if ((SECONDS - started_at >= STARTUP_TIMEOUT_SECONDS)); then
      printf 'Timed out waiting %s seconds for %s; last log lines:\n' \
        "${STARTUP_TIMEOUT_SECONDS}" "${model}" >&2
      tail -n 100 -- "${log_path}" >&2 || true
      return 1
    fi
    sleep 5
  done
}

run_local_model() {
  local model="$1"

  model_settings "${model}"
  printf '==> Local vLLM: %s\n' "${model}"
  launch_server "${model}"
  wait_for_server "${model}"

  printf '==> Local vLLM: %s smoke test\n' "${model}"
  run_evaluation "${model}" "${MAX_WORKERS}" "${SMOKE_STATE_ROOT}" 1 "${VLLM_BASE_URL}"
  if [[ "${SMOKE_ONLY}" == false ]]; then
    printf '==> Local vLLM: %s full benchmark\n' "${model}"
    run_evaluation "${model}" "${MAX_WORKERS}" "${STATE_ROOT}" "" "${VLLM_BASE_URL}"
  fi

  if [[ "${DRY_RUN}" == true ]]; then
    printf 'DRY-RUN: stop vLLM server for %s\n' "${model}"
  else
    stop_server
  fi
}

main() {
  local model

  parse_args "$@"
  select_models

  [[ -x "${PYTHON}" ]] || die "Python executable not found: ${PYTHON}"
  [[ -x "${VLLM}" ]] || die "vLLM executable not found: ${VLLM}"
  [[ -f "${EVALUATOR}" ]] || die "evaluator not found: ${EVALUATOR}"

  export HF_HOME="${HF_CACHE}"
  export HF_HUB_CACHE="${HF_CACHE}/hub"
  export XDG_CACHE_HOME="${RUNTIME_CACHE}"
  export TRITON_CACHE_DIR="${WORK_ROOT}/triton"
  export VLLM_ENGINE_READY_TIMEOUT_S="${STARTUP_TIMEOUT_SECONDS}"
  prepare_directories

  for model in "${SELECTED_API_MODELS[@]}"; do
    run_api_model "${model}"
  done
  for model in "${SELECTED_LOCAL_MODELS[@]}"; do
    run_local_model "${model}"
  done
}

main "$@"
