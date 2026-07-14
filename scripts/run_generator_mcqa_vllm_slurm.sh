#!/bin/bash
#SBATCH --job-name=gen_mcqa_vllm
#SBATCH --output=/vast/projects/myatskar/design-documents/joseph/generator_mcqa/logs/slurm-%x-%j.out
#SBATCH --error=/vast/projects/myatskar/design-documents/joseph/generator_mcqa/logs/slurm-%x-%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --mem=1792G
#SBATCH --time=1-00:00:00

# SLURM port of scripts/run_generator_mcqa_benchmark.sh --local-only for the
# PARCC cluster: serves one generator checkpoint with vLLM on a dgx-b200 node
# and runs the constrained MCQA evaluation against it.
#
# Usage: sbatch scripts/run_generator_mcqa_vllm_slurm.sh <gemma_4_12b|glm_5_nvfp4>
#
# State is written with --no-summary; reports/generator_mcqa/accuracies.csv is
# regenerated only on the canonical machine via --import-state.

set -euo pipefail

MODEL_KEY="${1:?usage: sbatch run_generator_mcqa_vllm_slurm.sh <model_key>}"

REPO_ROOT="/vast/projects/myatskar/design-documents/joseph/fine-tuning-or-retrieval"
WORK_ROOT="/vast/projects/myatskar/design-documents/joseph/generator_mcqa"
STATE_ROOT="${WORK_ROOT}/state"
SMOKE_STATE_ROOT="${WORK_ROOT}/smoke"
LOG_DIR="${WORK_ROOT}/logs"
PID_DIR="${WORK_ROOT}/pids"

VLLM="/vast/projects/myatskar/design-documents/conda_env/vllm/bin/vllm"
PYTHON="/vast/projects/myatskar/design-documents/conda_env/tuning/bin/python"
EVALUATOR="${REPO_ROOT}/scripts/evaluate_generator_mcqa.py"

VLLM_HOST="127.0.0.1"
VLLM_PORT="8000"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
STARTUP_TIMEOUT_SECONDS=7200

export HF_HOME="/vast/projects/myatskar/design-documents/hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/${USER}-gen-mcqa-${SLURM_JOB_ID:-manual}/xdg"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/${USER}-gen-mcqa-${SLURM_JOB_ID:-manual}/triton"
export VLLM_ENGINE_READY_TIMEOUT_S=7200
mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" \
  "${STATE_ROOT}" "${SMOKE_STATE_ROOT}" "${LOG_DIR}" "${PID_DIR}"

# Parallelism follows the GPU allocation so the job can shrink to a partial
# node (e.g. sbatch --gpus=4 --cpus-per-task=112 --mem=896G). GLM-5-NVFP4 is
# 481GB of weights, so it needs at least 4 B200s.
NGPUS="${SLURM_GPUS_ON_NODE:-8}"

# Server settings otherwise mirror scripts/generator_mcqa_config.py.
case "${MODEL_KEY}" in
  gemma_4_12b)
    MODEL_ID="google/gemma-4-12B-it"
    DATA_PARALLEL_SIZE="${NGPUS}"
    TENSOR_PARALLEL_SIZE=1
    MAX_NUM_SEQS=32
    MAX_WORKERS=32
    REASONING_PARSER="gemma4"
    QUANTIZATION=""
    TRUST_REMOTE_CODE=true
    ENABLE_EXPERT_PARALLEL=false
    ;;
  glm_5_nvfp4)
    MODEL_ID="nvidia/GLM-5-NVFP4"
    DATA_PARALLEL_SIZE=1
    TENSOR_PARALLEL_SIZE="${NGPUS}"
    MAX_NUM_SEQS=16
    MAX_WORKERS=8
    REASONING_PARSER="glm45"
    QUANTIZATION=""
    TRUST_REMOTE_CODE=true
    ENABLE_EXPERT_PARALLEL=true
    ;;
  *)
    echo "Unsupported model key for this launcher: ${MODEL_KEY}" >&2
    exit 64
    ;;
esac

SERVER_PID=""
SERVER_PID_FILE=""

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

launch_server() {
  local log_path="${LOG_DIR}/${MODEL_KEY}.log"
  local pid_path="${PID_DIR}/${MODEL_KEY}.pid"
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

  if [[ -s "${pid_path}" ]]; then
    local stale_pid
    read -r stale_pid <"${pid_path}" || true
    if [[ "${stale_pid}" =~ ^[0-9]+$ ]] && kill -0 "${stale_pid}" 2>/dev/null; then
      echo "${MODEL_KEY} already appears to be running as PID ${stale_pid} (${pid_path})" >&2
      exit 1
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
  local log_path="${LOG_DIR}/${MODEL_KEY}.log"
  local health_url="http://${VLLM_HOST}:${VLLM_PORT}/health"
  local models_url="${VLLM_BASE_URL}/models"
  local started_at="${SECONDS}"
  local models_response

  printf '==> Waiting for %s on %s (log: %s)\n' "${MODEL_KEY}" "${VLLM_BASE_URL}" "${log_path}"
  while true; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      printf 'vLLM exited before becoming ready; last log lines:\n' >&2
      tail -n 100 -- "${log_path}" >&2 || true
      return 1
    fi
    if curl --fail --silent --show-error --max-time 5 "${health_url}" >/dev/null 2>&1; then
      models_response="$(curl --fail --silent --show-error --max-time 5 "${models_url}" 2>/dev/null || true)"
      if [[ "${models_response}" == *"${MODEL_ID}"* ]]; then
        printf '==> vLLM ready: %s\n' "${MODEL_KEY}"
        return 0
      fi
    fi
    if ((SECONDS - started_at >= STARTUP_TIMEOUT_SECONDS)); then
      printf 'Timed out waiting %s seconds for %s; last log lines:\n' \
        "${STARTUP_TIMEOUT_SECONDS}" "${MODEL_KEY}" >&2
      tail -n 100 -- "${log_path}" >&2 || true
      return 1
    fi
    sleep 5
  done
}

cd "${REPO_ROOT}"

printf '==> Launching vLLM for %s (%s)\n' "${MODEL_KEY}" "${MODEL_ID}"
launch_server
wait_for_server

printf '==> %s smoke test\n' "${MODEL_KEY}"
"${PYTHON}" "${EVALUATOR}" \
  --model-key "${MODEL_KEY}" \
  --protocols constrained \
  --state-root "${SMOKE_STATE_ROOT}" \
  --max-workers "${MAX_WORKERS}" \
  --base-url "${VLLM_BASE_URL}" \
  --limit 1 --no-summary

printf '==> %s full benchmark\n' "${MODEL_KEY}"
"${PYTHON}" "${EVALUATOR}" \
  --model-key "${MODEL_KEY}" \
  --protocols constrained \
  --state-root "${STATE_ROOT}" \
  --max-workers "${MAX_WORKERS}" \
  --base-url "${VLLM_BASE_URL}" \
  --no-summary

printf '==> %s done\n' "${MODEL_KEY}"
