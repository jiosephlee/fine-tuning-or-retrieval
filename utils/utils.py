import os
import re
import json
from functools import lru_cache
from openai import OpenAI

try:
    from utils.keys import OPENAI_API_KEY, DATABRICKS_TOKEN
except (ImportError, ModuleNotFoundError):
    OPENAI_API_KEY = None
    DATABRICKS_TOKEN = None

# LiteLLM (PARCC) key/base — optional; falls back to env vars if not in utils.keys.
try:
    from utils.keys import LITELLM_API_KEY
except (ImportError, ModuleNotFoundError):
    LITELLM_API_KEY = None
try:
    from utils.keys import LITELLM_BASE_URL
except (ImportError, ModuleNotFoundError):
    LITELLM_BASE_URL = None

LITELLM_API_KEY = LITELLM_API_KEY or os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
LITELLM_BASE_URL = LITELLM_BASE_URL or os.environ.get("LITELLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://litellm.parcc.upenn.edu/v1"

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    client = OpenAI()
else:
    client = None

if DATABRICKS_TOKEN:
    client_safe = OpenAI(
        api_key=DATABRICKS_TOKEN,
        base_url="https://adb-4750903324350629.9.azuredatabricks.net/serving-endpoints"
    )
else:
    client_safe = None

# PARCC LiteLLM proxy (OpenAI-compatible). Used automatically for models whose id
# contains a "/" (e.g. "zai-org/GLM-5.2-FP8"). NOTE: the proxy can take 4+ minutes to
# warm up a model on first use, hence the generous timeout.
if LITELLM_API_KEY:
    client_parcc = OpenAI(
        api_key=LITELLM_API_KEY,
        base_url=LITELLM_BASE_URL,
        timeout=600,
        max_retries=0,
    )
else:
    client_parcc = None


# Minimum completion budget for PARCC reasoning models. Reasoning tokens count against
# max_tokens, so a small cap (e.g. 2000) can be fully consumed by thinking, leaving the
# answer empty. Floor every litellm call at this value (matches the analogous-reasoning
# reference default). If a call is still truncated (finish_reason='length') with empty
# content, we escalate the budget up to LITELLM_MAX_MAX_TOKENS.
LITELLM_MIN_MAX_TOKENS = 16384
LITELLM_MAX_MAX_TOKENS = 32768
VLLM_MAX_MAX_TOKENS = int(os.environ.get("VLLM_MAX_MAX_TOKENS", "32768"))
# Reasoning models (gpt-oss) sometimes spend the entire completion budget in the
# reasoning channel and emit no final answer. Greedy decoding (temperature 0)
# makes that deterministic, so a plain re-run repeats it. Retry an empty response
# up to this many times with a nudged sampler to break the runaway trajectory.
VLLM_EMPTY_RETRIES = int(os.environ.get("VLLM_EMPTY_RETRIES", "3"))
VALID_LLM_PROVIDERS = {"auto", "openai", "litellm", "vllm"}


def _create_litellm_completion(client, api_params):
    """Call a litellm/vLLM chat completion, escalating max_tokens if a reasoning model
    exhausts its budget on thinking and returns empty content (finish_reason='length').
    Escalation stops once we get content, hit a non-truncation stop, or reach the cap."""
    params = dict(api_params)
    while True:
        completion = client.chat.completions.create(**params)
        choice = completion.choices[0]
        content = (choice.message.content or "").strip()
        if content or choice.finish_reason != "length":
            return completion
        current = params.get("max_tokens", LITELLM_MIN_MAX_TOKENS)
        if current >= LITELLM_MAX_MAX_TOKENS:
            return completion  # give up; caller handles empty content
        params["max_tokens"] = min(current * 2, LITELLM_MAX_MAX_TOKENS)


def _clamp_max_tokens_for_context(exc, params):
    """If ``exc`` is a vLLM context-length overflow, shrink ``params['max_tokens']``
    to the largest completion that still fits and return True; else return False.

    vLLM reports e.g. "This model's maximum context length is 131072 tokens.
    However, you requested 140000 tokens (24000 in the messages, 116000 in the
    completion). ...". We keep the full prompt and hand the completion whatever
    room remains (minus a small safety margin). This lets callers request a very
    large ``max_tokens`` (the max window) and have it auto-fit any prompt size."""
    message = str(getattr(exc, "message", "") or exc)
    if "maximum context length" not in message and "reduce the length" not in message:
        return False
    ctx = re.search(r"maximum context length is (\d+)", message)
    prompt = re.search(r"(\d+) in the (?:messages|input|prompt)", message)
    if not ctx or not prompt:
        return False
    budget = max(int(ctx.group(1)) - int(prompt.group(1)) - 256, 1)
    current = params.get("max_tokens")
    if current is not None and budget >= current:
        return False  # would not actually shrink; avoid an infinite retry loop
    params["max_tokens"] = budget
    return True


def _create_vllm_completion(client, api_params, require_complete=False):
    """Call local vLLM, growing an exhausted completion budget on truncation.

    Unlike the PARCC proxy path, the first request honors the caller's requested
    ``max_tokens``. This avoids reserving a 16k completion for every local request.

    By default a length-truncated but non-empty completion is returned as-is (fine
    for free-form prose). When ``require_complete`` is set — e.g. the caller must
    ``json.loads`` the result — a length finish also triggers a grow-and-retry, so
    reasoning models that spend the budget thinking and then emit a half-written
    JSON object don't slip a truncated, unparseable payload through.

    An over-large ``max_tokens`` (prompt + completion > context window) is caught
    and clamped to fit, so callers can request the full context window and let the
    remaining room be used for the completion.
    """
    params = dict(api_params)
    empty_retries = 0
    while True:
        try:
            completion = client.chat.completions.create(**params)
        except Exception as exc:
            if _clamp_max_tokens_for_context(exc, params):
                continue
            raise
        choice = completion.choices[0]
        content = (choice.message.content or "").strip()
        truncated = choice.finish_reason == "length"
        if not content and empty_retries < VLLM_EMPTY_RETRIES:
            # No final answer: the reasoning channel consumed the turn (finish_reason
            # "length" when it exhausted the budget, "stop" when it gave up). Greedy
            # decoding repeats this every time, so nudge temperature/top_p to break the
            # trajectory rather than returning an empty outline that skips the view.
            empty_retries += 1
            params["temperature"] = 0.4 + 0.3 * empty_retries
            params["top_p"] = 0.95
            continue
        if not truncated or (content and not require_complete):
            return completion
        current = params.get("max_tokens", 1)
        if current >= VLLM_MAX_MAX_TOKENS:
            return completion
        params["max_tokens"] = min(current * 2, VLLM_MAX_MAX_TOKENS)


def is_litellm_model(model: str) -> bool:
    """PARCC/LiteLLM models are namespaced with a provider prefix, e.g. 'zai-org/GLM-5.2-FP8'."""
    return isinstance(model, str) and "/" in model


def resolve_llm_provider(model: str, provider: str | None = "auto") -> str:
    """Resolve an explicit provider while preserving the historical auto route."""
    provider = (provider or "auto").lower()
    if provider not in VALID_LLM_PROVIDERS:
        choices = ", ".join(sorted(VALID_LLM_PROVIDERS))
        raise ValueError(f"Unknown LLM provider '{provider}'. Expected one of: {choices}")
    if provider == "auto":
        return "litellm" if is_litellm_model(model) else "openai"
    return provider


@lru_cache(maxsize=8)
def _get_vllm_client(base_url: str, api_key: str) -> OpenAI:
    """Return a cached, thread-safe OpenAI client for a local vLLM endpoint."""
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=600,
        max_retries=0,
    )


def get_vllm_client(base_url: str | None = None) -> OpenAI:
    """Build a client for an OpenAI-compatible vLLM server.

    The base URL must include the API prefix, for example
    ``http://127.0.0.1:8000/v1``.
    """
    resolved_url = base_url or os.environ.get("VLLM_BASE_URL")
    if not resolved_url:
        raise ValueError(
            "vLLM provider requested but no base URL was supplied. Pass base_url "
            "or set VLLM_BASE_URL (for example http://127.0.0.1:8000/v1)."
        )
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    return _get_vllm_client(resolved_url.rstrip("/"), api_key)


def _is_transient_server_error(exc) -> bool:
    """True for retryable upstream failures: timeouts, connection drops, and 5xx
    (esp. 502/503/504 while a PARCC model cold-starts). The OpenAI SDK maps a 504 to
    InternalServerError, so also inspect any attached status code."""
    import openai
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (500, 502, 503, 504)


# Short, human-readable folder names for known generator models. Falls back to a
# deterministic slugify (see model_slug) for anything not listed here.
MODEL_SLUGS = {
    "zai-org/GLM-5.2-FP8": "glm",
    "nvidia/GLM-5.2-NVFP4": "glm",
    "nvidia/GLM-5.1-NVFP4": "glm_5_1",
    "deepseek-ai/DeepSeek-V4-Flash": "deepseek",
    "Qwen/Qwen3.6-35B-A3B": "qwen",
    "Qwen/Qwen3-VL-235B-A22B-Instruct": "qwen_vl",
    "unsloth/Kimi-K2.6": "kimi",
    "openai/gpt-oss-20b": "gpt_oss_20b",
    "openai/gpt-oss-120b": "gpt_oss_120b",
    "gpt-5": "gpt_5",
    "gpt-5-mini": "gpt_5_mini",
    "gpt-4.1": "gpt_4_1",
    "gpt-4.1-mini": "gpt_4_1_mini",
    "gpt-4o-mini": "gpt_4o_mini",
}


def model_slug(model: str, override: str | None = None) -> str:
    """Return a filesystem-safe folder name identifying the generator model.

    Priority: explicit override > MODEL_SLUGS alias > slugify(part after '/')."""
    if override:
        return override
    if model in MODEL_SLUGS:
        return MODEL_SLUGS[model]
    base = model.split("/")[-1] if isinstance(model, str) else str(model)
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", base).strip("_").lower()
    return slug or "unknown"


def synthetic_dir(source: str, kind: str, slug: str, domain: str, root: str = "../../data") -> str:
    """Single source of truth for model-namespaced synthetic-text paths.

    e.g. synthetic_dir('arxiv', 'explanations', 'glm', 'DPO')
         -> '../../data/arxiv/explanations/glm/DPO/'
    """
    return os.path.join(root, source, kind, slug, domain, "")


def explanations_dir(source: str, slug: str, domain: str, root: str = "../../data") -> str:
    return synthetic_dir(source, "explanations", slug, domain, root=root)


REASONING_EFFORT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "data-preparation", "multiview",
    "reasoning_effort.json",
)


def load_reasoning_efforts(pipeline: str, override: str | None = None,
                           path: str = REASONING_EFFORT_CONFIG) -> dict:
    """Return {view: {step: effort}} for a multiview pipeline ('arxiv'|'medical'|'legal').

    Reads the per-step 'custom' profile from reasoning_effort.json. If ``override``
    ('low'|'medium'|'high') is given, every step is forced to that level (uniform mode)."""
    with open(path) as f:
        profile = json.load(f)[pipeline]
    if override:
        return {view: {step: override for step in steps} for view, steps in profile.items()}
    return profile


def paraphrased_dir(source: str, slug: str, domain: str, root: str = "../../data") -> str:
    return synthetic_dir(source, "paraphrased", slug, domain, root=root)


def _sanitize_for_json_payload(value):
    """Recursively sanitize values before sending to the OpenAI JSON payload."""
    if isinstance(value, str):
        # Remove null bytes and unpaired surrogate code points that can break JSON parsing.
        return "".join(
            ch for ch in value
            if ch != "\x00" and not (0xD800 <= ord(ch) <= 0xDFFF)
        )
    if isinstance(value, list):
        return [_sanitize_for_json_payload(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_for_json_payload(v) for k, v in value.items()}
    return value


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence (```json ... ``` or ``` ... ```).

    Some OpenAI-compatible backends (e.g. GLM via vLLM) wrap JSON-mode output in a
    fenced block, which breaks json.loads. This normalizes to the raw payload."""
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if stripped.startswith("```"):
        # drop the opening fence line (``` or ```json) and the closing fence
        stripped = stripped[3:]
        newline = stripped.find("\n")
        if newline != -1 and stripped[:newline].strip().isalpha():
            stripped = stripped[newline + 1:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
        return stripped.strip()
    return stripped


def extract_dict_list(data, preferred_key=None):
    """Best-effort extraction of a list-of-dicts from LLM JSON output.

    json_object mode guarantees valid JSON but not the requested schema; models
    (especially high-reasoning gpt-oss) sometimes return the list at the top
    level or under a differently named key. Try, in order: an explicit preferred
    key, a top-level list, then the first value that is a list of dicts. Returns
    [] when nothing usable is found instead of raising KeyError."""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if preferred_key and isinstance(data.get(preferred_key), list):
            return [x for x in data[preferred_key] if isinstance(x, dict)]
        for value in data.values():
            if isinstance(value, list) and any(isinstance(x, dict) for x in value):
                return [x for x in value if isinstance(x, dict)]
    return []


def query_llm(prompt, max_tokens=1000, temperature=0, top_p=0, max_try_num=10,
              model="gpt-4o-mini", debug=False, return_json=False, json_schema=None,
              logprobs=False, system_prompt_included=True, is_hippa=False,
              reasoning_effort=None, provider="auto", base_url=None):
    if debug:
        if system_prompt_included:
            print(f"System prompt: {prompt['system']}")
            print(f"User prompt: {prompt['user']}")
        else:
            print(prompt)
        print(f"Model: {model}")
    if is_hippa and ('gpt' not in model and 'o3' not in model):
        raise ValueError("HIPPA compliance requires GPT models")
    curr_try_num = 0
    while curr_try_num < max_try_num:
        try:
            resolved_provider = resolve_llm_provider(model, provider)
            if resolved_provider in {"openai", "litellm", "vllm"}:
                response = query_gpt(
                    prompt, model=model, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p,
                    return_json=return_json, json_schema=json_schema,
                    logprobs=logprobs,
                    system_prompt_included=system_prompt_included,
                    is_hippa=is_hippa, debug=debug,
                    reasoning_effort=reasoning_effort,
                    provider=resolved_provider, base_url=base_url,
                )
                if logprobs:
                    return response.choices[0].message.content.strip(), response.choices[0].logprobs
            return response
        except Exception as e:
            import openai, time, re as _re
            if isinstance(e, openai.RateLimitError):
                curr_try_num += 1
                # Try to parse wait time from error message (e.g. "Please try again in 573ms")
                wait_time = 2
                msg = str(e)
                ms_match = _re.search(r'try again in (\d+)ms', msg)
                s_match = _re.search(r'try again in ([\d.]+)s', msg)
                if ms_match:
                    wait_time = max(int(ms_match.group(1)) / 1000 + 0.5, 1)
                elif s_match:
                    wait_time = max(float(s_match.group(1)) + 0.5, 1)
                time.sleep(wait_time)
            elif _is_transient_server_error(e):
                # Transient upstream errors — most often a PARCC model cold-starting,
                # where nginx returns 504 until the model is loaded (can take minutes).
                # Back off substantially and keep retrying up to max_try_num.
                curr_try_num += 1
                if curr_try_num >= max_try_num:
                    raise e
                wait_time = min(60, 15 * curr_try_num)
                if debug:
                    print(f"Transient error ({type(e).__name__}); retry {curr_try_num}/{max_try_num} after {wait_time}s")
                time.sleep(wait_time)
            else:
                raise e
    return None

def query_gpt(prompt: str | dict, model: str = 'gpt-4.1-mini', max_tokens: int = 4000,
              temperature: float = 0, top_p: float = 0, logprobs: bool = False,
              return_json: bool = False, json_schema=None,
              system_prompt_included: bool = False, reasoning_effort='high',
              is_hippa: bool = False, debug: bool = False,
              provider: str = "auto", base_url: str | None = None):
    """OpenAI API wrapper; For HIPPA compliance, use client_safe e.g. model='openai-gpt-4o-high-quota-chat'"""
    resolved_provider = resolve_llm_provider(model, provider)
    if is_hippa:
        if client_safe is None:
            raise ValueError("HIPPA compliance requires DATABRICKS_TOKEN to be set, but it was not found in utils.keys or environment variables.")
        temp_client = client_safe
    elif resolved_provider == "litellm":
        if client_parcc is None:
            raise ValueError("LiteLLM model requested but LITELLM_API_KEY is not set (utils.keys or env).")
        temp_client = client_parcc
    elif resolved_provider == "vllm":
        temp_client = get_vllm_client(base_url)
    else:
        if client is None:
            raise ValueError("OpenAI provider requested but OPENAI_API_KEY is not set.")
        temp_client = client

    if system_prompt_included:
        # Format chat prompt with system and user messages
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    messages = _sanitize_for_json_payload(messages)
    if resolved_provider in {"litellm", "vllm"}:
        # PARCC/LiteLLM OpenAI-compatible (vLLM) backends: send standard sampling params
        # but omit 'seed' (some backends reject it). vLLM also requires top_p in (0, 1];
        # only forward top_p/temperature when positive, otherwise let the backend default.
        # These are reasoning models: reasoning tokens count against max_tokens and can
        # exhaust a small budget, leaving content empty. Enforce a floor so the final
        # answer has room after the model finishes thinking.
        api_params = {
            "model": model,
            "messages": messages,
            "max_tokens": (
                max(max_tokens, LITELLM_MIN_MAX_TOKENS)
                if resolved_provider == "litellm"
                else max_tokens
            ),
        }
        if temperature and temperature > 0:
            api_params["temperature"] = temperature
        if top_p and top_p > 0:
            api_params["top_p"] = top_p
        if reasoning_effort is not None:
            api_params["reasoning_effort"] = reasoning_effort
        # Qwen3/Qwen3.5 gate reasoning through the chat template's enable_thinking
        # flag, not the OpenAI reasoning_effort field (which they silently ignore),
        # so drop the no-op reasoning_effort and set thinking explicitly. Thinking on
        # pairs with `--reasoning-parser qwen3` so the <think> span is routed to
        # reasoning_content and guided JSON decoding applies to the clean answer only.
        # Thinking is controllable via QWEN_ENABLE_THINKING (default on): on dense
        # inputs (e.g. arxiv papers) reasoning can run away and consume the whole
        # budget without emitting an answer, so those runs disable it.
        if resolved_provider == "vllm" and "qwen" in model.lower():
            api_params.pop("reasoning_effort", None)
            think = os.environ.get("QWEN_ENABLE_THINKING", "1").strip().lower() not in {"0", "false", "no"}
            api_params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": think}}
        if resolved_provider == "vllm":
            # Greedy decoding (temperature 0) plus a large token budget can fall into
            # degenerate repetition loops on dense/technical inputs (e.g. runaway
            # "\boldsymbol{\boldsymbol{..." on math-heavy papers). A mild frequency
            # penalty discourages that with negligible impact on normal prose.
            # NOTE: a multiplicative repetition_penalty was tried and removed — at the
            # strength needed to break loops it also penalized structural JSON tokens,
            # degrading schema adherence (wrong/again-missing keys). The parse guards
            # below (empty / malformed / missing-key -> skip the view) handle the rare
            # residual loop instead.
            api_params["frequency_penalty"] = 0.3
            # Qwen recommends sampling over greedy (temperature 0) decoding — greedy is
            # prone to repetition loops on long generations (worst on the 27B). Allow a
            # run to switch to sampling via env without touching every call site.
            _t = os.environ.get("QWEN_TEMPERATURE")
            _p = os.environ.get("QWEN_TOP_P")
            if _t is not None:
                api_params["temperature"] = float(_t)
            if _p is not None:
                api_params["top_p"] = float(_p)
    elif 'gpt-5' in model:
        api_params = {
            "model": model,
            "messages": messages,
        }
        if reasoning_effort is not None:
            api_params["reasoning_effort"] = reasoning_effort
    elif 'o3' in model or 'o1' in model or 'o4' in model:
        api_params = {
            "model": model,
            "messages": messages,
        }
        if reasoning_effort is not None:
            api_params["reasoning_effort"] = reasoning_effort
    else:
        api_params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": 0
        }
    if logprobs:
        api_params["logprobs"] = logprobs
        api_params["top_logprobs"] = 3

    if return_json:
        if json_schema is None:
            api_params["response_format"] = {"type": "json_object"}
            if resolved_provider == "litellm":
                completion = _create_litellm_completion(temp_client, api_params)
            elif resolved_provider == "vllm":
                completion = _create_vllm_completion(temp_client, api_params, require_complete=True)
            else:
                completion = temp_client.chat.completions.create(**api_params)
            response = _strip_code_fences(completion.choices[0].message.content or "")
            # Self-hosted backends don't strictly enforce the json_object grammar (weak
            # models emit unescaped quotes; runaway repetition yields unterminated
            # strings). Blank a malformed payload so the caller's empty-response guard
            # skips this view instead of crashing on json.loads downstream.
            if resolved_provider in {"litellm", "vllm"} and response.strip():
                try:
                    json.loads(response)
                except json.JSONDecodeError:
                    response = ""
        else:
            api_params["response_format"] = json_schema
            if resolved_provider in {"litellm", "vllm"}:
                # OpenAI-compatible self-hosted backends generally return the
                # JSON text but do not implement the SDK beta parser helper.
                completion = (
                    _create_litellm_completion(temp_client, api_params)
                    if resolved_provider == "litellm"
                    else _create_vllm_completion(temp_client, api_params, require_complete=True)
                )
                raw = _strip_code_fences(completion.choices[0].message.content or "")
                try:
                    response = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError("vLLM returned invalid JSON for the requested schema") from exc
            else:
                completion = temp_client.beta.chat.completions.parse(**api_params)
                response = completion.choices[0].message.parsed
    else:
        if resolved_provider == "litellm":
            completion = _create_litellm_completion(temp_client, api_params)
        elif resolved_provider == "vllm":
            # Prose too: with thinking on, reasoning can eat the budget and truncate the
            # answer. Grow-on-truncation so the saved content isn't cut off mid-sentence.
            completion = _create_vllm_completion(temp_client, api_params, require_complete=True)
        else:
            completion = temp_client.chat.completions.create(**api_params)
        # Reasoning models (e.g. GLM via vLLM) return content=None when the token budget
        # is exhausted by reasoning tokens; guard so .strip() doesn't crash callers.
        response = (completion.choices[0].message.content or "").strip()
    if debug:
        print(f"Response: {response}")
    if logprobs:
        return response, completion.choices[0].logprobs
    else:
        return response
