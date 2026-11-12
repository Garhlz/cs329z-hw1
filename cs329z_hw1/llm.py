"""课程提供的语言模型封装。

本模块只提供一个基本操作：接收消息列表的聊天补全调用，返回字符串。

    lm = LM()                                  # 作业使用的模型
    text = lm([{"role": "system", "content": "..."},
               {"role": "user", "content": "..."}])

接口不提供工具调用参数或结构化输出参数；这些功能需要你在此调用之上自行构建。

封装提供：
* 磁盘缓存：每次调用缓存到 .lm_cache/，按模型、采样设置和完整消息内容区分。
  代码及请求不变时重复运行不产生费用。
* 费用账本：每个实际发送到服务商的调用追加到 .lm_cache/ledger.jsonl，
  记录 token 用量和美元费用。运行 uv run python -m cs329z_hw1.llm 查看支出报告。
* 预算停止：设置 CS329Z_BUDGET_USD 后，当账本累计金额达到上限，
  抛出 BudgetExceeded，不再发起新请求。
* 截断处理：模型因达到 max_tokens 停止时，抛出 OutputTruncated，
  不直接返回被截断的文本。异常携带已生成文本（.text）和 token 用量。
  该调用仍计费并写入账本，但不会缓存。

ScriptedLM 是确定性测试使用的替代模型，重放固定回复列表，不访问网络。"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

Message = dict  # {"role": "system" | "user" | "assistant", "content": str}

# 各角色的默认模型，可用 CS329Z_<ROLE>_MODEL 环境变量覆盖。
# 例如 CS329Z_GRADING_MODEL 指定作业模型。
DEFAULT_MODELS = {
    "grading": "openai/gpt-6-luna",  # 作业使用的模型
    "user": "openai/gpt-6-luna",  # 模拟用户
    "judge": "openai/gpt-6-sol",  # 评估对话记录
}

# OpenRouter 通过兼容 OpenAI 的接口提供多个厂商的模型。
# 模型名称含厂商前缀，例如 openai/gpt-6-luna。
# 可用模型见 https://openrouter.ai/models。
OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_URL = "https://api.deepseek.com"

# 支持推理强度的模型使用此默认值。
# 可用 CS329Z_REASONING_EFFORT 覆盖。
DEFAULT_REASONING_EFFORT = "low"

# 每百万 token 的美元价格：(普通输入、缓存输入、输出)。
# 键使用不含厂商前缀的模型名。优先记录服务商返回的费用，否则按表估算。
# 创建 LM 时检查价格，防止付费调用后才发现无法计费。
# OpenRouter 价格来源：https://openrouter.ai/models，查询日期 2026-10-05。
PRICES = {
    # DeepSeek 官网高峰美元价格，查询日期 2026-11-12。
    # 按高峰价格保守估算，低峰实际扣费可能更低。
    "deepseek-flash": (0.30, 0.006, 1.20),
    "deepseek-v4-pro": (1.32, 0.044, 3.96),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-6-astra": (10.00, 1.00, 50.00),
    "gpt-5.5": (5.00, 0.50, 30.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6-sol": (2.00, 0.20, 10.00),
}

VALID_ROLES = ("system", "user", "assistant")


class BudgetExceeded(RuntimeError):
    """累计支出达到预算时抛出，阻止新的付费请求。"""


class ScriptExhausted(RuntimeError):
    """模拟模型的调用次数超过预设回复数量时抛出。"""


class OutputTruncated(RuntimeError):
    """模型因达到 max_tokens 而停止时抛出。

    text 是截断前已生成的文本，通常可能为空：
    推理模型的隐藏推理也占用同一输出额度，可能在生成任何可见文本前就用完额度。
    output_tokens 是全部输出 token 数；reasoning_tokens 是其中隐藏推理的数量，
    服务商未报告时为 0。增大 max_tokens 后再次调用，属于新的付费请求。"""

    def __init__(
        self,
        text: str,
        *,
        model: str,
        max_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
    ):
        self.text = text
        self.model = model
        self.max_tokens = max_tokens
        self.output_tokens = output_tokens
        self.reasoning_tokens = reasoning_tokens
        shown = f"{len(text)} characters of text" if text else "no visible text"
        super().__init__(
            f"{model} stopped at max_tokens={max_tokens} with {shown} "
            f"({output_tokens} output tokens, {reasoning_tokens} of them hidden reasoning). "
            "Ask for a shorter reply, or call again with a larger max_tokens."
        )


def resolve_model(role_or_model: str | None = None) -> str:
    """将 grading、user、judge 角色解析为配置的模型名称。

None 使用 grading；其他字符串视为模型名称，原样返回。"""
    name = role_or_model or "grading"
    if name in DEFAULT_MODELS:
        default = (
            "deepseek-flash"
            if os.environ.get("CS329Z_PROVIDER", "openrouter").lower() == "deepseek"
            else DEFAULT_MODELS[name]
        )
        return os.environ.get(f"CS329Z_{name.upper()}_MODEL", default)
    return name


def _price(model: str) -> tuple[float, float, float]:
    model = model.rsplit("/", 1)[-1]  # "openai/gpt-6-luna" -> "gpt-6-luna"
    best = None
    for prefix, price in PRICES.items():
        if model == prefix or model.startswith(prefix + "-20"):
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, price)
    if best is None:
        raise KeyError(
            f"No price on file for model {model!r}. Add it to PRICES in cs329z_hw1/llm.py."
        )
    return best[1]


def cost_usd(
    model: str, input_tokens: int, cached_tokens: int, output_tokens: int
) -> float:
    p_in, p_cached, p_out = _price(model)
    fresh = max(input_tokens - cached_tokens, 0)
    return (fresh * p_in + cached_tokens * p_cached + output_tokens * p_out) / 1_000_000


def _cache_dir() -> Path:
    return Path(os.environ.get("CS329Z_CACHE_DIR", REPO_ROOT / ".lm_cache"))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(REPO_ROOT / ".env")


_ledger_lock = threading.Lock()
# 记录拒绝 temperature 或推理强度参数的模型。
# 首次调用报错后更新，后续调用省略不支持的参数。
_capabilities_lock = threading.Lock()
_NO_TEMPERATURE: set[str] = set()
_NO_EFFORT: set[str] = set()


def read_ledger() -> list[dict]:
    path = _cache_dir() / "ledger.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def total_spend() -> float:
    return sum(row.get("cost_usd", 0.0) for row in read_ledger())


def _append_ledger(row: dict) -> None:
    path = _cache_dir() / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ledger_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def _validate_messages(messages: Sequence[Message]) -> None:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError(
            "messages must be a non-empty list of {'role', 'content'} dicts"
        )
    for m in messages:
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            raise ValueError(
                f"each message must be a dict with exactly the keys 'role' and 'content', got {m!r}"
            )
        if m["role"] not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {m['role']!r}")
        if not isinstance(m["content"], str):
            raise ValueError("message content must be a string")


class LM:
    """聊天模型：输入消息列表，输出字符串。

    参数
    ----
    model：
        角色名称（grading、user、judge），或 OpenRouter 模型名
        （例如 openai/gpt-6-sol）；None 表示 grading。
    temperature、max_tokens：
        采样设置，同时参与缓存键。推理模型会将部分 max_tokens 用于隐藏推理，
        因此应预留足够额度。不支持 temperature 的模型会省略该参数调用。
    tag：
        写入费用账本的自定义标签，便于分组统计支出。
        测试套件会将它设为测试名称。
    cache：
        设为 False 时，此实例跳过磁盘缓存。
    salt：
        混入缓存键的额外字符串。不同 salt 的实例针对相同消息获得独立样本。"""

    def __init__(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tag: str | None = None,
        cache: bool = True,
        salt: str = "",
    ) -> None:
        _load_env()
        self.provider = os.environ.get("CS329Z_PROVIDER", "openrouter").lower()
        if self.provider not in {"openrouter", "deepseek"}:
            raise ValueError("CS329Z_PROVIDER must be openrouter or deepseek")
        self.base_url = DEEPSEEK_URL if self.provider == "deepseek" else OPENROUTER_URL
        self.model = resolve_model(model)
        _price(self.model)  # 先检查计价配置，避免付费后才报错
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tag = tag
        self.cache = cache
        self.salt = salt
        # 当前实例的累计调用统计。
        self.calls = 0
        self.cached_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.seconds = 0.0
        self._client = None
        # 仅保护共享状态；网络请求不持有这些锁。
        self._stats_lock = threading.Lock()
        self._client_lock = threading.Lock()

    # -- 对外接口 -------------------------------------------------------
    # 满足了LMCallable = Callable[[list[Message]], str]这个结构化类型
    def __call__(self, messages: Sequence[Message]) -> str:
        _validate_messages(messages)
        messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        key = self._key(messages)
        path = _cache_dir() / key[:2] / f"{key}.json"
        with self._stats_lock:
            self.calls += 1
        if self.cache and path.exists():
            with self._stats_lock:
                self.cached_calls += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]

        self._check_budget()
        start = time.time()

        # 在这里完成模型调用
        text, usage = self._complete(messages)

        elapsed = time.time() - start
        cost = usage.get("cost")
        if cost is None:
            cost = cost_usd(
                self.model, usage["input"], usage["cached"], usage["output"]
            )
        with self._stats_lock:
            self.input_tokens += usage["input"]
            self.output_tokens += usage["output"]
            self.cost_usd += cost
            self.seconds += elapsed
        truncated = usage.get("finish_reason") == "length"
        _append_ledger(
            {
                "ts": round(start, 3),
                "model": self.model,
                "provider": self.provider,
                "tag": self.tag or os.environ.get("CS329Z_TAG", ""),
                "input_tokens": usage["input"],
                "cached_tokens": usage["cached"],
                "output_tokens": usage["output"],
                "cost_usd": cost,
                "seconds": round(elapsed, 3),
                "finish_reason": usage.get("finish_reason"),
            }
        )
        if truncated:  # 截断请求仍计费并记账，但不缓存
            raise OutputTruncated(
                text,
                model=self.model,
                max_tokens=self.max_tokens,
                output_tokens=usage["output"],
                reasoning_tokens=usage.get("reasoning", 0),
            )
        if self.cache and text.strip():  # 空回复不缓存
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(
                json.dumps({"text": text, "model": self.model}), encoding="utf-8"
            )
            tmp.replace(path)
        return text

    def usage(self) -> dict:
        """返回当前实例自创建以来的累计统计快照。"""
        with self._stats_lock:
            return {
                "model": self.model,
                "calls": self.calls,
                "cached_calls": self.cached_calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": self.cost_usd,
                "seconds": self.seconds,
            }

    # -- 内部实现 --------------------------------------------------------

    def _key(self, messages: list[Message]) -> str:
        effort = os.environ.get("CS329Z_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
        blob = json.dumps(
            [
                self.base_url,
                self.model,
                self.temperature,
                self.max_tokens,
                effort,
                self.salt,
                messages,
            ],
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _check_budget(self) -> None:
        cap = os.environ.get("CS329Z_BUDGET_USD")
        if not cap:
            return
        spent = total_spend()
        if spent >= float(cap):
            raise BudgetExceeded(
                f"Spent ${spent:.2f} of the ${float(cap):.2f} cap set by CS329Z_BUDGET_USD. "
                "Raise the cap in .env if you mean to continue."
            )

    def _get_client(self):
        with self._client_lock:
            return self._initialize_client()

    def _initialize_client(self):
        if self._client is None:
            from openai import OpenAI

            key_name = (
                "DEEPSEEK_API_KEY"
                if self.provider == "deepseek"
                else "OPENROUTER_API_KEY"
            )
            key = os.environ.get(key_name)
            if not key:
                raise RuntimeError(
                    f"{key_name} is not set. Copy .env.example to .env and add your key."
                )
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=key,
                max_retries=6,
                timeout=120,
                default_headers={"X-Title": "CS329Z HW1"},
            )
        return self._client

    def _complete(self, messages: list[Message]) -> tuple[str, dict]:
        client = self._get_client()
        with _capabilities_lock:
            no_temperature = self.model in _NO_TEMPERATURE
            no_effort = self.model in _NO_EFFORT
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            # OpenRouter 扩展参数：请求回复携带本次调用费用。
            # 下方按配置补充推理强度。
            "extra_body": {"usage": {"include": True}},
        }
        if not no_temperature:
            kwargs["temperature"] = self.temperature
        effort = os.environ.get("CS329Z_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
        if self.provider == "deepseek":
            # 关闭思考模式，降低开销并避免额外的推理历史要求。
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if self.provider == "openrouter" and effort and not no_effort:
            kwargs["extra_body"]["reasoning"] = {"effort": effort}
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:  # 部分模型不支持温度或推理强度参数
            text = str(exc)
            if "temperature" in text and "temperature" in kwargs:
                with _capabilities_lock:
                    _NO_TEMPERATURE.add(self.model)
                kwargs.pop("temperature")
                response = client.chat.completions.create(**kwargs)
            elif "reasoning" in text and "reasoning" in kwargs["extra_body"]:
                with _capabilities_lock:
                    _NO_EFFORT.add(self.model)
                kwargs["extra_body"].pop("reasoning")
                response = client.chat.completions.create(**kwargs)
            else:
                raise
        choice = response.choices[0]
        text = choice.message.content or ""
        u = response.usage
        details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(u, "prompt_cache_hit_tokens", None)
        if cached is None:
            cached = getattr(details, "cached_tokens", 0) or 0
        out_details = getattr(u, "completion_tokens_details", None)
        reasoning = getattr(out_details, "reasoning_tokens", 0) or 0
        usage = {
            "input": u.prompt_tokens,
            "cached": cached,
            "output": u.completion_tokens,
            "reasoning": reasoning,
            "finish_reason": getattr(choice, "finish_reason", None),
        }
        cost = getattr(u, "cost", None)  # OpenRouter 返回的本次实际费用
        if isinstance(cost, (int, float)):
            usage["cost"] = float(cost)
        return text, usage


class ScriptedLM:
    """重放固定脚本的模拟模型，供确定性测试替代真实 LM 使用。

    脚本每项表示一次模型回复，支持三种形式：
    * 字符串：原样返回。
    * 字典 {"text": str, "tool_call": {"name": str, "args": dict}}：
      通过 formatter(text, ToolCall) 转为字符串。
      测试将你的 run_format_tool_call 适配器作为 formatter，
      因此预设工具调用会使用你定义的格式。
    * 可调用对象 f(messages) -> str。

    每次调用的消息记录在 .calls 中，测试可检查代码实际发给模型的完整内容。
    脚本耗尽时抛出 ScriptExhausted；若 repeat_last=True，则无限重放最后一项。"""

    model = "scripted"

    def __init__(
        self,
        script: Sequence[Any],
        *,
        formatter: Callable[[str, Any], str] | None = None,
        repeat_last: bool = False,
    ) -> None:
        self.script = list(script)
        self.formatter = formatter
        self.repeat_last = repeat_last
        self.calls: list[list[Message]] = []
        self.replies: list[str] = []

    def __call__(self, messages: Sequence[Message]) -> str:
        _validate_messages(messages)
        index = len(self.calls)
        self.calls.append(copy.deepcopy(list(messages)))
        if index >= len(self.script):
            if not (self.repeat_last and self.script):
                raise ScriptExhausted(
                    f"ScriptedLM was called {index + 1} times but the script has "
                    f"{len(self.script)} entries. Your code made a model call the test "
                    "did not expect."
                )
            index = len(self.script) - 1
        entry = self.script[index]
        if callable(entry):
            reply = entry(self.calls[-1])
        elif isinstance(entry, dict):
            reply = self._render(entry)
        else:
            reply = str(entry)
        self.replies.append(reply)
        return reply

    @property
    def remaining(self) -> int:
        return max(len(self.script) - len(self.calls), 0)

    def _render(self, entry: dict) -> str:
        text = entry.get("text", "")
        call = entry.get("tool_call")
        if call is None:
            return text
        if self.formatter is None:
            raise RuntimeError(
                "ScriptedLM needs a formatter to render scripted tool calls"
            )
        from cs329z_hw1.types import ToolCall

        return self.formatter(
            text, ToolCall(name=call["name"], args=dict(call.get("args", {})))
        )

    def usage(self) -> dict:
        return {"model": "scripted", "calls": len(self.calls), "cost_usd": 0.0}


def spending_report(rows: list[dict] | None = None) -> str:
    rows = read_ledger() if rows is None else rows
    if not rows:
        return "No LLM calls recorded yet."
    by_model: dict[str, dict] = {}
    by_tag: dict[str, dict] = {}
    for row in rows:
        model = row.get("model", "?")
        short = model.split("-20")[0]  # 去掉模型快照日期
        tag_key = f"{row.get('tag') or '(untagged)'} [{short}]"
        for table, key in ((by_model, model), (by_tag, tag_key)):
            agg = table.setdefault(
                key, {"calls": 0, "cut": 0, "in": 0, "out": 0, "usd": 0.0, "sec": 0.0}
            )
            agg["calls"] += 1
            agg["cut"] += row.get("finish_reason") == "length"
            agg["in"] += row.get("input_tokens", 0)
            agg["out"] += row.get("output_tokens", 0)
            agg["usd"] += row.get("cost_usd", 0.0)
            agg["sec"] += row.get("seconds", 0.0)
    lines = []
    for title, table in (("By model", by_model), ("By tag and model", by_tag)):
        lines.append(title)
        lines.append(
            f"  {'name':<72}{'calls':>7}{'cut':>5}{'in tok':>12}{'out tok':>10}{'USD':>9}{'sec':>8}"
        )
        for key, agg in sorted(table.items(), key=lambda kv: -kv[1]["usd"]):
            lines.append(
                f"  {key[-71:]:<72}{agg['calls']:>7}{agg['cut'] or '':>5}{agg['in']:>12}{agg['out']:>10}"
                f"{agg['usd']:>9.3f}{agg['sec']:>8.0f}"
            )
        lines.append("")
    total = sum(r.get("cost_usd", 0.0) for r in rows)
    cut = sum(r.get("finish_reason") == "length" for r in rows)
    lines.append(
        f"Total: {len(rows)} calls, ${total:.3f}"
        + (f"; {cut} cut off at max_tokens" if cut else "")
    )
    return "\n".join(lines)


if __name__ == "__main__":  # uv run python -m cs329z_hw1.llm
    print(spending_report())
