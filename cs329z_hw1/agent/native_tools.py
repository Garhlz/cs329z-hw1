"""原生工具 API 与课程文本循环之间的桥接，复用缓存、计费及审批逻辑。"""

import hashlib
import json

from cs329z_hw1.agent.tool_protocol import (
    SECURITY_INSTRUCTION,
    TASK_INSTRUCTION,
    capability_instructions,
    format_tool_call,
    parse_response,
)
from cs329z_hw1.llm import LM
from cs329z_hw1.types import ToolCall

NATIVE_INSTRUCTION = """Use the provided native function tools when needed.
Do not write XML, DSML, or textual tool-call blocks. For final answers use plain text.
Use English search keywords for English corpora; reply in the user's requested language.
""" + "\n" + SECURITY_INSTRUCTION + "\n" + TASK_INSTRUCTION


class NativeToolLM(LM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tools = []

    def bind_tools(self, schemas):
        self.tools = [{"type": "function", "function": schema} for schema in schemas]

    def _key(self, messages):
        # 工具定义和传输模式参与缓存键，不能复用旧文本协议的回复。
        payload = json.dumps(
            [super()._key(messages), self.tools, "native-v1"], sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _request_options(self, messages):
        native = []
        index = 0
        while index < len(messages):
            message = messages[index]
            # 第一条系统消息原本是文本协议工具说明，原生路径改由 tools 声明。
            if index == 0 and message["role"] == "system":
                native.append({
                    "role": "system",
                    "content": NATIVE_INSTRUCTION + "\n" + capability_instructions(
                        {tool["function"]["name"] for tool in self.tools}
                    ),
                })
                index += 1
                continue
            if message["role"] == "assistant":
                parsed = parse_response(message["content"])
                next_message = (
                    messages[index + 1] if index + 1 < len(messages) else None
                )
                if parsed.tool_call and not parsed.error and next_message:
                    call = parsed.tool_call
                    prefix = f"Tool {call.name} ("
                    if next_message["role"] == "user" and next_message[
                        "content"
                    ].startswith(prefix):
                        # 为重放的每一对调用/结果生成稳定且唯一的关联 ID。
                        # 压缩按完整轮次保留或移除这两条消息。
                        key = message["content"]
                        call_id = (
                            "call_"
                            + hashlib.sha256((str(index) + key).encode()).hexdigest()[
                                :24
                            ]
                        )
                        native.append(
                            {
                                "role": "assistant",
                                "content": parsed.text or None,
                                "tool_calls": [
                                    {
                                        "id": call_id,
                                        "type": "function",
                                        "function": {
                                            "name": call.name,
                                            "arguments": json.dumps(
                                                call.args, ensure_ascii=False
                                            ),
                                        },
                                    }
                                ],
                            }
                        )
                        native.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": next_message["content"],
                            }
                        )
                        index += 2
                        continue
            native.append(dict(message))
            index += 1
        options = {"messages": native}
        if self.tools:
            options.update(
                tools=self.tools, tool_choice="auto", parallel_tool_calls=False
            )
        return options

    def _response_text(self, message):
        calls = getattr(message, "tool_calls", None) or []
        if not calls:
            return message.content or ""
        replies = []
        for call in calls:
            try:
                args = json.loads(call.function.arguments)
            except (ValueError, TypeError):
                # 留给原有解析器报告错误；绝不执行无法解码的参数。
                return '<tool_call>{"invalid_native_arguments": true}</tool_call>'
            if not isinstance(args, dict):
                return '<tool_call>{"invalid_native_arguments": true}</tool_call>'
            text = format_tool_call(
                (message.content or "") if not replies else "",
                ToolCall(call.function.name, args),
            )
            replies.append(text)
        # 即使服务端忽略禁止并行的设置，多调用仍由原有解析器拒绝，不静默丢弃。
        return "\n".join(replies)
