from cs329z_hw1.types import ToolCall, ParsedResponse
import json

from cs329z_hw1.agent.dsml import START as DSML_START, TAG as DSML_TAG, parse_dsml

"""工具调用格式示例（说明文本后跟一个 JSON 调用块）：

I will calculate the sum.

<tool_call>
{"name": "add", "args": {"a": 17, "b": 25}}
</tool_call>
"""


def render_tools(schemas: list[dict]) -> str:
    """将注册表的工具说明与参数 Schema 写入模型提示词，并说明调用格式。"""
    return f"""
You can use the following tools:

{json.dumps(schemas, ensure_ascii=False, indent=2)}

To call a tool, output exactly one tool call:

<tool_call>
{{"name": "tool_name", "args": {{...}}}}
</tool_call>

Rules:
- Use only tools from the list above.
- Follow the parameter schema exactly.
- Output at most one tool call per response.
- Use valid JSON.
- Do not wrap the tool call in Markdown code fences.
- Use only the literal <tool_call> JSON format above, never DSML or provider-native markup.
- For a final answer, write plain text without a tool-call block.
- Reply in the user's requested language.
"""


def parse_response(text: str) -> ParsedResponse:
    """解析至多一个工具调用；错误通过 error 返回，不执行工具。"""
    opening = "<tool_call>"
    closing = "</tool_call>"
    start = text.find(opening)
    dsml = DSML_START.search(text)
    if dsml is not None and (start == -1 or dsml.start() < start):
        return parse_dsml(text)
    if start == -1:
        if DSML_TAG.search(text):
            return parse_dsml(text)
        return ParsedResponse(text=text.strip())

    # 所有位置都基于原始 text，避免切片和 lstrip 后下标发生偏移。
    json_start = start + len(opening)
    while json_start < len(text) and text[json_start].isspace():
        json_start += 1

    try:
        # 解码器识别完整 JSON 边界，参数字符串里的标签不会被当作调用。
        obj, json_end = json.JSONDecoder().raw_decode(text, json_start)
    except (ValueError, RecursionError):
        return ParsedResponse(
            text=text.strip(),
            error="Invalid or incomplete tool-call JSON.",
        )

    # 此处只校验协议结构；工具名称和参数 Schema 由注册表校验。
    if (
        not isinstance(obj, dict)
        or not isinstance(obj.get("name"), str)
        or not obj["name"].strip()
        or not isinstance(obj.get("args"), dict)
    ):
        return ParsedResponse(
            text=text.strip(),
            error="Tool-call JSON requires a non-empty name and an args object.",
        )

    closing_start = json_end
    while closing_start < len(text) and text[closing_start].isspace():
        closing_start += 1
    if not text.startswith(closing, closing_start):
        return ParsedResponse(
            text=text.strip(),
            error="Missing closing </tool_call> tag.",
        )

    call_end = closing_start + len(closing)
    suffix = text[call_end:]
    # 只检查 JSON 和结束标签之后的内容，不扫描参数内部的引用文本。
    # 第二个调用即使不完整，也不能执行第一个调用。
    if opening in suffix or DSML_TAG.search(text[:start] + suffix):
        return ParsedResponse(
            text=text.strip(),
            error="More than one tool call in a response.",
        )

    return ParsedResponse(
        text=(text[:start] + suffix).strip(),
        tool_call=ToolCall(name=obj["name"], args=obj["args"]),
    )


def format_tool_call(text: str, call: ToolCall) -> str:
    """按同一协议序列化调用；由 JSON 编码器保留参数类型并处理转义。"""
    payload = {
        "name": call.name,
        "args": call.args,
    }
    return (
        text.strip()
        + "\n<tool_call>\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n</tool_call>"
    )
