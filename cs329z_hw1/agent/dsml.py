"""兼容 DeepSeek DSML 文本调用；不执行工具，不修补截断或歧义输入。"""

import re

from cs329z_hw1.types import ParsedResponse, ToolCall
import json

# 兼容官方单竖线标记和实际回复中出现的双竖线标记。
MARKER = r"[｜|]{1,2}DSML[｜|]{1,2}"
START = re.compile(r"<" + MARKER + r"\s*(?:calls|tool_calls)\s*>")
TAG = re.compile(
    r"<(/?)" + MARKER + r"\s*(calls|tool_calls|invoke|parameter)\b([^>]*)>"
)
ATTR = re.compile(r'\s*([a-z_]+)\s*=\s*"([^"]*)"')


def _attributes(text):
    attributes = {}
    while text.strip():
        match = ATTR.match(text)
        if not match or match[1] in attributes:
            raise ValueError("Invalid or duplicate DSML attribute.")
        attributes[match[1]] = match[2]
        text = text[match.end() :]
    return attributes


def parse_dsml(text: str) -> ParsedResponse:
    """严格读取一个完整调用；字符串参数中含嵌套协议标记时拒绝歧义。"""
    start = START.search(text)
    if start is None:
        return ParsedResponse(text=text.strip(), error="Incomplete DSML call block.")
    position = start.end()

    def read_tag(kind, closing=False):
        nonlocal position
        while position < len(text) and text[position].isspace():
            position += 1
        match = TAG.match(text, position)
        if not match or match[2] != kind or bool(match[1]) != closing:
            raise ValueError(
                f"Expected DSML {'closing ' if closing else ''}{kind} tag."
            )
        position = match.end()
        attributes = _attributes(match[3])
        if closing and attributes:
            raise ValueError("Closing DSML tags cannot have attributes.")
        return attributes

    try:
        attributes = read_tag("invoke")
        if set(attributes) != {"name"} or not attributes["name"].strip():
            raise ValueError("DSML invoke requires a non-empty name.")
        name = attributes["name"]
        args = {}
        while True:
            probe = TAG.match(
                text, position + len(text[position:]) - len(text[position:].lstrip())
            )
            if probe and probe[1] and probe[2] == "invoke":
                break
            attributes = read_tag("parameter")
            if set(attributes) != {"name", "string"} or attributes["string"] not in {
                "true",
                "false",
            }:
                raise ValueError("DSML parameter requires name and string=true/false.")
            key = attributes["name"]
            if key in args:
                raise ValueError("Duplicate DSML parameter.")
            if attributes["string"] == "false":
                while position < len(text) and text[position].isspace():
                    position += 1
                value, position = json.JSONDecoder().raw_decode(text, position)
            else:
                # 字符串按原文保留；内嵌协议标记可能存在歧义，明确报错而非猜测边界。
                boundary = TAG.search(text, position)
                if boundary is None or not boundary[1] or boundary[2] != "parameter":
                    raise ValueError(
                        "Missing or ambiguous DSML string parameter boundary."
                    )
                value = text[position : boundary.start()]
                position = boundary.start()
            read_tag("parameter", closing=True)
            args[key] = value
        read_tag("invoke", closing=True)
        wrapper = "tool_calls" if "tool_calls" in start.group() else "calls"
        read_tag(wrapper, closing=True)
        outside = text[: start.start()] + text[position:]
        if TAG.search(outside) or "<tool_call>" in outside:
            raise ValueError("More than one or mixed-protocol tool call.")

        # 兼容日志中的封装形式：invoke 指定工具，参数却是整个 {name,args}。
        if (
            set(args) == {"name", "args"}
            and args["name"] == name
            and isinstance(args["args"], dict)
        ):
            args = args["args"]
        return ParsedResponse(
            text=outside.strip(), tool_call=ToolCall(name=name, args=args)
        )
    except (ValueError, RecursionError) as exc:
        return ParsedResponse(text=text.strip(), error=f"Invalid DSML call: {exc}")
