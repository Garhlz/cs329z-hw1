import json
from cs329z_hw1.types import ToolSpec, ToolResult, Status

from jsonschema import validate as validate_schema, ValidationError


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec]):
        self.tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self.tools.values()
        ]

    def validate(self, name: str, args: dict) -> ToolResult | None:
        # 如果验证成功的话返回正确
        if not self.tools.get(name):
            return ToolResult(
                status="unknown_tool",
                content=f"Unknown tool: {name}",
            )
        tool = self.tools[name]
        try:
            validate_schema(
                instance=args,
                schema=tool.parameters,
            )
        except ValidationError as e:
            return ToolResult(
                status="invalid_args",
                content=e.message,
            )

        return None

    def execute(self, name: str, args: dict) -> ToolResult:
        err = self.validate(name, args)
        if err is not None:
            return err

        tool = self.tools[name]

        try:
            result = tool.fn(**args)
            return ToolResult(
                status="ok",
                content=str(result),
            )
        except Exception as e:
            return ToolResult(
                status="tool_error",
                content=str(e),
            )
