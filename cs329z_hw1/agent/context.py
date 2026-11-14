"""内部上下文记录；元数据不进入课程 LM 的消息接口。"""

from dataclasses import dataclass
from typing import Literal


MessageSource = Literal["system", "user", "agent", "tool_result", "error", "summary"]


@dataclass(frozen=True)
class ContextMessage:
    role: Literal["system", "user", "assistant"]
    content: str
    source: MessageSource
    # 一次 send() 对应一个请求；同一请求可以经历多轮模型调用。
    request_id: int
    # 助手回复及其工具结果或错误反馈共用同一 turn_id，裁剪时整体处理。
    # 编号在会话内递增；系统消息为 0，用户请求单独占一个编号以免与工具轮次混删。
    turn_id: int

    def to_lm_message(self) -> dict[str, str]:
        """每次创建独立字典，只发送 LM 允许的两个字段。"""
        return {"role": self.role, "content": self.content}
