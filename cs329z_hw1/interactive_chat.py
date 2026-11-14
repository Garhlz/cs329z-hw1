"""支持 /steer 中途补充消息的终端聊天入口。

    uv run python -m cs329z_hw1.interactive_chat [--mode confirm|auto] [--memory-dir DIR] [--out DIR]

This builds one session with ``cs329z_hw1.adapters.run_agent_session(LM(), None,
config)``, so it runs your full Cardinal toolset on the grading model.
You are the user: in confirm mode you are asked to approve
tool calls, and ``ask_user`` questions are put to you. Type ``exit`` or
press Ctrl-D to leave.

Every conversation is saved when it ends, as ``runs/chat-<timestamp>.json``
(or under --out): the full transcript of events, plus what you typed and what
the agent replied, and the model usage. Problem (field_notes) asks you to
submit these files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # so `cs329z_hw1.adapters` imports from any directory

from cs329z_hw1.llm import LM
from cs329z_hw1.types import AgentConfig
from cs329z_hw1.agent.console_user import QueuedConsoleUser


def show_progress(event):
    """默认只显示操作和状态，完整输出通过 --verbose-tools 开启。"""
    name = event["name"]
    kind = event["type"]
    if kind == "tool_call":
        arguments = json.dumps(event["args"], ensure_ascii=False)
        if len(arguments) > 180:
            arguments = arguments[:177] + "..."
        print(f"\n→ {name} {arguments}", flush=True)
    elif kind == "approval_pending":
        print(f"  {name} 等待批准", flush=True)
    elif kind == "tool_running":
        print(f"  {name} 执行中…", flush=True)
    elif kind == "tool_result":
        status = event["status"]
        label = {"ok": "完成", "denied": "已拒绝"}.get(status, "失败")
        print(
            f"  {'✓' if status == 'ok' else '✗'} {name} {label} "
            f"· {event['elapsed']:.2f}s（含审批等待）",
            flush=True,
        )
        if status != "ok":
            reason = " ".join(event["content"].split())
            print("  " + reason[:240], flush=True)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Chat with your Cardinal Agent.")
    parser.add_argument("--mode", choices=["confirm", "auto"], default="confirm")
    parser.add_argument(
        "--memory-dir", type=Path, default=Path(".lm_cache/chat_memory")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs"), help="where to save the conversation"
    )
    parser.add_argument(
        "--verbose-tools", action="store_true", help="show full tool results"
    )
    args = parser.parse_args()

    from cs329z_hw1 import adapters  # your wiring

    user = QueuedConsoleUser()
    config = AgentConfig(
        mode=args.mode,
        user=user,
        memory_dir=args.memory_dir,
        aux_lm=LM(tag="chat/aux"),
    )
    from cs329z_hw1.agent.native_tools import NativeToolLM

    lm = NativeToolLM(tag="chat/native")
    session = adapters.run_agent_session(lm, None, config)
    lm.bind_tools(session.registry.schemas())

    def progress(event):
        show_progress(event)
        if args.verbose_tools and event["type"] == "tool_result":
            print(event["content"], flush=True)

    session.on_progress = progress
    print(
        f"Cardinal Agent ({args.mode} mode, memory in {args.memory_dir}). Type exit to leave."
    )
    print(
        "While the agent is working, type /steer <message> to add instructions before its next model call."
    )
    started = time.strftime("%Y%m%d-%H%M%S")
    turns: list[dict] = []

    def save_conversation():
        if not turns:
            return
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"chat-{started}.json"
        record = {
            "started": started,
            "mode": args.mode,
            "model": lm.model,
            "turns": turns,
            "transcript": session.transcript,
            "usage": {"agent": lm.usage(), "aux": config.aux_lm.usage()},
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    while True:
        try:
            message = user.read_line("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in ("exit", "quit"):
            break
        if not message:
            continue
        try:
            result = session.send(message)
        except Exception:
            turns.append({"user": message, "agent": "", "status": "exception"})
            save_conversation()
            raise
        note = "" if result.status == "done" else f" [stopped: {result.status}]"
        print(f"\nagent{note}> {result.text}")
        turns.append({"user": message, "agent": result.text, "status": result.status})
        save_conversation()

    if turns:
        print(f"Saved this conversation to {save_conversation()}")


def main() -> None:
    from prompt_toolkit.patch_stdout import patch_stdout

    # 主线程 print 输出时暂时移开编辑行，再重绘提示符及已输入文字。
    with patch_stdout():
        _main()


if __name__ == "__main__":
    main()
