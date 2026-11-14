"""统一读取终端输入；/steer 消息走独立队列，不会被误当成审批答案。"""

import sys
import threading
from queue import Empty, Queue

from cs329z_hw1.types import Approval
from cs329z_hw1.user import UserIO


class QueuedConsoleUser(UserIO):
    def __init__(self, stream=None):
        self._stream = sys.stdin if stream is None else stream
        self._interactive = stream is None and sys.stdin.isatty()
        self._line_editing = self._interactive
        self._prompt_session = None
        if self._interactive:
            from prompt_toolkit import PromptSession

            # 由专用行编辑器处理中文宽度、退格、历史和提示符重绘。
            self._prompt_session = PromptSession()
        self._answers = Queue()
        self._steering = Queue()
        self._eof = False
        # 只有此线程读取 stdin；聊天主循环和审批函数都从队列取值。
        self._reader = threading.Thread(target=self._read_input, daemon=True)
        self._started = False

    def _input_lines(self):
        if not self._line_editing:
            yield from self._stream
            return
        while True:
            try:
                # 提示符由行编辑器维护，退格只编辑用户内容。
                # 使用统一提示符：具体聊天、审批或澄清要求由主线程另行显示。
                yield self._prompt_session.prompt("input> ", handle_sigint=False)
            except (EOFError, KeyboardInterrupt):
                return

    def _read_input(self):
        try:
            for line in self._input_lines():
                # 某些终端以 surrogateescape 保留无法解码的字节。
                # 先恢复这些字节，再按 UTF-8 解码；有效中文可恢复，损坏字节替换为 �。
                # 不能仅修改缓存的编码方式，否则请求序列化时仍会报错。
                try:
                    line = line.encode("utf-8", errors="surrogateescape").decode(
                        "utf-8", errors="replace"
                    )
                except UnicodeEncodeError:
                    line = line.encode("utf-8", errors="replace").decode("utf-8")
                text = line.rstrip("\r\n")
                if text == "/steer" or text.startswith("/steer "):
                    message = text[len("/steer") :].strip()
                    if message:
                        self._steering.put(message)
                else:
                    self._answers.put(text)
        finally:
            # EOF 唤醒等待用户输入的主线程，避免永久阻塞。
            self._answers.put(None)

    def read_line(self, prompt=""):
        """聊天、审批和澄清统一从此入口读取，不再直接调用 input。"""
        if self._line_editing:
            if prompt.strip():
                print("\n" + prompt.strip(), flush=True)
        else:
            print(prompt, end="", flush=True)
        if self._eof:
            raise EOFError
        # 首次真正请求输入时再启动，避免启动横幅与第一个提示符争抢屏幕。
        if not self._started:
            self._reader.start()
            self._started = True
        answer = self._answers.get()
        if answer is None:  # None 作为关闭阻塞get的哨兵值
            self._eof = True
            raise EOFError
        return answer

    def pending_message(self):
        """非阻塞，每轮最多取一条；没有补充消息立即返回 None。"""
        try:
            return self._steering.get_nowait()
        except Empty:
            return None

    def approve(self, call):
        print(f"\nThe agent wants to run {call.name} with {call.args}")
        try:
            reply = self.read_line("Allow? [y/N, or a reason] ").strip()
        except EOFError:
            return Approval(False, "Input closed before approval.")
        if reply.lower() in {"y", "yes"}:
            return Approval(True)
        return Approval(False, reply or "The user declined this action.")

    def ask(self, question):
        print(f"\nThe agent asks: {question}")
        return self.read_line("> ").strip()
