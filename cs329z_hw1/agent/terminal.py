"""终端工具适配：保留状态和错误，分别限制标准输出与标准错误的长度。"""

from cs329z_hw1 import sandbox


def _excerpt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    note = "\n[output truncated; showing head and tail]\n"
    remaining = max(0, limit - len(note))
    head = (remaining + 1) // 2
    tail = remaining // 2
    return text[:head] + note + (text[-tail:] if tail else "")


def execute_terminal(cmd: str, *, workspace, timeout: float, max_chars: int) -> str:
    result = sandbox.run_terminal(cmd, workspace=workspace, timeout=timeout)
    # 先保留退出状态和原因，再将剩余额度分给两个输出流。
    header = (
        f"exit_code: {result.exit_code}\n"
        f"timed_out: {result.timed_out}\n"
        f"rejected: {result.rejected}\n"
        f"reason: {result.reason}\n"
        "stdout:\n"
    )
    available = max(0, max_chars - len(header) - len("\nstderr:\n"))
    # 一个流没有用完的额度可供另一个流使用。
    err_limit = min(len(result.stderr), available // 2)
    out_limit = min(len(result.stdout), available - err_limit)
    err_limit = available - out_limit
    content = header + _excerpt(result.stdout, out_limit) + "\nstderr:\n" + _excerpt(result.stderr, err_limit)
    if result.rejected or result.timed_out or result.exit_code != 0:
        # 注册表将异常文本原样转换为 tool_error，正常返回则转换为 ok。
        raise RuntimeError(content)
    return content
