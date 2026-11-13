"""课程提供的终端沙箱。

run_terminal(cmd) 执行模型生成的一条 shell 命令，返回 TerminalResult。
它包含以下四层限制，分别说明其覆盖范围和局限。

1. 白名单检查（check_command）：在未被引号包围的 |、&、;、括号和换行处分割
   简单命令。每条命令必须以 ALLOWED_PROGRAMS 或 SHELL_BUILTINS 中的名称开头。
   拒绝：以路径指定程序（/bin/rm、./x.py）、变量赋值（X=1 cmd）、
   命令替换（$(...)、反引号）、$'...' 字符串和 here-document（<<）。
   被拒绝的命令不会启动。
   不检查参数：python3、awk 的 system()、find -exec、xargs 仍能启动其他程序
   或打开任意路径；> ~/notes.txt 这样的重定向也能通过检查。

2. 受限环境：命令在工作区中由 bash 执行，不继承当前进程的环境变量，
   因此不包含 API 密钥。PATH 只指向一个包含允许程序链接的目录。
   python3 是基础解释器，不包含项目依赖包。
   不限制：通过绝对路径启动程序。

3. 超时与输出上限：命令在独立进程组运行。超时或命令退出后，
   杀死组内所有残留进程。stdout、stderr 各最多保留 MAX_OUTPUT_BYTES 字节，
   未达上限的输出完整返回。
   不限制：脱离进程组的进程（os.setsid()），以及内存、CPU 和磁盘用量。

4. 操作系统隔离（可用时启用，jail() 返回实际使用的方式）。
   只有这一层也能约束 python3 和绝对路径访问。
   * macOS 的 sandbox-exec：禁止网络、禁止工作区外写入，
     禁止读取用户主目录和当前仓库（包括 .env），
     工作区和 Python 安装目录除外；机器上其他文件仍可读取。
   * Linux 的 bwrap（bubblewrap）：安装且允许使用时应用同样规则，
     并让命令看不到其他进程。
   * 否则为 "none"：仍启用前 3 层，但运行 python3 的命令可以联网，
     并读写当前用户有权限访问的任意文件。

工作区保存语料：docs/ 为副本；文件系统支持时，
emails.jsonl 使用写时复制克隆（macOS APFS；Linux btrfs 和 XFS）。
其他情况下使用指向真实邮件档案的符号链接，并把真实档案设为只读。

Windows（不使用 WSL）启用前 3 层，但有以下区别；没有第 4 层：
jail() 返回 "none"，所以 python3 命令可以联网并读写当前用户有权限访问的文件。
* shell 使用 Git for Windows 的 bash.exe，允许的程序来自其 usr/bin。
  未安装 Git for Windows 时，所有命令均被拒绝，并返回安装提示。
* PATH 目录为每个允许程序保存一个小脚本，而非链接，
  因为普通 Windows 用户不能创建符号链接。
* 命令在作业对象（job object）中运行，超时时终止该对象。
* 输出中的 Windows 换行（\r\n）转换为 \n。
* emails.jsonl 使用真实档案的硬链接，并在当前进程退出前保持只读；
  无法创建硬链接时使用副本。
具体实现见 sandbox_windows.py。"""

from __future__ import annotations

import atexit
import functools
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from cs329z_hw1 import data, sandbox_windows

ALLOWED_PROGRAMS = (
    "python3", "python", "cat", "head", "tail", "grep", "wc", "sort", "uniq", "cut",
    "sed", "awk", "tr", "ls", "find", "jq", "diff", "comm", "paste", "join", "nl",
    "tee", "xargs", "basename", "dirname", "date", "seq", "mkdir",
)  # fmt: skip
SHELL_BUILTINS = ("cd", "echo", "printf", "pwd", "read", "test", "[", "true", "false")
MAX_OUTPUT_BYTES = 4_000_000  # 每个输出流分别限制

_KEYWORDS = {
    "if",
    "then",
    "elif",
    "else",
    "fi",
    "while",
    "until",
    "do",
    "done",
    "!",
    "{",
    "}",
}
_WINDOWS = sys.platform == "win32"
if (
    _WINDOWS
):  # 程序和 shell 来自 Git for Windows；未安装时 shell 为 None
    _SYSTEM_PATH = str(sandbox_windows.git_usr_bin() or "")
    _SHELL = os.path.join(_SYSTEM_PATH, "bash.exe") if _SYSTEM_PATH else None
else:
    _SYSTEM_PATH = "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"
    _SHELL = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"


@dataclass
class TerminalResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None  # 被拒绝或超时时为 None
    timed_out: bool = False
    rejected: bool = (
        False  # 白名单拒绝命令或 shell 不存在时为 True
    )
    reason: str = ""  # 被拒绝或终止的原因，其他情况为空字符串


def _simple_commands(cmd: str) -> list[str]:
    """在未被引号包围的 |、&、;、(、) 和换行处分割 cmd，并去掉 shell 注释。"""
    parts, cur, quote, i = [], [], "", 0
    while i < len(cmd):
        c = cmd[i]
        if quote == "'":  # 单引号内的字符不作特殊解释
            quote = "" if c == "'" else quote
        elif c == "\\":  # 保留反斜杠及其转义字符，不拆开处理
            c = cmd[i : i + 2]
        elif c == "`" or cmd.startswith(("$(", "$'"), i):
            raise ValueError(
                "command substitution ($(...) or backticks) and $'...' are not supported"
            )
        elif quote == '"':
            quote = "" if c == '"' else quote
        elif c in "'\"":
            quote = c
        elif cmd.startswith("<<", i):
            raise ValueError(
                "here-documents (<<) are not supported; use python3 -c '...' instead"
            )
        elif c == "#" and (
            i == 0 or cmd[i - 1] in " \t\n;|&()"
        ):  # 遇到注释，跳至行尾
            i = cmd.find("\n", i) if "\n" in cmd[i:] else len(cmd)
            continue
        elif c in "|;()\n" or (
            c == "&" and cmd[i - 1 : i] not in ("<", ">") and cmd[i + 1 : i + 2] != ">"
        ):
            parts.append(
                "".join(cur)
            )  # 2>&1 和 &> 中的 & 属于重定向，不是命令分隔符
            cur, c = [], " "
        cur.append(c)
        i += len(c)
    if quote:
        raise ValueError("unbalanced quote")
    return parts + ["".join(cur)]


def _program(simple_command: str) -> str | None:
    """返回 shell 将作为程序执行的单词；没有可执行程序时返回 None。"""
    words = shlex.split(simple_command)
    while words:
        word = words.pop(0)
        if word == "for":  # for x in a b c 后续单词是循环取值，不是命令
            return None
        if re.match(r"\d*[<>]|&>", word):  # 重定向
            if word[-1] in "<>&":  # 操作符与目标分属两个单词
                words = words[1:]
        elif word not in _KEYWORDS:
            return word
    return None


def check_command(cmd: str) -> str:
    """cmd 通过白名单检查时返回空字符串 ""，否则返回拒绝原因。"""
    allowed = ALLOWED_PROGRAMS + SHELL_BUILTINS
    try:
        programs = [_program(part) for part in _simple_commands(cmd)]
    except ValueError as e:
        return f"Command rejected: {e}."
    for program in programs:
        if program is not None and program not in allowed:
            return f"Command rejected: {program!r} is not an allowed program. Allowed: {', '.join(allowed)}."
    if not any(programs):
        return "Command rejected: it contains no program to run."
    return ""


@functools.cache
def _state_dir() -> Path:
    """为当前进程创建临时目录，并在进程退出时删除。"""
    path = Path(tempfile.mkdtemp(prefix="cs329z_hw1_sandbox_")).resolve()
    atexit.register(_remove_tree, path)
    return path


def _remove_tree(path: Path) -> None:
    if (
        _WINDOWS
    ):  # 只读文件（邮件档案的硬链接）不能直接删除，先恢复写权限
        for file in path.rglob("*"):
            file.chmod(0o666)
    shutil.rmtree(path, ignore_errors=True)


@functools.cache
def _bin_dir() -> Path:
    """创建用于 PATH 的目录：为本机找到的每个允许程序建立符号链接。
    Windows 下改用启动对应程序的小脚本。"""
    bin_dir = _state_dir() / "bin"
    bin_dir.mkdir()
    python = os.path.realpath(getattr(sys, "_base_executable", "") or sys.executable)
    for name in ALLOWED_PROGRAMS:
        target = (
            python
            if name.startswith("python")
            else shutil.which(name, path=_SYSTEM_PATH)
        )
        if target and _WINDOWS:
            sandbox_windows.write_shim(bin_dir / name, target)
        elif target:
            (bin_dir / name).symlink_to(target)
    return bin_dir


def make_workspace(path: str | Path | None = None) -> Path:
    """创建 path 目录（None 时创建新的临时目录）并返回其路径。

    如果目录中尚无语料，则放入 emails.jsonl 和 docs/。
    邮件档案尚未下载时，不创建 emails.jsonl。"""
    ws = (
        Path(path).resolve()
        if path is not None
        else Path(tempfile.mkdtemp(prefix="cs329z_hw1_ws_"))
    )
    ws.mkdir(parents=True, exist_ok=True)
    emails, docs = ws / "emails.jsonl", ws / "docs"
    if data.EMAILS_PATH.exists() and not os.path.lexists(emails):
        if (
            _WINDOWS
        ):  # 不使用克隆：优先硬链接到真实档案并设只读，失败时复制
            try:
                os.link(data.EMAILS_PATH, emails)
            except OSError:
                shutil.copyfile(data.EMAILS_PATH, emails)
            else:
                _read_only_until_exit(data.EMAILS_PATH)
        else:
            clone = (
                ["cp", "-c"] if sys.platform == "darwin" else ["cp", "--reflink=always"]
            )
            if (
                subprocess.run(
                    [*clone, data.EMAILS_PATH, emails], capture_output=True
                ).returncode
                != 0
            ):
                emails.unlink(
                    missing_ok=True
                )  # 克隆失败，改用指向真实只读档案的符号链接
                os.chmod(data.EMAILS_PATH, 0o444)
                emails.symlink_to(data.EMAILS_PATH.resolve())
    if data.DOCS_DIR.is_dir() and not docs.exists():
        shutil.copytree(data.DOCS_DIR, docs)
    return ws


@functools.cache
def _read_only_until_exit(path: Path) -> None:
    """将 path 设为只读，并在当前进程退出时恢复可写，
    以便 data/download.py 可以替换邮件档案。"""
    os.chmod(path, 0o444)
    atexit.register(lambda: path.exists() and os.chmod(path, 0o666))


def default_workspace() -> Path:
    """返回 run_terminal 未指定工作区时使用的默认目录：
    该目录为当前进程临时创建，并在进程退出时删除。"""
    return make_workspace(_state_dir() / "workspace")


def _jail_argv(workspace: Path) -> list[str]:
    """返回为 workspace 启动操作系统隔离的命令前缀；无隔离时返回 []。"""
    ws, home, repo, py, bin_dir, emails = (
        str(Path(p).resolve())
        for p in (
            workspace,
            Path.home(),
            data.REPO_ROOT,
            sys.base_prefix,
            _bin_dir(),
            data.EMAILS_PATH,
        )
    )
    if sys.platform == "darwin" and os.path.exists("/usr/bin/sandbox-exec"):
        q = json.dumps  # 为隔离配置中的路径添加引号和转义
        profile = (
            "(version 1)(allow default)(deny network*)"
            f'(deny file-write*)(allow file-write* (subpath {q(ws)}) (subpath "/dev/fd")'
            '(literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr"))'
            f"(deny file-read* (subpath {q(home)}) (subpath {q(repo)}))"
            f"(allow file-read* (subpath {q(ws)}) (subpath {q(py)}) (literal {q(emails)}))"
        )
        return ["/usr/bin/sandbox-exec", "-p", profile]
    if sys.platform == "linux" and shutil.which("bwrap"):
        return [
            shutil.which("bwrap"), "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
            "--tmpfs", home, "--tmpfs", repo, "--ro-bind", py, py, "--ro-bind", bin_dir, bin_dir,
            "--ro-bind-try", emails, emails, "--bind", ws, ws,
            "--unshare-net", "--unshare-pid", "--die-with-parent",
        ]  # fmt: skip
    return []


@functools.cache
def jail() -> str:
    """返回实际使用的操作系统隔离方式：sandbox-exec、bwrap 或 none。

    用一条简单命令试运行一次；失败时不使用隔离，
    例如当前进程自身已处于沙箱中。Windows 没有此类隔离。"""
    if _WINDOWS:
        return "none"
    argv = _jail_argv(_state_dir())
    try:
        if (
            argv
            and subprocess.run(
                [*argv, _SHELL, "-c", "true"], capture_output=True, timeout=10
            ).returncode
            == 0
        ):
            return Path(argv[0]).name
    except (OSError, subprocess.SubprocessError):
        pass
    return "none"


def _drain(pipe, buf: bytearray) -> None:
    """持续读取 pipe 直到结束，在 buf 中最多保留 MAX_OUTPUT_BYTES 字节。"""
    while chunk := os.read(pipe.fileno(), 65536):
        buf += chunk[: MAX_OUTPUT_BYTES - len(buf)]


def _text(buf: bytearray) -> str:
    note = (
        f"\n[sandbox: output cut off after {MAX_OUTPUT_BYTES} bytes]"
        if len(buf) >= MAX_OUTPUT_BYTES
        else ""
    )
    text = bytes(buf).decode("utf-8", errors="replace")
    return (text.replace("\r\n", "\n") if _WINDOWS else text) + note


def run_terminal(
    cmd: str, *, workspace: str | Path | None = None, timeout: float = 10.0
) -> TerminalResult:
    """在沙箱中执行 cmd，返回执行结果。

    workspace 指定工作目录，会先经过 make_workspace 处理，
    因此新建或空目录会自动放入语料。None 表示使用 default_workspace()。
    超时和白名单拒绝通过返回结果报告，不抛出对应异常。"""
    reason = check_command(cmd)
    if reason:
        return TerminalResult(rejected=True, reason=reason)
    if _SHELL is None:  # Windows 未安装 Git for Windows
        return TerminalResult(rejected=True, reason=sandbox_windows.INSTALL_HINT)
    ws = make_workspace(workspace) if workspace is not None else default_workspace()
    env = {
        "PATH": str(_bin_dir()), "HOME": str(ws), "TMPDIR": str(ws),
        "LC_ALL": "C", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
        **(sandbox_windows.child_env(ws) if _WINDOWS else {}),
    }  # fmt: skip
    argv = (_jail_argv(ws) if jail() != "none" else []) + [_SHELL, "-c", cmd]
    job = (
        sandbox_windows.Job() if _WINDOWS else None
    )  # Windows 使用作业对象代替进程组
    proc = subprocess.Popen(
        argv, cwd=ws, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=not _WINDOWS,  # 新会话使用独立进程组
        creationflags=subprocess.CREATE_NO_WINDOW if _WINDOWS else 0,
    )  # fmt: skip
    if job:
        job.add(proc)
    out, err = bytearray(), bytearray()
    readers = [
        threading.Thread(target=_drain, args=(pipe, buf), daemon=True)
        for pipe, buf in ((proc.stdout, out), (proc.stderr, err))
    ]
    for reader in readers:
        reader.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    if (
        job
    ):  # 终止组内所有残留进程，包括命令本身及其留下的后台任务
        job.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    proc.wait()
    for reader in readers:
        reader.join(timeout=1.0)
    return TerminalResult(
        stdout=_text(out),
        stderr=_text(err),
        exit_code=None if timed_out else proc.returncode,
        timed_out=timed_out,
        reason=f"Timed out after {timeout:g} seconds; the command was killed."
        if timed_out
        else "",
    )
