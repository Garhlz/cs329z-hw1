"""涉及题目：agent_loop、terminal、compaction、memory、interactive、
guardrails 和 sim_eval。

创建一个智能体会话。lm 只用于智能体自身的回复轮次。
其他模型调用（上下文压缩摘要、email_qa 和 daily_digest 工具内部调用）
必须使用 config.aux_lm。测试总会设置它；为 None 时回退使用 lm。

tools 指定允许调用的工具。None 表示使用 Cardinal 工具集：
search_emails、read_email、email_qa、daily_digest、search_docs、read_doc、
run_terminal、remember、recall、ask_user，定义见 cs329z_hw1/cardinal.py。
创建这种会话时不得读取邮件档案。
显式传入列表时，只包含列表指定的工具；config.user 已设置时额外加入 ask_user；
config.memory_dir 已设置时额外加入 remember 和 recall。

测试使用以下成员：

    session.send(user_message: str) -> AgentResult
    session.transcript -> list[dict]     # 只追加、不修改旧事件的日志
    session.set_mode(mode: str) -> None  # "confirm" 或 "auto"

一个会话保存一段对话，每次 send 延续该对话。
用相同 config.memory_dir 创建第二个会话时，会启动新对话，只共享持久化记忆。

下列规则按题目分组。token 计数使用 cs329z_hw1.tokens；
transcript 事件定义见 cs329z_hw1/types.py。

agent_loop：
* send 不因模型回复内容或工具行为而抛异常。模型调用抛 OutputTruncated 时，
    按格式错误回复处理：记录一个内容非空的 error 事件，然后继续运行。
    lm 的其他异常向外传播；工具函数内部的异常按普通 tool_error 处理。
    AgentResult.text 始终非空；AgentResult.transcript 是当时日志的副本。
* 发给 lm 的消息只能使用 system、user、assistant 角色。
    第一次模型调用的首条消息必须列出所有工具名称。
* 每次模型调用前，若本次 send 已调用模型 config.max_turns 次，
    以 max_turns 状态停止；否则，若本次 send 累计 token 数
    >= config.token_budget，以 token_budget 状态停止。
    累计量为本次 send 各模型调用的
    count_message_tokens(messages) + count_tokens(reply) 之和。
    最终答案所在调用即使超预算，也以 done 结束；
    最后一个允许轮次中的工具调用仍须执行。
* parse_response 设置 error，或注册表返回 unknown_tool、invalid_args 时，
    回复算格式错误；tool_error、denied 不算。
    连续第三次格式错误以 malformed 状态结束本次 send；
    合法回复会清零连续错误计数。结束后会话仍可使用。
* 无法解析的回复记录 assistant 和 error 事件。
    其他带工具调用的回复依次记录 assistant、tool_call、tool_result，
    其中 tool_result 保留 ToolResult 的 status。
    每个 tool_result 和 error 事件的内容必须原样出现在下一次模型调用的某条消息中。
    事件不得修改或删除。

terminal：
* run_terminal 工具调用 sandbox.run_terminal，并传入
    workspace=config.workspace、timeout=config.terminal_timeout。
* 工具结果包含 stdout、stderr 和退出码。
    仅命令实际运行且退出码为 0 时返回 ok；
    非零退出、超时、被拒绝均返回 tool_error。
    被拒绝命令的结果须包含沙箱的 reason；
    超时结果须包含 "timed out"、"timeout"、"time limit" 或 "killed"。
    所有情况下均继续运行。
* 命令输出 600,000 个字符后，每次模型调用仍须满足 config.context_budget；
    tool_result 事件保存模型实际看到的内容。

compaction：
* 每次模型调用均满足
    count_message_tokens(messages) <= config.context_budget，
    即使单条工具结果已超过预算。
* 当前用户消息必须出现在为它发起的每次模型调用中。
* 摘要由 config.aux_lm 生成，不能使用 lm。

memory：
* 所有持久化内容均存于 config.memory_dir 下。
    使用其他目录的会话不得向模型展示这些内容。
* 空存储上的 recall 返回 ok，且 content 非空。

interactive：
* confirm 模式下，每次执行 read_only=False 的工具前，
    恰好调用一次 config.user.approve(call)。
    只读工具是否审批由你决定；ask_user、remember、recall 从不请求审批。
    注册表已拒绝的调用不得送审。auto 模式下从不调用 approve。
* 未批准的工具不得执行；记录 tool_call 和状态为 denied 的 tool_result，
    内容非空且包含用户拒绝的理由，然后继续运行。
* 每次模型调用前恰好调用一次 config.user.pending_message()；
    若返回消息，将其追加为 user 事件，并纳入此次模型调用。
* ask_user 调用 config.user.ask(question)，将答案作为 ok 工具结果返回。

guardrails：
* 若任意 config.deny_rules 中的 rule 使 re.search(rule, cmd) 匹配
    run_terminal 的命令，则两种模式下都不得执行，也不调用 approve；
    tool_result 返回 denied，内容非空。
* 工具结果中的任何内容都不能修改模式、审批策略或拒绝规则。"""

from cs329z_hw1.agent.task_progress import TaskProgress, bounded_search_results

import copy
import logging
import time
from cs329z_hw1.agent.context import ContextMessage
from cs329z_hw1.agent.memory import FactMemory
import json
from cs329z_hw1.types import (
    AgentConfig,
    LMCallable,
    ToolSpec,
    ToolCall,
    ToolResult,
    AgentResult,
    Email,
)

from cs329z_hw1.adapters import (
    run_render_tools,
    run_tool_registry,
    run_parse_response,
    run_daily_digest,
    run_email_qa,
)

from cs329z_hw1.llm import OutputTruncated

from cs329z_hw1.tokens import (
    count_message_tokens,
    count_tokens,
)
from cs329z_hw1.agent.store import EmailStore, DocChunkStore

from cs329z_hw1.data import email_text

from cs329z_hw1.cardinal import cardinal_spec


class AgentSession:
    def __init__(
        self, lm: LMCallable, tools: list[ToolSpec] | None, config: AgentConfig
    ):
        # 界面可设置 on_progress；默认不产生终端输出，不参与模型上下文。
        self.on_progress = None
        self.lm = lm
        self.config = config
        self.memory = FactMemory(config.memory_dir)

        # 空列表表示不提供工具；默认 Cardinal 工具集由后续题目接入。
        if tools is None:
            self.tools = self.build_cardinal_tools(config)
        else:
            self.tools = list(tools)
            # 只追加配置启用的工具，不覆盖调用方显式提供的同名工具。
            names = {tool.name for tool in self.tools}
            optional = self.build_optional_tools(config)
            for tool in optional:
                if tool.name not in names:
                    self.tools.append(tool)
                    names.add(tool.name)

        self.registry = run_tool_registry(self.tools)
        # 模型上下文仅使用 system、user、assistant；工具结果通过 user 消息反馈。
        # request_id 按 send 递增，turn_id 按内部消息轮次递增，均不跨请求重置。
        self.task_progress = TaskProgress()
        self._request_id = 0
        self._turn_id = 0
        self.messages: list[ContextMessage] = [
            ContextMessage(
                role="system",
                content=run_render_tools(self.registry.schemas()),
                source="system",
                request_id=0,
                turn_id=0,
            )
        ]
        self.set_mode(config.mode)

        self.transcript = []  # 只追加的事件日志，与模型上下文分开保存

    def set_mode(self, mode: str) -> None:
        """更新会话模式；下一次工具调用按新的审批策略处理。"""
        if mode not in {"auto", "confirm"}:
            raise ValueError("mode must be 'auto' or 'confirm'")
        self.config.mode = mode

    def send(self, user_message: str) -> AgentResult:
        """延续当前对话；返回本次停止原因及完整日志的独立快照。"""
        # 简单“继续”延续调查清单；新问题启动新阶段，避免不同任务互相污染。
        if user_message.strip().lower().rstrip("?!？。") not in {
            "继续",
            "continue",
            "go on",
        }:
            self.task_progress = TaskProgress()
        else:
            self.task_progress.stage_reads = 0
        self._request_id += 1
        self._turn_id += 1
        self.messages.append(
            ContextMessage(
                role="user",
                content=user_message,
                source="user",
                request_id=self._request_id,
                turn_id=self._turn_id,
            )
        )

        self.transcript.append(
            {
                "type": "user",
                "content": user_message,
            }
        )

        # 预算和连续错误次数只针对本次 send，历史消息与日志继续保留。
        turns = 0
        total_tokens = 0
        malformed_count = 0

        # 调用前先检查轮次，再检查累计 token；最后允许轮次的工具仍须执行。
        while True:
            if turns >= self.config.max_turns:
                return AgentResult(
                    text=self.task_progress.stopped_message(),
                    status="max_turns",
                    transcript=copy.deepcopy(self.transcript),
                )

            if total_tokens >= self.config.token_budget:
                return AgentResult(
                    text="Token budget reached.",
                    status="token_budget",
                    transcript=copy.deepcopy(self.transcript),
                )

            # 每次主模型调用前恰好轮询一次；先注入再压缩，当前请求的新指令会被保留。
            if self.config.user is not None:
                pending = self.config.user.pending_message()
                if pending is not None:
                    self._turn_id += 1
                    self.messages.append(
                        ContextMessage(
                            role="user",
                            content=pending,
                            source="user",
                            request_id=self._request_id,
                            turn_id=self._turn_id,
                        )
                    )
                    self.transcript.append({"type": "user", "content": pending})

            # 每轮重建进度说明，旧版本不进入摘要；编号 -1 专用于此临时系统消息。
            self.messages = [m for m in self.messages if m.turn_id != -1]
            if self.task_progress.tool_count or turns == self.config.max_turns - 1:
                self.messages.insert(
                    1,
                    ContextMessage(
                        role="system",
                        content=self.task_progress.instruction(
                            self.config.max_turns - turns
                        ),
                        source="system",
                        request_id=0,
                        turn_id=-1,
                    ),
                )

            from cs329z_hw1.agent.compaction import compact_context

            self.messages = compact_context(
                self.messages,
                request_id=self._request_id,
                budget=self.config.context_budget,
                aux_lm=self.config.aux_lm
                if self.config.aux_lm is not None
                else self.lm,
            )
            cur_messages = [m.to_lm_message() for m in self.messages]
            # 本轮助手回复与后续工具结果或错误反馈使用同一个 turn_id。
            self._turn_id += 1
            turns += 1

            # 只恢复输出截断异常；其他模型异常向外传播，便于调用方定位故障。
            try:
                reply = self.lm(cur_messages)
            except OutputTruncated as exc:
                # 保留实际输出用于日志和计数，但不解析或执行被截断的调用。
                reply = exc.text
                parse_error = "Model output was truncated."
            else:
                parse_error = None

            # 按课程规则累计每次调用的完整输入和可见输出，而不是仅计算新增消息。
            total_tokens += count_message_tokens(cur_messages) + count_tokens(reply)

            if parse_error is None:
                # 将模型文本解析为最终答案或单个工具调用，不在此处执行工具。
                parsed = run_parse_response(reply)
                parse_error = parsed.error

            self.messages.append(
                ContextMessage(
                    role="assistant",
                    content=reply,
                    source="agent",
                    request_id=self._request_id,
                    turn_id=self._turn_id,
                )
            )
            self.transcript.append(
                {
                    "type": "assistant",
                    "content": reply,
                }
            )

            # 解析失败时把错误原文同时写入日志和下一轮上下文，让模型尝试修复。
            if parse_error is not None:
                malformed_count += 1

                error_text = f"Tool call error: {parse_error}"

                self.messages.append(
                    ContextMessage(
                        role="user",
                        content=error_text,
                        source="error",
                        request_id=self._request_id,
                        turn_id=self._turn_id,
                    )
                )

                self.transcript.append(
                    {
                        "type": "error",
                        "content": error_text,
                    }
                )

                # 连续三次格式错误才停止；日志仍保留，会话可继续接收下一次 send。
                if malformed_count >= 3:
                    return AgentResult(
                        text=error_text,
                        status="malformed",
                        transcript=copy.deepcopy(self.transcript),
                    )

                continue

            # 协议约定：无工具调用即为最终答案；空文本使用非空提示兜底。
            parsed_text = parsed.text.strip()
            if parsed.tool_call is None:
                return AgentResult(
                    text="No answer provided." if parsed_text == "" else parsed_text,
                    status="done",
                    transcript=copy.deepcopy(self.transcript),
                )

            call = parsed.tool_call

            self.transcript.append(
                {
                    "type": "tool_call",
                    "name": call.name,
                    "args": copy.deepcopy(call.args),
                }
            )

            # 先记录调用参数快照，再执行工具，防止工具修改参数影响旧日志。
            started = time.monotonic()
            self._notify_progress("tool_call", name=call.name, args=call.args)
            tool_result = self._execute_with_approval(call)
            self.task_progress.record(call, tool_result)
            self._notify_progress(
                "tool_result",
                name=call.name,
                status=tool_result.status,
                content=tool_result.content,
                elapsed=time.monotonic() - started,
            )

            # 巨大结果先限长，再写入日志；历史压缩不能代替单条结果的长度控制。
            limit = max(256, self.config.context_budget // 6 * 4)
            if len(tool_result.content) > limit:
                note = "\n[tool output truncated]\n"
                room = limit - len(note)
                tool_result.content = (
                    tool_result.content[: room // 2]
                    + note
                    + tool_result.content[-(room - room // 2) :]
                )

            self.transcript.append(
                {
                    "type": "tool_result",
                    "name": call.name,
                    "status": tool_result.status,
                    "content": tool_result.content,
                }
            )

            # 将结果状态和内容反馈给模型；使用 user 角色以符合 LM 接口。
            self.messages.append(
                ContextMessage(
                    role="user",
                    content=f"Tool {call.name} ({tool_result.status}): {tool_result.content}",
                    source="tool_result",
                    request_id=self._request_id,
                    turn_id=self._turn_id,
                )
            )

            # 未知工具和无效参数算格式错误；工具执行失败或被拒绝会重置连续错误次数。
            if tool_result.status in {
                "unknown_tool",
                "invalid_args",
            }:
                malformed_count += 1
            else:
                malformed_count = 0

            # 此分支使用本次工具结果生成停止说明，不依赖解析错误分支的变量。
            if malformed_count >= 3:
                return AgentResult(
                    text=tool_result.status + ": " + tool_result.content,
                    status="malformed",
                    transcript=copy.deepcopy(self.transcript),
                )

            # 工具结果已进入上下文，下一轮由模型决定继续调用还是回答。

    def _notify_progress(self, event_type, **details):
        """进度通知与执行逻辑解耦；界面异常不能改变工具执行结果。"""
        if self.on_progress is not None:
            try:
                self.on_progress(copy.deepcopy({"type": event_type, **details}))
            except Exception:
                logging.getLogger(__name__).exception("Progress callback failed.")

    def _execute_with_approval(self, call: ToolCall) -> ToolResult:
        """先校验再审批；拒绝调用也作为普通观察结果反馈给模型。"""
        invalid = self.registry.validate(call.name, call.args)
        if invalid is not None:
            return invalid

        tool = self.registry.tools[call.name]
        # 只读工具及三个内部交互/记忆工具免审批；auto 模式从不请求审批。
        needs_approval = (
            self.config.mode == "confirm"
            and not tool.read_only
            and call.name not in {"ask_user", "remember", "recall"}
        )
        if needs_approval:
            self._notify_progress("approval_pending", name=call.name)
            if self.config.user is None:
                # 没有审批接口时不能默认为批准，避免绕过 confirm 模式。
                return ToolResult(
                    status="denied",
                    content="Approval is required, but no user interface is configured.",
                )
            # 审批方获得独立参数副本，不能改写随后实际执行的调用。
            approval = self.config.user.approve(copy.deepcopy(call))
            if not approval.approved:
                return ToolResult(
                    status="denied",
                    content=approval.reason.strip() or "The user declined this action.",
                )

        self._notify_progress("tool_running", name=call.name)
        return self.registry.execute(call.name, call.args)

    def build_optional_tools(self, config: AgentConfig) -> list[ToolSpec]:
        """显式工具列表仅按配置追加交互及记忆工具。"""

        def ask_user(question: str) -> str:
            if config.user is None:
                raise RuntimeError("No user interface is configured.")
            return config.user.ask(question)

        def remember(fact: str) -> str:
            return self.memory.remember(fact)

        def recall(query: str) -> str:
            return self.memory.recall(query)

        tools = []
        if config.user is not None:
            tools.append(
                cardinal_spec(
                    "ask_user", "Ask the user a clarification question.", ask_user
                )
            )
        if config.memory_dir is not None:
            tools.extend(
                [
                    cardinal_spec(
                        "remember",
                        "Store a durable user fact or preference for future sessions. Preserve exact details.",
                        remember,
                    ),
                    cardinal_spec(
                        "recall",
                        "Search saved user facts and preferences by keywords. Recall relevant context before answering personalized questions.",
                        recall,
                    ),
                ]
            )
        return tools

    def build_cardinal_tools(self, config: AgentConfig) -> list[ToolSpec]:
        # 闭包共享同一个邮件存储；创建工具时不读取文件或建立索引。
        store = EmailStore()
        aux_lm = config.aux_lm if config.aux_lm is not None else self.lm
        # 两个文档工具捕获同一个存储对象；首次调用任一工具时才构建索引。
        doc_store = DocChunkStore()

        def search_emails(query: str, k: int = 5) -> str:
            # 搜索只返回定位信息和短片段；流水线仍使用 store.search 获取完整邮件。
            results = [
                {
                    "id": email["id"],
                    "subject": email["subject"],
                    "date": email["date"],
                    "from": email["from"],
                    "snippet": email["body"][:400],
                }
                for email in store.search(query, k)
            ]
            return bounded_search_results(
                results, max(256, config.context_budget // 6 * 4)
            )

        def read_email(email_id: str) -> str:
            store.ensure_data_loaded()
            return json.dumps(store.by_id[email_id], ensure_ascii=False)

        def email_qa(question: str) -> str:
            return json.dumps(
                run_email_qa(question, store.search, aux_lm), ensure_ascii=False
            )

        def daily_digest(date: str) -> str:
            store.ensure_data_loaded()
            emails = [email for email in store.emails if email["date"][:10] == date]
            return run_daily_digest(emails, aux_lm)

        def search_docs(query: str, k: int = 5) -> str:
            return bounded_search_results(
                doc_store.search(query, k), max(256, config.context_budget // 6 * 4)
            )

        def read_doc(doc_id: str, start_line: int = 1) -> str:
            return json.dumps(doc_store.read(doc_id, start_line), ensure_ascii=False)

        def run_terminal(cmd: str) -> str:
            from cs329z_hw1.agent.terminal import execute_terminal

            return execute_terminal(
                cmd,
                workspace=config.workspace,
                timeout=config.terminal_timeout,
                # 每次结果最多占上下文约八分之一，给历史、工具说明和错误预留空间。
                max_chars=max(512, config.context_budget // 8 * 4),
            )

        def remember(fact: str) -> str:
            return self.memory.remember(fact)

        def recall(query: str) -> str:
            return self.memory.recall(query)

        def ask_user(question: str) -> str:
            if config.user is None:
                raise RuntimeError("No user interface is configured.")
            return config.user.ask(question)

        return [
            cardinal_spec(
                "search_emails",
                "Search archived emails using English keywords (translate non-English queries first); returns IDs and snippets. Use read_email for full content.",
                search_emails,
            ),
            cardinal_spec(
                "read_email", "Read a complete archived email by ID.", read_email
            ),
            cardinal_spec(
                "email_qa",
                "Answer a factual question using archived email evidence.",
                email_qa,
            ),
            cardinal_spec(
                "daily_digest",
                "Summarize archived emails on a UTC date (YYYY-MM-DD).",
                daily_digest,
            ),
            cardinal_spec(
                "search_docs",
                "Search English company documents for policy or product information. Translate non-English queries to English. For staff and leadership, try org chart or leadership. No matches do not mean the corpus is empty.",
                search_docs,
            ),
            cardinal_spec(
                "read_doc",
                "Read a company document window by ID and starting line.",
                read_doc,
            ),
            cardinal_spec(
                "run_terminal",
                "Run a shell command in the workspace containing emails.jsonl and docs/. Returns exit status, stdout and stderr. Output may be truncated; prefer focused commands.",
                run_terminal,
            ),
            cardinal_spec(
                "remember",
                "Store a durable user fact or preference for future sessions. Preserve exact details.",
                remember,
            ),
            cardinal_spec(
                "recall",
                "Search saved user facts and preferences by keywords. Recall relevant context before answering personalized questions.",
                recall,
            ),
            cardinal_spec(
                "ask_user", "Ask the user a clarification question.", ask_user
            ),
        ]
