"""连接你的实现与课程测试的唯一文件。你需要编辑此文件。

下列函数初始均为占位实现。完成某题时，在 cs329z_hw1 包内实现功能，
再将占位函数改为调用你的实现。必须保持函数签名不变：
测试只调用这些入口，不直接调用你的其他代码。

评分时会把整个 cs329z_hw1/ 包复制到干净的初始仓库，其他内容均被丢弃。
辅助代码必须放在该包内；tests/ 或仓库根目录中的文件不会参与评分。

这里使用的 Email、ToolSpec、ToolCall、ToolResult、ParsedResponse、
AgentConfig、AgentResult、SearchResult、DocWindow 均定义在
cs329z_hw1/types.py。lm 始终符合 LMCallable 接口：
接收由 {"role", "content"} 字典组成的消息列表，只返回一个字符串。
真实模型测试传入 LM，确定性测试传入 ScriptedLM。调用示例：

    reply = lm([{"role": "system", "content": "Answer in one word."},
                {"role": "user", "content": "What color is the sky?"}])  # -> "Blue."

消息角色只能为 system、user、assistant。接口不提供工具调用或结构化输出功能；
第 2 部分需要你在这一文本调用之上，自行实现工具调用协议。"""

from __future__ import annotations

from typing import Any, Callable, Optional

from cs329z_hw1.types import (
    AgentConfig,
    DocWindow,
    Email,
    LMCallable,
    ParsedResponse,
    SearchResult,
    ToolCall,
    ToolSpec,
)

# ======================================================================
# 第 1 部分：LLM 流水线
# ======================================================================


def run_priority(email: Email, lm: LMCallable) -> dict:
    """题目 priority：邮件优先级分类。

    通过 lm 调用模型，从收件人角度判断一封邮件的优先级。
    返回 {"category": "urgent" | "normal" | "ignore" | "unknown", "reason": str}。
    unknown 表示模型回复没有给出标签。

    测试检查：
    * 无论模型如何回复，category 必须是上述四个小写值之一，reason 必须为非空字符串。
    * 回复中没有任何类别标签时（空串、空白或未提及三个标签的文本），
      返回 unknown，不能自行猜测。
    * 真实模型测试中不得出现 unknown。
    * 函数不得抛异常，也不得修改 email。
    * 真实模型测试要求与 tests/fixtures/priority_gold.json 的标签至少一致 85%。"""
    from cs329z_hw1.pipelines.priority import classify_priority

    return classify_priority(email, lm)


def run_daily_digest(emails: list[Email], lm: LMCallable) -> str:
    """题目 daily_digest：每日晨间摘要。

    emails 是同一个 UTC 日历日的邮件列表，可能为空。返回字符串形式的晨间摘要。

    测试检查：
    * 对任何模型回复以及空邮件列表，均返回字符串。
    * 不修改 emails。
    * 真实模型测试使用 tests/fixtures/digest_days.json：
      模型原样生成的摘要必须非空且不超过 300 词，以 len(digest.split()) 计数；
      每封标准 urgent 邮件涉及的公司、项目或地点名称都必须出现；
      不得出现标准 ignore 邮件列出的任何禁提词。"""
    from cs329z_hw1.pipelines.daily_digest import daily_digest

    return daily_digest(emails, lm)


def run_bm25_build(docs: list[str]) -> Any:
    """题目 bm25：构建并返回 docs 的 BM25 索引。

    文档编号采用列表下标。必须使用 cs329z_hw1.tokenizer.tokenize 分词。
    没有 token 的文档仍计入文档总数 N 和平均长度。
    测试不检查索引对象内部结构，只会把它传回 run_bm25_search。
    完整邮件档案的索引必须在 15 秒内建好。"""

    # 每个词对应在每篇doc中出现的频率

    from cs329z_hw1.pipelines.bm25 import build_index

    return build_index(docs)


def run_bm25_search(index: Any, query: str, k: int) -> list[tuple[int, float]]:
    """题目 bm25：返回 query 的前 k 个 (doc_id, score)。

    每项为 (int, float)，按分数从高到低排列；同分时 doc_id 较小者优先。
    doc_id 是文档在 run_bm25_build 的 docs 输入列表中的下标。

    测试检查：
    * 分数符合讲义公式，k1=1.5、b=0.75，使用 Lucene IDF，误差不超过 1e-6。
      查询与文档使用相同分词方式；重复查询词只计算一次。
    * 不返回与查询没有共同词的文档，因此结果可少于 k 项。
      查询没有 token，或查询词不在任何文档中出现时，返回 []。
    * 查询只读取查询词对应的索引项：档案上的平均查询耗时小于 20 毫秒；
      在 200,000 篇文档中仅出现于 3 篇的词，其查询耗时小于 2 毫秒。"""
    from cs329z_hw1.pipelines.bm25 import search_index

    return search_index(index, query, k)


def run_email_qa(
    question: str,
    search: Callable[[str, int], list[Email]],
    lm: LMCallable,
) -> dict:
    """题目 email_qa：基于邮件档案回答 question。

    search(query, k) 返回查询的前 k 封邮件；测试使用你的 BM25 索引构造此函数，
    k 由你选择。search 是一个接收查询字符串和 Top-K 数量的可调用对象。
    返回 {"answer": str | None, "support": list[str]}；找不到答案时 answer 为 None。

    测试检查：
    * 无论模型如何回复，每个问题最多调用 search 三次，即跳数预算为 3。
    * support 是邮件 ID 列表，每个 ID 必须来自本次调用期间 search 返回的邮件；
      该列表允许为空。
    * search 无结果或模型回复不符合任何格式时，函数不得抛异常。
    * 真实模型测试使用 tests/fixtures/email_qa.json：
      至少 70% 的答案包含一个可接受答案；每个正确答案的 support
      必须包含至少一封标准依据邮件。"""
    # 用邮件 ID 去重并累计各轮证据；这些状态只属于本次问答。
    from cs329z_hw1.pipelines.email_qa import answer_question

    return answer_question(question, search, lm)


# ======================================================================
# 第 2 部分：Cardinal 智能体
# ======================================================================


def run_tool_registry(tools: list[ToolSpec]) -> Any:
    """题目 tool_registry：按 tools 的输入顺序创建并返回工具注册表。

    测试使用返回对象的三个方法：

        registry.schemas() -> list[dict]
        registry.validate(name: str, args: dict) -> ToolResult | None
        registry.execute(name: str, args: dict) -> ToolResult

    测试检查：
    * schemas() 按注册顺序为每个工具返回一个字典，
      且只能包含 name、description、parameters 三个键。
    * validate 返回与相同调用的 execute 一致的 unknown_tool 或 invalid_args 结果；
      调用可执行时返回 None。它不执行工具，也不抛异常。
    * execute 不抛异常。名称未知时返回 unknown_tool。
      args 不符合工具 parameters 的 JSON Schema 时返回 invalid_args，且不执行工具。
      工具函数抛异常时返回 tool_error，content 包含 str(exception)。
      否则恰好调用一次 fn(**args)，返回 ok，且 content == str(return value)。
    * 发生上述任何结果后，注册表仍可继续使用。"""
    from cs329z_hw1.agent.tool_registry import ToolRegistry

    return ToolRegistry(tools)


def run_render_tools(schemas: list[dict]) -> str:
    """题目 tool_protocol：生成工具说明提示词。

    输入 schemas 是 registry.schemas() 的输出。返回文本须描述这些工具及调用方式，
    包含每个工具的名称和参数名。
    真实模型测试：模型阅读该说明后，必须正确完成
    tests/fixtures/agent_protocol_requests.json 中至少 18/20 个请求的工具调用。"""
    raise NotImplementedError


def run_parse_response(text: str) -> ParsedResponse:
    """题目 tool_protocol：将一次模型回复解析为 ParsedResponse(text, tool_call, error)。

    测试检查：
    * 对任何字符串均不抛异常。
    * text 是移除工具调用后、去掉首尾空白的回复文本。
      解析 run_format_tool_call("Looking.", call) 时，text == "Looking."。
    * 无工具调用时，tool_call 和 error 均为 None。
    * 多于一个工具调用时，tool_call 为 None，error 为非空字符串。
    * 格式化回复在任意字符处被截断时，tool_call 必须为 None 或与原调用完全一致，
      不能变成另一个调用。截断位置处于参数内部时，tool_call 为 None，且设置 error。"""
    raise NotImplementedError


def run_format_tool_call(text: str, call: ToolCall) -> str:
    """题目 tool_protocol 的测试适配器：将 text 和 call 写成你定义格式的助手回复。

    ScriptedLM 用它将预设工具调用转换为智能体能够解析的文本。

    测试检查：
    * run_parse_response(run_format_tool_call(text, call)) 不产生错误，
      name 相同，参数与 call.args 相等且保留 JSON 类型区别：
      1、1.0、true 和 "1" 不能混为一谈；text == text.strip()。
      输入 text 和 call.args 均可能为空。
    * 参数值可以是任意字符串，包括包含你所定义格式的完整工具调用的字符串。
    * 不修改 call。"""
    raise NotImplementedError


def run_agent_session(
    lm: LMCallable,
    tools: Optional[list[ToolSpec]],
    config: AgentConfig,
) -> Any:
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
    raise NotImplementedError


def run_build_doc_index(docs: list[dict]) -> Any:
    """题目 search_docs：为公司文档建立索引。

    docs 来自 cs329z_hw1.data.load_docs()，每份文档包含 doc_id、title、text。
    返回 run_search_docs 和 run_read_doc 所需的数据，
    例如文档或分块的 BM25 索引及原始文本。
    测试只构建一次，并将索引传给这两个入口。
    search_docs 和 read_doc 工具应在首次需要时构建索引，并共享该索引。"""
    raise NotImplementedError


def run_search_docs(index: Any, query: str, k: int = 5) -> list[SearchResult]:
    """题目 search_docs：检索 index 中的文档。

    测试检查：
    * 返回最多 k 个字典，包含 doc_id、title、snippet。
      doc_id 和 title 与 cs329z_hw1.data.load_docs() 的值一致。
    * 无匹配或查询无 token 时返回 []。
    * snippet 按 count_tokens 计数不超过 150 token，
      且与原文共享至少 4 个连续单词。
    * 某词只在一份文档出现时，该文档必须排第一，且 snippet 包含该词。
    * 相同调用每次都返回相同结果。"""
    raise NotImplementedError


def run_read_doc(index: Any, doc_id: str, start_line: int = 1) -> DocWindow:
    """题目 search_docs：读取 index 中一份文档的有限行窗口。

    行来自 cs329z_hw1.data.load_docs() 文本的 text.splitlines()，从 1 开始编号。

    测试检查：
    * start_line 与传入参数一致；text 必须恰好由 start_line..end_line
      的各行用换行连接而成，不附加行号；total_lines 为全文行数。
    * text 按 count_tokens 计数不超过 1500 token。
    * 从 end_line + 1 再读时连续无缺口，最后窗口的 end_line 等于 total_lines。
    * doc_id 不存在，或 start_line 不在 1..total_lines 内时，
      抛出 ValueError 或 LookupError。"""
    raise NotImplementedError
