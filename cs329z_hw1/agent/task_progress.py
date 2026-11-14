"""从实际工具事件维护进度，不依赖有损摘要保存已读清单。"""

import json


class TaskProgress:
    def __init__(self):
        self.queries = []
        self.read_ids = []
        self.tool_count = 0
        self.stage_reads = 0

    def record(self, call, result):
        self.tool_count += 1
        if result.status != "ok":
            return
        if call.name in {"search_emails", "search_docs"}:
            query = call.args.get("query", "")
            if query not in self.queries:
                self.queries.append(query)
        if call.name in {"read_email", "read_doc"}:
            identifier = call.args.get("email_id", call.args.get("doc_id"))
            if identifier not in self.read_ids:
                self.read_ids.append(identifier)
                self.stage_reads += 1

    def instruction(self, remaining):
        # 完整记录留在 Python；仅展示有界的最近记录，避免状态本身挤满上下文。
        text = (
            "Execution progress (metadata, not evidence): "
            + json.dumps(
                {
                    "queries": [q[:160] for q in self.queries[-8:]],
                    "read_ids": self.read_ids[-30:],
                    "unique_read_count": len(self.read_ids),
                    "remaining_model_calls": remaining,
                },
                ensure_ascii=False,
            )
            + "\nAvoid repeating reads or equivalent searches unless verification is necessary. "
            "For a broad investigation, first review up to five representative items, "
            "then give a cited preliminary overview and ask which direction to investigate next. "
            "Do not claim exhaustive coverage from top-k searches. "
            "This sampling policy does not replace explicit requests for exact answers or exhaustive computation."
        )
        if self.stage_reads >= 5:
            text += "\nThis stage has read five or more items. For a broad task, deliver the preliminary overview now."
        if remaining <= 2:
            text += "\nConclude soon using existing evidence; state unresolved questions and coverage limits."
        if remaining == 1:
            text += "\nFINAL MODEL CALL: return a plain-text answer or partial findings now; do not request another tool."
        return text

    def stopped_message(self):
        return (
            "Model turn limit reached. No final conclusion was generated. "
            f"Completed {self.tool_count} tool attempts; read {len(self.read_ids)} distinct items. "
            "Recent evidence IDs: "
            + (", ".join(self.read_ids[-12:]) or "(none)")
            + ". Narrow the scope or request a preliminary summary of existing evidence."
        )


def bounded_search_results(results, max_chars):
    """先调整列表与片段，再序列化；不会生成被截断的 JSON。"""
    accepted = []
    for item in results:
        item = dict(item)
        if "snippet" in item:
            item["snippet"] = item["snippet"][:240]
        candidate = {
            "results": accepted + [item],
            "returned_count": len(accepted) + 1,
            "truncated": True,
        }
        if len(json.dumps(candidate, ensure_ascii=False)) > max_chars:
            break
        accepted.append(item)
    return json.dumps(
        {
            "results": accepted,
            "returned_count": len(accepted),
            "truncated": len(accepted) < len(results),
        },
        ensure_ascii=False,
    )
