"""根因结论签发：从已封榜盲评场签发不可变版本的最终根因集合。

流程（全部在同一个 SQLite 写事务 BEGIN IMMEDIATE 内完成，不信任任何客户端预判）：
- 重读评审场（必须已封榜、同场尚无签发记录）与其绑定的不可变版本快照；
- 在绑定版本快照上重算 Schulze 名次（按冻结选票现算）、证据强度审查与有效时间窗；
- 业务条件，任一不满足都拒绝并逐项给出原因：
  1) 集合恰好覆盖最高并列名次（exact cover of top rank）；
  2) 候选之间不存在 causes 祖先关系（传递闭包；含直接 / 间接因果）；
  3) 每个候选的证据状态不得为 insufficient 或 refuted（仅 supported / disputed 允许）。
- 全部通过后写入一条此后不再修改的 conclusion_issuances 记录：
  规范化（紧凑、键按字典序）结论 payload 与其 SHA-256 摘要，关联评审场与版本。

同场不可重复签发（session_id 唯一 + 事务内复查）；记录只增不改，后续图版本变化不影响。
本模块只写 conclusion_issuances 表：不改写图、版本、评审票或任何历史快照。
"""
from __future__ import annotations

import json
import sqlite3
from collections import deque
from datetime import datetime, timezone
from typing import Any

from app import conclusion, graph, review

MAX_NOTE_LEN = 2000


class IssuanceError(Exception):
    """签发业务错误：status_code 由路由层原样返回（已签发 / 未封榜等为 409）。"""

    def __init__(self, message: str, code: str = "invalid", status_code: int = 400,
                 validation: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.validation = validation


# ---------- 读取辅助 ----------

def _session_row(conn: sqlite3.Connection, session_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM review_sessions WHERE id = ?", (session_id,),
    ).fetchone()
    if row is None:
        raise IssuanceError(f"评审场不存在：{session_id}", "session_not_found", 404)
    return row


def _existing_issuance(conn: sqlite3.Connection, session_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM conclusion_issuances WHERE session_id = ?", (session_id,),
    ).fetchone()


def _ballots(conn: sqlite3.Connection, session_id: int) -> list[list[str]]:
    rows = conn.execute(
        "SELECT ranking FROM review_ballots WHERE session_id = ? ORDER BY reviewer",
        (session_id,),
    ).fetchall()
    return [json.loads(r["ranking"]) for r in rows]


def _candidate_ids(conn: sqlite3.Connection, session_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT node_id FROM review_candidates WHERE session_id = ? ORDER BY node_id",
        (session_id,),
    ).fetchall()
    return [r["node_id"] for r in rows]


# ---------- causes 祖先关系（在绑定版本的事件 causes 子图上） ----------

def _cause_children(nodes: dict[str, graph.Node], edges: dict[str, graph.Edge]) -> dict[str, list[str]]:
    """children[u] = 沿 causes 边的直接下游事件（u 是其父），按 id 排序保证见证路径确定。"""
    events = {nid for nid, n in nodes.items() if n.type == "event"}
    children: dict[str, list[str]] = {nid: [] for nid in events}
    for e in edges.values():
        if e.kind == "causes" and e.source_id in events and e.target_id in events:
            children[e.source_id].append(e.target_id)
    for u in children:
        children[u].sort()
    return children


def _causal_path(children: dict[str, list[str]], src: str, dst: str) -> list[str] | None:
    """在 causes 子图上 BFS 求 src → dst 的确定性因果链（含两端）；不可达返回 None。"""
    if src not in children or dst not in children:
        return None
    prev: dict[str, str] = {}
    seen = {src}
    q: deque[str] = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        for v in children[u]:
            if v not in seen:
                seen.add(v)
                prev[v] = u
                q.append(v)
    if dst not in seen:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return path


# ---------- 校验（dry-run 与签发共用；纯读，不写任何表） ----------

def build_validation(conn: sqlite3.Connection, session_id: int,
                     submitted: list[str], note: str) -> dict[str, Any]:
    """在事务内重读全部依据并逐项校验，返回结构化校验结果。

    checks 每项形如 {key, label, passed, status[pending|passed|failed], reasons:[...]}；
    pending = 前置条件不满足（未封榜等）时无法执行该项检查。
    candidates 给出每个候选的名次、证据状态与有效时间窗，供前端逐项展示。
    """
    row = _session_row(conn, session_id)
    existing = _existing_issuance(conn, session_id)
    candidate_ids = _candidate_ids(conn, session_id)
    candidate_set = set(candidate_ids)
    version_id = row["version_id"]

    # 节点标题取绑定版本快照（版本不可变，展示随快照稳定）
    titles: dict[str, str] = {}
    if candidate_ids:
        marks = ",".join("?" for _ in candidate_ids)
        for r in conn.execute(
            f"SELECT id, title FROM nodes WHERE version_id = ? AND id IN ({marks})",
            (version_id, *candidate_ids),
        ):
            titles[r["id"]] = r["title"]

    # 0) 状态门槛
    state_reasons: list[dict[str, Any]] = []
    if existing is not None:
        state_reasons.append({"code": "already_issued",
                              "message": f"评审场 {session_id} 已签发结论（签发记录 #{existing['id']}），不可重复签发",
                              "issuance_id": existing["id"]})
    if row["status"] != "closed":
        state_reasons.append({"code": "session_not_closed",
                              "message": "评审场尚未封榜，只有已封榜的评审场才能签发根因结论"})
    if not (note or "").strip():
        state_reasons.append({"code": "empty_note", "message": "签发说明不能为空"})
    closed = row["status"] == "closed" and existing is None

    # 1) 重读绑定版本快照；快照与版本记录均不可变，缺失只可能是数据损坏
    g: graph.Graph | None = None
    version_row = conn.execute("SELECT * FROM versions WHERE id = ?", (version_id,)).fetchone()
    if version_row is None:
        state_reasons.append({"code": "version_not_found",
                              "message": f"评审场绑定的版本 v{version_id} 不存在"})
    else:
        g = graph.load_graph(conn, version_id)

    # 2) 按冻结选票重算 Schulze 名次（与封榜后展示完全同源，现算而非缓存）
    ranks: list[dict[str, Any]] = []
    top_rank: list[str] = []
    rank_of: dict[str, int] = {}
    if closed and g is not None:
        result = review.schulze_result(candidate_ids, _ballots(conn, session_id))
        ranks = result["ranks"]
        top_rank = ranks[0]["node_ids"] if ranks else []
        rank_of = {nid: layer["rank"] for layer in ranks for nid in layer["node_ids"]}

    # 节点 / 边按 id 排序后再计算，时间窗与证据审查结果确定（与结论包同口径）
    sorted_nodes = {nid: g.nodes[nid] for nid in sorted(g.nodes)} if g is not None else {}
    sorted_edges = {eid: g.edges[eid] for eid in sorted(g.edges)} if g is not None else {}
    windows = graph.compute_time_windows(sorted_nodes, sorted_edges)[0] if g is not None else {}
    reviews = graph.compute_evidence_review(sorted_nodes, sorted_edges) if g is not None else {}

    selected = [c for c in submitted if c in candidate_set]
    extra = [c for c in submitted if c not in candidate_set]

    # 3) 条件一：集合恰好覆盖最高并列名次
    cover_reasons: list[dict[str, Any]] = []
    if closed and g is not None:
        missing = [c for c in top_rank if c not in selected]
        non_top = [c for c in selected if c not in set(top_rank)]
        if missing:
            cover_reasons.append({
                "code": "missing_top_rank",
                "message": "缺少最高并列名次中的候选：" + _fmt_candidates(missing, titles),
                "node_ids": sorted(missing),
            })
        if non_top:
            cover_reasons.append({
                "code": "non_top_rank_included",
                "message": "包含不属于最高并列名次的候选：" + _fmt_candidates(non_top, titles),
                "node_ids": sorted(non_top),
            })
        if extra:
            cover_reasons.append({
                "code": "not_candidate",
                "message": "包含本场候选之外的节点：" + ", ".join(sorted(extra)),
                "node_ids": sorted(extra),
            })
    selected_dupes = sorted({c for c in submitted if submitted.count(c) > 1 and c in candidate_set})
    if closed and g is not None and selected_dupes:
        cover_reasons.append({
            "code": "duplicate_root_cause",
            "message": "提交的根因集合存在重复候选：" + _fmt_candidates(selected_dupes, titles),
            "node_ids": selected_dupes,
        })

    # 4) 条件二：候选之间不存在 causes 祖先关系（传递闭包）
    ancestor_reasons: list[dict[str, Any]] = []
    if closed and g is not None:
        children = _cause_children(sorted_nodes, sorted_edges)
        for i, a in enumerate(sorted(selected)):
            for b in sorted(selected)[i + 1:]:
                chain = _causal_path(children, a, b)
                if chain is not None:
                    ancestor_reasons.append({
                        "code": "causes_ancestor",
                        "message": (f"「{titles.get(a, a)}」沿 causes 是「{titles.get(b, b)}」的祖先，"
                                    "根因集合不得同时包含存在因果上下游关系的候选"),
                        "ancestor_id": a,
                        "descendant_id": b,
                        "chain_node_ids": chain,
                    })
                    continue
                chain = _causal_path(children, b, a)
                if chain is not None:
                    ancestor_reasons.append({
                        "code": "causes_ancestor",
                        "message": (f"「{titles.get(b, b)}」沿 causes 是「{titles.get(a, a)}」的祖先，"
                                    "根因集合不得同时包含存在因果上下游关系的候选"),
                        "ancestor_id": b,
                        "descendant_id": a,
                        "chain_node_ids": chain,
                    })

    # 5) 条件三：每个候选证据状态不得为 insufficient 或 refuted
    evidence_reasons: list[dict[str, Any]] = []
    candidate_statuses: list[dict[str, Any]] = []
    if closed and g is not None:
        for nid in sorted(selected):
            rv = reviews.get(nid)
            status = rv["status"] if rv is not None else "insufficient"
            if status in ("insufficient", "refuted"):
                evidence_reasons.append({
                    "code": f"evidence_{status}",
                    "message": (f"「{titles.get(nid, nid)}」的证据审查状态为 {status}，"
                                "最终根因的证据状态不得为 insufficient 或 refuted"),
                    "node_id": nid,
                    "evidence_status": status,
                })

    def check(key: str, label: str, reasons: list[dict[str, Any]]) -> dict[str, Any]:
        if not closed or g is None:
            return {"key": key, "label": label, "status": "pending", "passed": False, "reasons": []}
        return {"key": key, "label": label, "status": "passed" if not reasons else "failed",
                "passed": not reasons, "reasons": reasons}

    checks = [
        {"key": "session_state", "label": "评审场已封榜、同场未重复签发、签发说明非空",
         "status": "passed" if not state_reasons else "failed",
         "passed": not state_reasons, "reasons": state_reasons},
        check("top_rank_coverage", "集合恰好覆盖最高并列名次", cover_reasons),
        check("no_causes_ancestor", "候选之间不存在 causes 祖先关系", ancestor_reasons),
        check("evidence_sufficient", "每个候选证据状态均不为 insufficient / refuted", evidence_reasons),
    ]
    passed = all(c["passed"] for c in checks)

    # 每个候选的逐项依据（名次 / 证据状态 / 有效时间窗），无论是否选中、是否通过都返回
    candidates_out: list[dict[str, Any]] = []
    for nid in candidate_ids:
        rv = reviews.get(nid) if g is not None else None
        win = graph.time_window_dict(sorted_nodes, windows, nid) if g is not None else None
        candidates_out.append({
            "node_id": nid,
            "title": titles.get(nid, nid),
            "rank": rank_of.get(nid),
            "is_top_rank": nid in set(top_rank),
            "selected": nid in set(selected),
            "evidence_status": rv["status"] if rv is not None else None,
            "support_score": rv["support_score"] if rv is not None else None,
            "refute_score": rv["refute_score"] if rv is not None else None,
            "time_window": win,
        })

    return {
        "session_id": session_id,
        "version_id": version_id,
        "review_status": row["status"],
        "already_issued": existing is not None,
        "passed": passed,
        "submitted_root_cause_ids": list(submitted),
        "top_rank_node_ids": top_rank,
        "ranks": ranks,
        "checks": checks,
        "candidates": candidates_out,
        "existing_issuance_id": existing["id"] if existing is not None else None,
    }


def _fmt_candidates(ids: list[str], titles: dict[str, str]) -> str:
    return "、".join(f"「{titles.get(c, c)}」" for c in sorted(ids))


# ---------- 规范化 payload 与摘要 ----------

def _build_payload(
    *,
    session_row: sqlite3.Row,
    version_row: sqlite3.Row,
    root_cause_ids: list[str],
    note: str,
    ranks: list[dict[str, Any]],
    nodes: dict[str, graph.Node],
    windows: dict[str, dict[str, Any]],
    reviews: dict[str, dict[str, Any]],
    issued_at: str,
) -> dict[str, Any]:
    """签发记录冻结的规范化结论 payload（顺序无关：摘要按规范化 JSON 计算）。"""
    return {
        "format": "incident-root-cause-issuance",
        "format_version": 1,
        "issued_at": issued_at,
        "note": note,
        "session": {
            "id": session_row["id"],
            "title": session_row["title"],
            "revision": session_row["revision"],
            "min_votes": session_row["min_votes"],
            "closed_at": session_row["closed_at"],
        },
        "version": {
            "id": version_row["id"],
            "kind": version_row["kind"],
            "message": version_row["message"],
            "base_id": version_row["base_id"],
            "created_at": version_row["created_at"],
        },
        "schulze": {
            "method": "schulze",
            "ranks": ranks,
        },
        "root_causes": [
            {
                "node_id": nid,
                "title": nodes[nid].title,
                "rank": next((layer["rank"] for layer in ranks if nid in layer["node_ids"]), None),
                "evidence_review": reviews[nid],
                "time_window": graph.time_window_dict(nodes, windows, nid),
            }
            for nid in root_cause_ids
        ],
    }


# ---------- 签发（写事务） ----------

def issue(conn: sqlite3.Connection, session_id: int,
          root_cause_ids: list[str], note: str) -> dict[str, Any]:
    """在同一写事务内重读全部依据并逐项校验，全通过才写入不可修改的签发记录。"""
    note = (note or "").strip()
    try:
        conn.execute("BEGIN IMMEDIATE")
        result = build_validation(conn, session_id, root_cause_ids, note)
        # 同场已签发 / 未封榜都是状态冲突（409），与业务条件不满足（400 + 逐项原因）区分开
        if result["already_issued"]:
            raise IssuanceError(
                f"评审场 {session_id} 已存在签发记录，不可重复签发", "already_issued",
                status_code=409, validation=result)
        if result["review_status"] != "closed":
            raise IssuanceError(
                f"评审场 {session_id} 尚未封榜，不能签发根因结论", "session_not_closed",
                status_code=409, validation=result)
        if not result["passed"]:
            raise IssuanceError(
                "根因结论签发校验未通过，已逐项列出原因", "issuance_validation_failed",
                status_code=400, validation=result)

        row = _session_row(conn, session_id)
        version_row = conn.execute(
            "SELECT * FROM versions WHERE id = ?", (row["version_id"],)).fetchone()
        g = graph.load_graph(conn, row["version_id"])
        sorted_nodes = {nid: g.nodes[nid] for nid in sorted(g.nodes)}
        sorted_edges = {eid: g.edges[eid] for eid in sorted(g.edges)}
        windows = graph.compute_time_windows(sorted_nodes, sorted_edges)[0]
        reviews = graph.compute_evidence_review(sorted_nodes, sorted_edges)
        final_ids = sorted(set(root_cause_ids))
        issued_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        payload = _build_payload(
            session_row=row, version_row=version_row, root_cause_ids=final_ids,
            note=note, ranks=result["ranks"], nodes=sorted_nodes, windows=windows,
            reviews=reviews, issued_at=issued_at,
        )
        payload_bytes = conclusion.canonical_json_bytes(payload)
        digest = conclusion.sha256_hex(payload_bytes)
        cur = conn.execute(
            "INSERT INTO conclusion_issuances "
            "(session_id, version_id, note, root_cause_ids, payload_json, digest_sha256, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, row["version_id"], note,
             json.dumps(final_ids, ensure_ascii=False),
             payload_bytes.decode("utf-8"), digest, issued_at),
        )
        issuance_id = cur.lastrowid
        conn.execute("COMMIT")
    except IssuanceError:
        conn.execute("ROLLBACK")
        raise
    except sqlite3.IntegrityError as exc:
        conn.execute("ROLLBACK")
        # UNIQUE(session_id)：并发下另一事务已为同一场签发
        raise IssuanceError(f"评审场 {session_id} 已存在签发记录，不可重复签发",
                            "already_issued", 409) from exc
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return get_issuance(conn, issuance_id)


# ---------- 查询（只读） ----------

def _issuance_dict(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    payload = json.loads(row["payload_json"])
    actual = conclusion.sha256_hex(conclusion.canonical_json_bytes(payload))
    session = conn.execute(
        "SELECT title, status, revision FROM review_sessions WHERE id = ?",
        (row["session_id"],),
    ).fetchone()
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "session_title": session["title"] if session is not None else None,
        "review_status": session["status"] if session is not None else None,
        "version_id": row["version_id"],
        "note": row["note"],
        "root_cause_ids": json.loads(row["root_cause_ids"]),
        "issued_at": row["created_at"],
        "digest": {
            "algorithm": "sha256",
            "value": row["digest_sha256"],
            "actual": actual,
            "match": actual == row["digest_sha256"],
        },
        "payload": payload,
    }


def get_issuance_by_session(conn: sqlite3.Connection, session_id: int) -> dict[str, Any] | None:
    row = _existing_issuance(conn, session_id)
    return _issuance_dict(conn, row) if row is not None else None


def get_issuance(conn: sqlite3.Connection, issuance_id: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM conclusion_issuances WHERE id = ?", (issuance_id,)).fetchone()
    if row is None:
        raise IssuanceError(f"签发记录不存在：{issuance_id}", "issuance_not_found", 404)
    return _issuance_dict(conn, row)


def list_issuances(conn: sqlite3.Connection) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT * FROM conclusion_issuances ORDER BY id DESC",
    ).fetchall()
    summaries: list[dict[str, Any]] = []
    for r in rows:
        summaries.append({
            "id": r["id"],
            "session_id": r["session_id"],
            "version_id": r["version_id"],
            "root_cause_ids": json.loads(r["root_cause_ids"]),
            "issued_at": r["created_at"],
            "digest_sha256": r["digest_sha256"],
        })
    return {"issuances": summaries}


def session_issuance_summary(conn: sqlite3.Connection, session_id: int) -> dict[str, Any] | None:
    """评审场详情附带的签发摘要（不含 payload）。"""
    row = _existing_issuance(conn, session_id)
    if row is None:
        return None
    return {
        "id": row["id"],
        "version_id": row["version_id"],
        "root_cause_ids": json.loads(row["root_cause_ids"]),
        "issued_at": row["created_at"],
        "digest_sha256": row["digest_sha256"],
    }
