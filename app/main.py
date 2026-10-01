"""FastAPI 入口：图编辑 API + 静态前端托管。"""
from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app import conclusion, database, graph, issuance, proposals, review
from app.schemas import (
    CommitRequest,
    CounterfactualRequest,
    IssuanceRequest,
    MergeCommitRequest,
    MergePreviewRequest,
    ProposalSaveRequest,
    ReviewBallotRequest,
    ReviewCloseRequest,
    ReviewCreateRequest,
    RollbackRequest,
)
from app.graph import GraphError


def get_conn():
    conn = database.connect()
    try:
        yield conn
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = database.connect()
    try:
        database.init_db(conn)
    finally:
        conn.close()
    yield


app = FastAPI(title="事故复盘因果图", version="1.0.0", lifespan=lifespan)


@app.exception_handler(GraphError)
async def graph_error_handler(request: Request, exc: GraphError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": exc.message, "code": exc.code, "cycle": exc.cycle,
                 "conflict": exc.conflict},
    )


@app.exception_handler(review.ReviewError)
async def review_error_handler(request: Request, exc: review.ReviewError) -> JSONResponse:
    # 过期封榜 / 重复封榜 / 票数不足等状态冲突由 review 层标记为 409
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "code": exc.code},
    )


@app.exception_handler(proposals.ProposalError)
async def proposal_error_handler(request: Request, exc: proposals.ProposalError) -> JSONResponse:
    # 提案已合并 / 裁决过期等状态冲突为 409；保存 / 合并中的整图校验错误仍走 GraphError
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "code": exc.code},
    )


@app.exception_handler(issuance.IssuanceError)
async def issuance_error_handler(request: Request, exc: issuance.IssuanceError) -> JSONResponse:
    # 已签发 / 未封榜等状态冲突为 409；业务条件不满足为 400 并附逐项校验结果
    content: dict[str, Any] = {"detail": exc.message, "code": exc.code}
    if exc.validation is not None:
        content["validation"] = exc.validation
    return JSONResponse(status_code=exc.status_code, content=content)


# ---------- 序列化 ----------

def node_dict(n: graph.Node, windows: dict | None = None, reviews: dict | None = None) -> dict[str, Any]:
    data = {"id": n.id, "type": n.type, "title": n.title, "description": n.description,
            "pos_x": n.pos_x, "pos_y": n.pos_y, "created_at": n.created_at,
            "earliest_time": n.earliest_time, "latest_time": n.latest_time,
            "reliability": n.reliability, "source_group": n.source_group,
            # 触发模式随每个快照原样返回；证据节点恒为 ANY（推演时忽略）
            "trigger_mode": n.trigger_mode if n.type == "event" else None}
    if windows is not None:
        data["time_window"] = graph.time_window_dict(windows["nodes"], windows["windows"], n.id)
    if reviews is not None:
        data["evidence_review"] = graph.evidence_review_dict(reviews["nodes"], reviews["reviews"], n.id)
    return data


def edge_dict(e: graph.Edge) -> dict[str, Any]:
    return {"id": e.id, "source_id": e.source_id, "target_id": e.target_id,
            "kind": e.kind, "label": e.label, "created_at": e.created_at}


def graph_dict(g: graph.Graph) -> dict[str, Any]:
    # 有效时间窗与证据强度均按当前所查看的版本重算（历史快照即按该版本拓扑/数据重算）
    windows, _conflict = graph.compute_time_windows(g.nodes, g.edges)
    reviews = graph.compute_evidence_review(g.nodes, g.edges)
    win_ctx = {"nodes": g.nodes, "windows": windows}
    rev_ctx = {"nodes": g.nodes, "reviews": reviews}
    return {
        "version_id": g.version_id,
        "nodes": [node_dict(n, win_ctx, rev_ctx) for n in g.nodes.values()],
        "edges": [edge_dict(e) for e in g.edges.values()],
    }


def version_dict(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    node_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE version_id = ?", (row["id"],)).fetchone()[0]
    edge_count = conn.execute("SELECT COUNT(*) FROM edges WHERE version_id = ?", (row["id"],)).fetchone()[0]
    merged_ids = [r[0] for r in conn.execute(
        "SELECT id FROM proposals WHERE merged_version_id = ? ORDER BY id", (row["id"],))]
    return {
        "id": row["id"],
        "message": row["message"],
        "kind": row["kind"],
        "base_id": row["base_id"],
        "created_at": row["created_at"],
        "node_count": node_count,
        "edge_count": edge_count,
        "merged_proposal_ids": merged_ids,
    }


# ---------- 查询 ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/state")
def get_state(conn: sqlite3.Connection = Depends(get_conn)):
    vid = graph.current_version_id(conn)
    g = graph.load_graph(conn, vid)
    return graph_dict(g)


@app.get("/api/versions")
def list_versions(conn: sqlite3.Connection = Depends(get_conn)):
    rows = conn.execute("SELECT * FROM versions ORDER BY id DESC").fetchall()
    current = graph.current_version_id(conn)
    return {"current_version_id": current, "versions": [version_dict(conn, r) for r in rows]}


@app.get("/api/versions/{version_id}")
def get_version(version_id: int, conn: sqlite3.Connection = Depends(get_conn)):
    g = graph.load_graph(conn, version_id)
    if g is None:
        raise HTTPException(status_code=404, detail=f"版本不存在：{version_id}")
    return graph_dict(g)


@app.get("/api/nodes/{node_id}/downstream")
def node_downstream(node_id: str, version_id: int | None = None, conn: sqlite3.Connection = Depends(get_conn)):
    vid = version_id or graph.current_version_id(conn)
    g = graph.load_graph(conn, vid)
    if g is None:
        raise HTTPException(status_code=404, detail=f"版本不存在：{vid}")
    try:
        return graph.downstream_of(g, node_id)
    except GraphError as exc:
        raise HTTPException(status_code=404, detail=exc.message)


# ---------- 反事实干预沙盘（只读：不写库、不产生 changeset / 版本） ----------

@app.post("/api/counterfactual/simulate")
def counterfactual_simulate(body: CounterfactualRequest, conn: sqlite3.Connection = Depends(get_conn)):
    vid = body.version_id or graph.current_version_id(conn)
    g = graph.load_graph(conn, vid)
    if g is None:
        raise HTTPException(status_code=404, detail=f"版本不存在：{vid}")
    try:
        return graph.simulate_counterfactual(g, body.target_id, body.prevented_ids)
    except GraphError as exc:
        raise HTTPException(status_code=400, detail=exc.message)


# ---------- 可复算复盘结论包（只读：不写库、不生成版本、不改变工作图） ----------

@app.get("/api/conclusion/export/{event_id}")
def export_conclusion(event_id: str, version_id: int | None = None,
                      conn: sqlite3.Connection = Depends(get_conn)):
    """导出自包含结论包文件：事件 + 全部 causes 祖先 + 关联证据 + 版本元数据 +
    按包内子图重算的时间窗 / 证据审查结果，并附 payload 的 SHA-256 摘要。"""
    vid = version_id or graph.current_version_id(conn)
    g = graph.load_graph(conn, vid)
    if g is None:
        raise HTTPException(status_code=404, detail=f"版本不存在：{vid}")
    row = conn.execute("SELECT * FROM versions WHERE id = ?", (vid,)).fetchone()
    try:
        package = conclusion.build_package(g, row, event_id)
    except GraphError as exc:
        raise HTTPException(status_code=404, detail=exc.message)
    # 文件内容本身即规范化 JSON（键按字典序、节点/连线按 id 排序），下载后可原样核验
    content = conclusion.package_file_bytes(package)
    filename = f"conclusion-{event_id}-v{vid}-{package['digest']['value'][:8]}.json"
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/conclusion/verify")
async def verify_conclusion(request: Request):
    """只读核验上传的结论包：先校验 digest，再仅凭包内原始节点与连线重建 DAG，
    重算时间窗与证据审查并与随包结果逐字段比对。不接触数据库。"""
    raw = await request.body()
    if len(raw) > conclusion.MAX_PACKAGE_BYTES:
        raise HTTPException(status_code=413, detail="结论包文件过大")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="文件不是有效的 UTF-8 JSON")
    try:
        return conclusion.verify_package(doc)
    except conclusion.PackageError as exc:
        raise HTTPException(status_code=400, detail=f"结论包结构非法：{exc}")


# ---------- 根因共识盲评（绑定不可变版本；只读写 review_* 表，不改写图 / 版本 / 结论包） ----------

@app.get("/api/reviews")
def list_reviews(conn: sqlite3.Connection = Depends(get_conn)):
    return {"sessions": review.list_sessions(conn)}


@app.post("/api/reviews", status_code=201)
def create_review(body: ReviewCreateRequest, conn: sqlite3.Connection = Depends(get_conn)):
    return review.create_session(
        conn,
        title=body.title,
        version_id=body.version_id,
        candidate_ids=body.candidate_ids,
        reviewers=body.reviewers,
        min_votes=body.min_votes,
    )


@app.get("/api/reviews/{session_id}")
def get_review(session_id: int, conn: sqlite3.Connection = Depends(get_conn)):
    """评审场详情。封榜前只含已提交名单与数量（不泄露任何排序）；封榜后附 Schulze 计票结果。"""
    return review.get_session(conn, session_id)


@app.post("/api/reviews/{session_id}/ballots")
def submit_review_ballot(session_id: int, body: ReviewBallotRequest,
                         conn: sqlite3.Connection = Depends(get_conn)):
    """提交 / 覆盖选票：须为登记评审人，排序须覆盖全部候选且不重复；封榜后一律拒绝。"""
    return review.submit_ballot(conn, session_id, reviewer=body.reviewer, ranking=body.ranking)


@app.post("/api/reviews/{session_id}/close")
def close_review(session_id: int, body: ReviewCloseRequest,
                 conn: sqlite3.Connection = Depends(get_conn)):
    """封榜：写事务内校验 session_revision 与最低票数并冻结选票；过期 / 重复封榜返回 409。"""
    return review.close_session(conn, session_id, expected_revision=body.expected_revision)


# ---------- 根因结论签发（只写 conclusion_issuances；不改写图 / 评审票 / 历史快照） ----------

@app.post("/api/reviews/{session_id}/issuance/validate")
def validate_issuance(session_id: int, body: IssuanceRequest,
                      conn: sqlite3.Connection = Depends(get_conn)):
    """只读预校验：重读绑定版本、Schulze 名次、证据审查与有效时间窗，
    返回逐项校验结果与每个候选的名次 / 证据状态 / 时间窗。不写任何表；正式签发时
    会在写事务内以同一口径重新计算，不沿用本结果。"""
    return issuance.build_validation(conn, session_id, body.root_cause_ids, body.note)


@app.post("/api/reviews/{session_id}/issuance", status_code=201)
def issue_conclusion(session_id: int, body: IssuanceRequest,
                     conn: sqlite3.Connection = Depends(get_conn)):
    """签发：同一 SQLite 写事务内重读全部依据并逐项校验，全部通过才写入不可修改的签发记录；
    任一条件不满足返回 400，响应 validation 字段逐项说明原因。"""
    return issuance.issue(conn, session_id, body.root_cause_ids, body.note)


@app.get("/api/reviews/{session_id}/issuance")
def get_session_issuance(session_id: int, conn: sqlite3.Connection = Depends(get_conn)):
    record = issuance.get_issuance_by_session(conn, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"评审场 {session_id} 尚无签发记录")
    return record


@app.get("/api/issuances")
def list_issuances(conn: sqlite3.Connection = Depends(get_conn)):
    return issuance.list_issuances(conn)


@app.get("/api/issuances/{issuance_id}")
def get_issuance(issuance_id: int, conn: sqlite3.Connection = Depends(get_conn)):
    return issuance.get_issuance(conn, issuance_id)


# ---------- 提交一组修改（事务：校验整张图，任一失败全部拒绝） ----------

@app.post("/api/changes/commit", response_model_exclude_none=True)
def commit_changes(body: CommitRequest, conn: sqlite3.Connection = Depends(get_conn)):
    ch = body.changes
    if not any([ch.node_creates, ch.node_updates, ch.node_deletes,
                ch.edge_creates, ch.edge_updates, ch.edge_deletes]):
        raise GraphError("没有任何可提交的修改", "empty_changeset")

    try:
        conn.execute("BEGIN IMMEDIATE")

        current = graph.current_version_id(conn)
        if ch.base_version_id != current:
            raise GraphError(
                f"基准版本已过期（页面基于 v{ch.base_version_id}，当前为 v{current}），请刷新后重新提交",
                "stale_version",
            )

        old = graph.load_graph(conn, current)
        # 任何校验失败（含成环）都抛 GraphError → 回滚，不留半成品
        new_nodes, new_edges = graph.apply_changes(old, ch)

        cur = conn.execute(
            "INSERT INTO versions (message, kind, base_id) VALUES (?, 'commit', ?)",
            (body.message, current),
        )
        new_id = cur.lastrowid
        graph.persist_graph(conn, new_id, new_nodes, new_edges)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    row = conn.execute("SELECT * FROM versions WHERE id = ?", (new_id,)).fetchone()
    return {"version_id": new_id, "message": body.message, "created_at": row["created_at"]}


# ---------- 回滚：把旧版快照复制为一个新版本 ----------

@app.post("/api/versions/rollback")
def rollback(body: RollbackRequest, conn: sqlite3.Connection = Depends(get_conn)):
    target = graph.load_graph(conn, body.target_version_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"版本不存在：{body.target_version_id}")

    current = graph.current_version_id(conn)
    if body.target_version_id == current:
        raise GraphError("该版本就是当前版本，无需回滚", "already_current")

    message = body.message or f"回滚到 v{body.target_version_id}"
    try:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            "INSERT INTO versions (message, kind, base_id) VALUES (?, 'rollback', ?)",
            (message, body.target_version_id),
        )
        new_id = cur.lastrowid
        graph.persist_graph(conn, new_id, target.nodes, target.edges)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    row = conn.execute("SELECT * FROM versions WHERE id = ?", (new_id,)).fetchone()
    return {"version_id": new_id, "message": message, "created_at": row["created_at"],
            "restored_from": body.target_version_id}


# ---------- 变更提案：保存（不生成版本）/ 预览 / 三方合并 ----------

@app.get("/api/proposals")
def list_change_proposals(conn: sqlite3.Connection = Depends(get_conn)):
    return {"proposals": proposals.list_proposals(conn)}


@app.post("/api/proposals", status_code=201)
def save_change_proposal(body: ProposalSaveRequest, conn: sqlite3.Connection = Depends(get_conn)):
    """把一组未提交修改保存为绑定 base_version_id 的提案。

    保存时在对应基线上复用 graph.apply_changes 整图校验，只写 proposals 表，不生成版本。
    """
    ch = body.changes
    if not any([ch.node_creates, ch.node_updates, ch.node_deletes,
                ch.edge_creates, ch.edge_updates, ch.edge_deletes]):
        raise GraphError("没有任何可保存的修改", "empty_changeset")
    return proposals.save_proposal(conn, title=body.title, changes=ch)


@app.get("/api/proposals/{proposal_id}")
def get_change_proposal(proposal_id: int, conn: sqlite3.Connection = Depends(get_conn)):
    return proposals.get_proposal(conn, proposal_id)


@app.get("/api/proposals/{proposal_id}/preview")
def preview_change_proposal(proposal_id: int, conn: sqlite3.Connection = Depends(get_conn)):
    """单提案只读预览：返回基线图与「应用该提案后」的完整图（不写库、不生成版本）。"""
    info, base, applied = proposals.proposal_graphs(conn, proposal_id)
    return {
        "proposal": info,
        "base_version_id": info["base_version_id"],
        "base_graph": graph_dict(base),
        "graph": graph_dict(applied),
    }


@app.post("/api/proposals/merge-preview")
def preview_proposal_merge(body: MergePreviewRequest, conn: sqlite3.Connection = Depends(get_conn)):
    """三方比较（只读）：基线 / 提案 A / 提案 B，列出自动采纳项、冲突与合并预览图。"""
    result = proposals.preview_merge(conn, body)
    merged_graph = result.pop("merged")
    result["merged_graph_full"] = graph_dict(merged_graph)
    return result


@app.post("/api/proposals/merge")
def merge_proposals(body: MergeCommitRequest, conn: sqlite3.Connection = Depends(get_conn)):
    """按逐项裁决在写事务中重新合并：复查提案状态与基线版本、重跑整图校验，
    全部通过才生成一个 merge 版本并同时标记两份提案已合并；任一失败全部回滚。"""
    return proposals.commit_merge(conn, body)


# 静态前端（API 路由先注册，未匹配的路径交给 StaticFiles）
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
