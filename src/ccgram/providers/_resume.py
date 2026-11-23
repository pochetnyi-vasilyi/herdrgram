"""Provider-side discovery helpers for resumable sessions."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from ccgram.providers.base import ResumableSession
from ccgram.utils import read_session_metadata_from_jsonl

_IndexParseError = (json.JSONDecodeError, OSError)

# Claude Code re-appends its title records as a session goes on, so the end of
# a transcript holds the current name; the start is read only as a fallback.
_TITLE_SCAN_BYTES = 256 * 1024


def index_message_count(entry: dict) -> int | None:
    """Return a positive message-count hint from a Claude index entry."""
    for key in ("messageCount", "msgCount", "msg_count", "messages"):
        value = entry.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def discover_claude_sessions(
    projects_path: Path,
    *,
    cwd: str | None = None,
    limit: int | None = None,
) -> list[ResumableSession]:
    """Discover Claude sessions, optionally restricted to one exact workspace."""
    resolved_cwd = _resolve_path(cwd) if cwd else None
    if cwd and resolved_cwd is None:
        return []

    candidates: list[tuple[float, ResumableSession, Path]] = []
    seen_ids: set[str] = set()
    if not projects_path.exists():
        return []

    try:
        project_dirs = list(projects_path.iterdir())
    except OSError:
        return []

    for project_dir in project_dirs:
        if not project_dir.is_dir():
            continue
        index_file = project_dir / "sessions-index.json"
        if index_file.exists():
            _scan_index(index_file, resolved_cwd, seen_ids, candidates)
        _scan_bare_jsonl(project_dir, resolved_cwd, seen_ids, candidates)

    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen = candidates[:limit] if limit is not None else candidates
    return [
        replace(session, title=read_claude_session_title(path))
        for _, session, path in chosen
    ]


def read_claude_session_title(path: Path) -> str:
    """The session's name as Claude Code shows it in the terminal title.

    A name given with ``/rename`` (a ``custom-title`` record) wins over the one
    Claude generated (``ai-title``); the latest record of each kind counts.
    Empty when the transcript holds neither or cannot be read.
    """
    try:
        with path.open("rb") as f:
            size = f.seek(0, 2)
            start = max(0, size - _TITLE_SCAN_BYTES)
            f.seek(start)
            tail = f.read()
            if start == 0:
                return _title_from(tail, partial_first_line=False)
            title = _title_from(tail, partial_first_line=True)
            if title:
                return title
            f.seek(0)
            return _title_from(f.read(_TITLE_SCAN_BYTES), partial_first_line=False)
    except OSError:
        return ""


def _title_from(data: bytes, *, partial_first_line: bool) -> str:
    lines = data.split(b"\n")
    if partial_first_line:
        lines = lines[1:]
    custom = generated = ""
    for line in lines:
        if b'"custom-title"' not in line and b'"ai-title"' not in line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        kind = entry.get("type")
        if kind == "custom-title":
            value = entry.get("customTitle")
            if isinstance(value, str) and value.strip():
                custom = value.strip()
        elif kind == "ai-title":
            value = entry.get("aiTitle")
            if isinstance(value, str) and value.strip():
                generated = value.strip()
    return custom or generated


def _resolve_path(path: str) -> str | None:
    try:
        return str(Path(path).expanduser().resolve())
    except OSError, ValueError:
        return None


def _scan_index(
    index_file: Path,
    resolved_cwd: str | None,
    seen_ids: set[str],
    candidates: list[tuple[float, ResumableSession, Path]],
) -> None:
    try:
        index_data = json.loads(index_file.read_text(encoding="utf-8"))
    except _IndexParseError:
        return
    if not isinstance(index_data, dict):
        return

    original_path = index_data.get("originalPath", "")
    entries = index_data.get("entries", [])
    if not isinstance(entries, list):
        return

    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        session = _parse_index_entry(
            raw_entry,
            original_path=original_path,
            resolved_cwd=resolved_cwd,
            seen_ids=seen_ids,
        )
        if session is not None:
            seen_ids.add(session.session_id)
            candidates.append(
                (session.mtime, session, Path(raw_entry.get("fullPath", "")))
            )


def _parse_index_entry(
    entry: dict,
    *,
    original_path: object,
    resolved_cwd: str | None,
    seen_ids: set[str],
) -> ResumableSession | None:
    session_id = entry.get("sessionId", "")
    full_path = entry.get("fullPath", "")
    if not isinstance(session_id, str) or not session_id or session_id in seen_ids:
        return None
    if not isinstance(full_path, str) or not full_path:
        return None

    project_path = entry.get("projectPath", original_path)
    if not isinstance(project_path, str) or not project_path:
        return None
    normalized_project = _resolve_path(project_path)
    if normalized_project is None or (
        resolved_cwd is not None and normalized_project != resolved_cwd
    ):
        return None

    session_file = Path(full_path)
    if not session_file.is_file():
        return None
    try:
        mtime = session_file.stat().st_mtime
    except OSError:
        mtime = 0.0

    summary = entry.get("summary", "") or entry.get("firstPrompt", "")
    if not isinstance(summary, str):
        summary = ""
    return ResumableSession(
        session_id=session_id,
        summary=summary or session_id[:12],
        cwd=normalized_project,
        provider_name="claude",
        mtime=mtime,
        msg_count=index_message_count(entry),
    )


def _scan_bare_jsonl(
    project_dir: Path,
    resolved_cwd: str | None,
    seen_ids: set[str],
    candidates: list[tuple[float, ResumableSession, Path]],
) -> None:
    try:
        jsonl_files = list(project_dir.glob("*.jsonl"))
    except OSError:
        return

    for jsonl_file in jsonl_files:
        session_id = jsonl_file.stem
        if session_id in seen_ids:
            continue
        file_cwd, summary = read_session_metadata_from_jsonl(jsonl_file)
        if not file_cwd:
            continue
        normalized_project = _resolve_path(file_cwd)
        if normalized_project is None:
            continue
        if resolved_cwd is not None and normalized_project != resolved_cwd:
            continue
        try:
            mtime = jsonl_file.stat().st_mtime
        except OSError:
            mtime = 0.0

        seen_ids.add(session_id)
        candidates.append(
            (
                mtime,
                ResumableSession(
                    session_id=session_id,
                    summary=summary or session_id[:12],
                    cwd=normalized_project,
                    provider_name="claude",
                    mtime=mtime,
                ),
                jsonl_file,
            )
        )
