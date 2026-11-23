"""Hookless discovery against the start time of the agent in the pane.

A live pane lifts the age limit on transcripts, and Codex writes a new
session's transcript only at its first turn, so the newest transcript in a
directory can be the previous session's. Discovery must not bind a transcript
written before the agent started, and must read a transcript the agent began
from its first line.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ccgram.handlers.recovery import transcript_discovery as discovery
from ccgram.providers.base import SessionStartEvent
from ccgram.window_state_ports import identity_state

MODULE = "ccgram.handlers.recovery.transcript_discovery"


def _identity() -> identity_state.IdentityProjection:
    return identity_state.IdentityProjection(
        window_id="w1",
        cwd="/repo",
        session_id="",
        transcript_path=None,
        provider_name="codex",
        window_name="agent",
        approval_mode="default",
    )


def _transcript(tmp_path: Path, *, began: float, written: float) -> Path:
    """A Codex-like transcript whose first record is stamped *began*."""
    path = tmp_path / "rollout.jsonl"
    stamp = (
        datetime.fromtimestamp(began, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[
            :-3
        ]
        + "Z"
    )
    path.write_text(
        json.dumps(
            {"timestamp": stamp, "type": "session_meta", "payload": {"id": "s1"}}
        )
        + "\n"
    )
    os.utime(path, (written, written))
    return path


async def _register(path: Path, started_at: float | None) -> MagicMock:
    provider = MagicMock()
    provider.discover_transcript.return_value = SessionStartEvent(
        session_id="s1", cwd="/repo", transcript_path=str(path), window_key="herdr:w1"
    )
    sync = MagicMock()
    with (
        patch(f"{MODULE}._agent_started_at", return_value=started_at),
        patch(f"{MODULE}._session_id_already_bound", return_value=False),
        patch(f"{MODULE}.session_map_sync", sync),
    ):
        await discovery._find_and_register_transcript(
            "w1", _identity(), [("codex", provider)], pane_alive=True
        )
    return sync


async def test_skips_a_transcript_last_written_before_the_agent_started(
    tmp_path,
) -> None:
    now = time.time()
    previous = _transcript(tmp_path, began=now - 3600, written=now - 3500)

    sync = await _register(previous, started_at=now - 60)

    sync.write_hookless_session_map.assert_not_called()
    sync.register_hookless_session.assert_not_called()


async def test_the_agents_new_transcript_is_read_from_its_first_line(tmp_path) -> None:
    now = time.time()
    own = _transcript(tmp_path, began=now - 30, written=now - 5)

    sync = await _register(own, started_at=now - 60)

    assert sync.write_hookless_session_map.call_args.kwargs["replay_from_start"] is True


async def test_a_resumed_transcript_is_bound_without_replaying_it(tmp_path) -> None:
    now = time.time()
    resumed = _transcript(tmp_path, began=now - 86400, written=now - 5)

    sync = await _register(resumed, started_at=now - 60)

    assert (
        sync.write_hookless_session_map.call_args.kwargs["replay_from_start"] is False
    )


async def test_unknown_agent_start_keeps_binding_the_newest_transcript(
    tmp_path,
) -> None:
    now = time.time()
    previous = _transcript(tmp_path, began=now - 3600, written=now - 3500)

    sync = await _register(previous, started_at=None)

    assert (
        sync.write_hookless_session_map.call_args.kwargs["replay_from_start"] is False
    )


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
def test_process_start_time_of_this_process_is_in_the_past() -> None:
    started = discovery._process_started_at(os.getpid())

    assert started is not None
    assert 0 <= time.time() - started < 7 * 86400


def test_transcript_start_reads_the_first_records_timestamp(tmp_path) -> None:
    path = _transcript(tmp_path, began=1795000000.25, written=time.time())

    assert discovery._transcript_started_at(str(path)) == pytest.approx(
        1795000000.25, abs=0.01
    )
    assert discovery._transcript_started_at(str(tmp_path / "missing.jsonl")) is None
