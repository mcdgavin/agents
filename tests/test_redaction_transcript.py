from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.job import JobContext
from livekit.agents.llm import FunctionCall, FunctionCallOutput
from livekit.agents.voice.agent_session import _RECORDING_ALL_ON
from livekit.agents.voice.events import (
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
    UserInputTranscribedEvent,
)
from livekit.agents.voice.redaction import (
    RedactionOptions,
    RedactionSink,
    RegexRedactor,
)

from .fake_llm import FakeLLM

pytestmark = pytest.mark.unit

_SENTINEL_TEXT = "my card is 4242 4242 4242 4242"


def _make_job_ctx(session: AgentSession) -> JobContext:
    ctx = JobContext.__new__(JobContext)
    ctx._primary_agent_session = session
    info = MagicMock()
    info.url = "wss://test.livekit.cloud"
    info.job.id = "test-job-id"
    info.job.room.sid = "test-room-sid"
    info.job.room.name = "test-room"
    info.job.agent_name = "test-agent"
    ctx._info = info
    ctx._tagger = MagicMock(evaluations=[], outcome=None)
    # short-circuit simulation_context() (used by _otel_metadata during upload)
    ctx._simulation_resolved = True
    ctx._simulation_ctx = None
    return ctx


async def _run_session_end(session: AgentSession) -> MagicMock:
    """Drive JobContext._on_session_end with the upload patched; return the mock upload."""
    session._recording_options = _RECORDING_ALL_ON
    ctx = _make_job_ctx(session)
    with (
        patch("livekit.agents.job._upload_session_report", new_callable=AsyncMock) as mock_upload,
        patch("livekit.agents.job.http_context.http_session", return_value=MagicMock()),
    ):
        await ctx._on_session_end()
    mock_upload.assert_called_once()
    return mock_upload


@pytest.mark.asyncio
async def test_transcript_sink_redacts_uploaded_session_report() -> None:
    session = AgentSession(
        llm=FakeLLM(),
        redaction=RedactionOptions(redactor=RegexRedactor(), sinks={RedactionSink.TRANSCRIPT}),
    )
    await session.start(Agent(instructions="test agent"))
    session.history.add_message(role="user", content=_SENTINEL_TEXT)
    await session.aclose()

    mock_upload = await _run_session_end(session)

    report = mock_upload.call_args.kwargs["report"]
    serialized = json.dumps(report.chat_history.to_dict())
    assert "4242" not in serialized
    assert "[CREDIT_CARD]" in serialized
    # the in-memory history must keep the raw value
    assert any(
        item.type == "message" and item.text_content == _SENTINEL_TEXT
        for item in session.history.items
    )


@pytest.mark.asyncio
async def test_transcript_sink_redacts_tool_calls_and_report_events() -> None:
    # LLM sink deliberately off: the model sees raw values and can echo them into
    # tool-call arguments, and the events list carries raw transcripts/items —
    # the persisted report (chat history AND events) must still come out clean
    session = AgentSession(
        llm=FakeLLM(),
        redaction=RedactionOptions(redactor=RegexRedactor(), sinks={RedactionSink.TRANSCRIPT}),
    )
    await session.start(Agent(instructions="test agent"))
    msg = session.history.add_message(role="user", content=_SENTINEL_TEXT)
    call = FunctionCall(
        name="charge_card", call_id="call_1", arguments='{"card": "4242 4242 4242 4242"}'
    )
    output = FunctionCallOutput(
        name="charge_card",
        call_id="call_1",
        output="charged card 4242 4242 4242 4242",
        is_error=False,
    )
    session.history.insert(call)
    session.history.insert(output)
    session._recorded_events.extend(
        [
            UserInputTranscribedEvent(transcript=_SENTINEL_TEXT, is_final=True),
            ConversationItemAddedEvent(item=msg),
            FunctionToolsExecutedEvent(function_calls=[call], function_call_outputs=[output]),
        ]
    )
    await session.aclose()

    mock_upload = await _run_session_end(session)

    report = mock_upload.call_args.kwargs["report"]
    serialized = json.dumps(report.to_dict())
    assert "4242" not in serialized
    assert "[CREDIT_CARD]" in serialized
    # the session's own state must keep the raw values
    assert any(
        item.type == "function_call" and "4242" in item.arguments for item in session.history.items
    )
    raw_transcripts = [
        ev.transcript for ev in session._recorded_events if ev.type == "user_input_transcribed"
    ]
    assert raw_transcripts == [_SENTINEL_TEXT]


@pytest.mark.asyncio
async def test_transcript_sink_disabled_leaves_report_raw() -> None:
    # default sinks = {LLM}: the transcript sink must require explicit opt-in
    session = AgentSession(
        llm=FakeLLM(),
        redaction=RedactionOptions(redactor=RegexRedactor()),
    )
    await session.start(Agent(instructions="test agent"))
    session.history.add_message(role="user", content=_SENTINEL_TEXT)
    await session.aclose()

    mock_upload = await _run_session_end(session)

    report = mock_upload.call_args.kwargs["report"]
    assert _SENTINEL_TEXT in json.dumps(report.chat_history.to_dict())


@pytest.mark.asyncio
async def test_no_redaction_configured_leaves_report_raw() -> None:
    session = AgentSession(llm=FakeLLM())
    await session.start(Agent(instructions="test agent"))
    session.history.add_message(role="user", content=_SENTINEL_TEXT)
    await session.aclose()

    mock_upload = await _run_session_end(session)

    report = mock_upload.call_args.kwargs["report"]
    assert _SENTINEL_TEXT in json.dumps(report.chat_history.to_dict())
