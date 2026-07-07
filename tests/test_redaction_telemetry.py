from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import Agent, AgentSession
from livekit.agents.llm import FunctionToolCall, function_tool
from livekit.agents.telemetry.traces import tracer as lk_tracer
from livekit.agents.voice.redaction import (
    RedactionOptions,
    RedactionSink,
    RegexRedactor,
)

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

_SENTINEL_TEXT = "my card is 4242 4242 4242 4242"
_REDACTED_TEXT = "my card is [CREDIT_CARD]"
_SENTINEL_IBAN = "DE89370400440532013000"


@pytest.fixture
def span_exporter() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    prev = lk_tracer._tracer_provider
    lk_tracer.set_provider(provider)
    yield exporter
    lk_tracer.set_provider(prev)


def _all_span_texts(exporter: InMemorySpanExporter) -> list[str]:
    """Flatten every span attribute value and event attribute value to strings."""
    texts: list[str] = []
    for span in exporter.get_finished_spans():
        for value in (span.attributes or {}).values():
            texts.append(str(value))
        for event in span.events:
            for value in (event.attributes or {}).values():
                texts.append(str(value))
    return texts


def _span_attr_texts(exporter: InMemorySpanExporter) -> list[str]:
    """Only span attribute values (excludes gen_ai.* events)."""
    texts: list[str] = []
    for span in exporter.get_finished_spans():
        for value in (span.attributes or {}).values():
            texts.append(str(value))
    return texts


async def _drive_turn(session: AgentSession, agent: Agent) -> None:
    await session.start(agent)
    try:
        await session.generate_reply(user_input=_SENTINEL_TEXT)
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_no_redaction_sentinel_visible_in_spans(
    span_exporter: InMemorySpanExporter,
) -> None:
    # control: proves the harness observes chat content in telemetry
    fake_llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(input=_SENTINEL_TEXT, content="ok", ttft=0.01, duration=0.02)
        ]
    )
    await _drive_turn(AgentSession(llm=fake_llm), Agent(instructions="test agent"))

    assert any("4242 4242 4242 4242" in text for text in _all_span_texts(span_exporter))


@pytest.mark.asyncio
async def test_llm_and_telemetry_sinks_keep_sentinel_out_of_all_spans(
    span_exporter: InMemorySpanExporter,
) -> None:
    @function_tool
    async def lookup_account() -> str:
        return f"account iban is {_SENTINEL_IBAN}"

    fake_llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input=_REDACTED_TEXT,
                content="",
                ttft=0.01,
                duration=0.02,
                tool_calls=[
                    FunctionToolCall(name="lookup_account", arguments="{}", call_id="call_1")
                ],
            ),
            FakeLLMResponse(
                input="account iban is [IBAN]", content="done", ttft=0.01, duration=0.02
            ),
        ]
    )
    session = AgentSession(
        llm=fake_llm,
        redaction=RedactionOptions(
            redactor=RegexRedactor(),
            sinks={RedactionSink.LLM, RedactionSink.TELEMETRY},
        ),
    )
    await _drive_turn(session, Agent(instructions="test agent", tools=[lookup_account]))

    leaked = [
        text
        for text in _all_span_texts(span_exporter)
        if "4242 4242 4242 4242" in text or _SENTINEL_IBAN in text
    ]
    assert leaked == []
    # sanity: telemetry still carries (redacted) content
    assert any("[CREDIT_CARD]" in text for text in _all_span_texts(span_exporter))
    # and the raw values are still in the in-memory history
    serialized_history = json.dumps(session.history.to_dict())
    assert "4242 4242 4242 4242" in serialized_history
    assert _SENTINEL_IBAN in serialized_history


@pytest.mark.asyncio
async def test_telemetry_only_sink_redacts_pipeline_span_attributes(
    span_exporter: InMemorySpanExporter,
) -> None:
    # telemetry-only: pipeline-owned span *attributes* must be redacted even
    # though the LLM itself receives raw text. (The gen_ai.* span events mirror
    # the actual LLM egress and are only redacted when the LLM sink is enabled.)
    fake_llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(input=_SENTINEL_TEXT, content="ok", ttft=0.01, duration=0.02)
        ]
    )
    session = AgentSession(
        llm=fake_llm,
        redaction=RedactionOptions(redactor=RegexRedactor(), sinks={RedactionSink.TELEMETRY}),
    )
    await _drive_turn(session, Agent(instructions="test agent"))

    leaked = [text for text in _span_attr_texts(span_exporter) if "4242 4242 4242 4242" in text]
    assert leaked == []


@pytest.mark.asyncio
async def test_telemetry_only_sink_redacts_tool_call_arguments_in_spans(
    span_exporter: InMemorySpanExporter,
) -> None:
    # with the LLM sink off, the model sees raw PII and can echo it into
    # tool-call arguments; span attributes (chat ctx, tool args, response
    # function calls) must still come out clean
    @function_tool
    async def charge_card(card: str) -> str:
        return "charged"

    fake_llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input=_SENTINEL_TEXT,
                content="",
                ttft=0.01,
                duration=0.02,
                tool_calls=[
                    FunctionToolCall(
                        name="charge_card",
                        arguments='{"card": "4242 4242 4242 4242"}',
                        call_id="call_1",
                    )
                ],
            ),
            FakeLLMResponse(input="charged", content="done", ttft=0.01, duration=0.02),
        ]
    )
    session = AgentSession(
        llm=fake_llm,
        redaction=RedactionOptions(redactor=RegexRedactor(), sinks={RedactionSink.TELEMETRY}),
    )
    await _drive_turn(session, Agent(instructions="test agent", tools=[charge_card]))

    leaked = [text for text in _span_attr_texts(span_exporter) if "4242 4242 4242 4242" in text]
    assert leaked == []
    assert any("[CREDIT_CARD]" in text for text in _span_attr_texts(span_exporter))
