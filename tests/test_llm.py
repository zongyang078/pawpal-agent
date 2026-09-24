"""Tests for the provider adapters.

Adapter translation is checked against captured request shapes rather than the
live APIs: what matters here is that a neutral transcript turns into the wire
format each vendor expects, and that their replies come back normalised.
"""

import json
from types import SimpleNamespace

import pytest

from llm import (
    AnthropicClient,
    AssistantTurn,
    FailingClient,
    FakeClient,
    LLMError,
    LLMResponse,
    OpenAIClient,
    ToolCall,
    ToolResult,
    ToolResultsTurn,
    UserTurn,
    build_client,
    provider_from_env,
)

TOOLS = [{
    "name": "get_schedule",
    "description": "Get today's schedule.",
    "parameters": {"type": "object", "properties": {}},
}]

TRANSCRIPT = [
    UserTurn(content="what's on today?"),
    AssistantTurn(text=None, tool_calls=[ToolCall(id="c1", name="get_schedule", arguments={})]),
    ToolResultsTurn(results=[ToolResult(id="c1", content="No tasks scheduled.")]),
]


class TestOpenAIAdapter:
    """Translation to and from the chat completions format."""

    def test_transcript_becomes_openai_messages(self):
        messages = OpenAIClient._messages("be helpful", TRANSCRIPT)

        assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
        assert messages[0]["content"] == "be helpful"
        call = messages[2]["tool_calls"][0]
        assert call["id"] == "c1"
        assert call["function"]["name"] == "get_schedule"
        assert json.loads(call["function"]["arguments"]) == {}
        assert messages[3]["tool_call_id"] == "c1"
        assert messages[3]["content"] == "No tasks scheduled."

    def test_tools_are_wrapped_in_a_function_envelope(self):
        wrapped = OpenAIClient._tools(TOOLS)
        assert wrapped[0]["type"] == "function"
        assert wrapped[0]["function"]["name"] == "get_schedule"
        assert wrapped[0]["function"]["parameters"] == TOOLS[0]["parameters"]

    def test_response_with_tool_calls_is_normalised(self, monkeypatch):
        raw = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                id="c9",
                function=SimpleNamespace(name="add_pet",
                                         arguments='{"name": "Mochi", "species": "dog"}'),
            )],
        ))])
        client = OpenAIClient(api_key="k")
        monkeypatch.setattr(client, "_get_client", lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: raw))
        ))

        response = client.complete("sys", [UserTurn(content="hi")], TOOLS)

        assert response.text is None
        assert response.tool_calls == [
            ToolCall(id="c9", name="add_pet", arguments={"name": "Mochi", "species": "dog"})
        ]

    def test_unparseable_arguments_raise_llm_error(self, monkeypatch):
        raw = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                id="c9", function=SimpleNamespace(name="add_pet", arguments="{not json"),
            )],
        ))])
        client = OpenAIClient(api_key="k")
        monkeypatch.setattr(client, "_get_client", lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: raw))
        ))

        with pytest.raises(LLMError, match="unparseable arguments"):
            client.complete("sys", [UserTurn(content="hi")], TOOLS)

    def test_transport_failure_becomes_llm_error(self, monkeypatch):
        def boom(**kwargs):
            raise ConnectionError("no route to host")

        client = OpenAIClient(api_key="k")
        monkeypatch.setattr(client, "_get_client", lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=boom))
        ))

        with pytest.raises(LLMError, match="OpenAI request failed"):
            client.complete("sys", [UserTurn(content="hi")], TOOLS)


class TestAnthropicAdapter:
    """Translation to and from the messages format."""

    def test_transcript_becomes_content_blocks(self):
        messages = AnthropicClient._messages(TRANSCRIPT)

        # Tool results ride on a user turn in this API, not a dedicated role.
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert messages[1]["content"][0]["type"] == "tool_use"
        assert messages[1]["content"][0]["id"] == "c1"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "c1"

    def test_assistant_text_precedes_tool_use(self):
        transcript = [AssistantTurn(
            text="Let me look.",
            tool_calls=[ToolCall(id="c1", name="get_schedule", arguments={})],
        )]
        blocks = AnthropicClient._messages(transcript)[0]["content"]
        assert [b["type"] for b in blocks] == ["text", "tool_use"]

    def test_tools_use_input_schema(self):
        wrapped = AnthropicClient._tools(TOOLS)
        assert wrapped[0]["input_schema"] == TOOLS[0]["parameters"]
        assert "parameters" not in wrapped[0]

    def test_response_blocks_are_normalised(self, monkeypatch):
        raw = SimpleNamespace(content=[
            SimpleNamespace(type="text", text="Adding her now."),
            SimpleNamespace(type="tool_use", id="c9", name="add_pet",
                            input={"name": "Luna", "species": "cat"}),
        ])
        client = AnthropicClient(api_key="k")
        monkeypatch.setattr(client, "_get_client", lambda: SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kw: raw)
        ))

        response = client.complete("sys", [UserTurn(content="hi")], TOOLS)

        assert response.text == "Adding her now."
        assert response.tool_calls == [
            ToolCall(id="c9", name="add_pet", arguments={"name": "Luna", "species": "cat"})
        ]

    def test_transport_failure_becomes_llm_error(self, monkeypatch):
        def boom(**kwargs):
            raise TimeoutError("timed out")

        client = AnthropicClient(api_key="k")
        monkeypatch.setattr(client, "_get_client", lambda: SimpleNamespace(
            messages=SimpleNamespace(create=boom)
        ))

        with pytest.raises(LLMError, match="Anthropic request failed"):
            client.complete("sys", [UserTurn(content="hi")], TOOLS)


class TestFakeClient:
    """The test double itself, since every agent LLM test leans on it."""

    def test_replays_in_order_and_records_calls(self):
        client = FakeClient([LLMResponse(text="one"), LLMResponse(text="two")])

        assert client.complete("sys", [], TOOLS).text == "one"
        assert client.complete("sys", [], TOOLS).text == "two"
        assert len(client.calls) == 2

    def test_running_out_of_script_raises(self):
        with pytest.raises(LLMError, match="ran out"):
            FakeClient([]).complete("sys", [], TOOLS)


class TestClientConstruction:
    """build_client() and provider selection."""

    def test_builds_the_requested_adapter(self):
        assert isinstance(build_client("openai", "k"), OpenAIClient)
        assert isinstance(build_client("anthropic", "k"), AnthropicClient)

    def test_no_key_means_no_client(self):
        assert build_client("openai", None) is None

    def test_unknown_provider_means_no_client(self):
        assert build_client("ollama", "k") is None

    def test_explicit_model_overrides_the_default(self):
        assert build_client("openai", "k", "gpt-4o").model == "gpt-4o"

    def test_anthropic_key_alone_selects_anthropic(self):
        """The old code took the key but left the provider at its openai default."""
        assert provider_from_env({"ANTHROPIC_API_KEY": "k"}) == ("anthropic", "k")

    def test_openai_key_alone_selects_openai(self):
        assert provider_from_env({"OPENAI_API_KEY": "k"}) == ("openai", "k")

    def test_no_keys_means_no_provider(self):
        assert provider_from_env({}) == ("openai", None)


class TestFailingClient:
    def test_always_raises(self):
        with pytest.raises(LLMError):
            FailingClient().complete("sys", [], TOOLS)
