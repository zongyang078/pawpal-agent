"""Provider abstraction for the agent's reasoning loop.

The agent speaks one neutral protocol -- a transcript in, an `LLMResponse` out
-- and each adapter translates to and from its vendor's wire format. That keeps
the ReAct loop in `agent.py` provider-agnostic and, more usefully, makes it
testable: `FakeClient` scripts a tool-calling conversation with no network.

Adding a provider means writing one adapter, not a second copy of the loop.
"""

import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# --- Neutral protocol types ---


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ToolResult:
    """The outcome of running a ToolCall, to be fed back to the model."""

    id: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """One model reply: either free text, tool calls, or both."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class UserTurn:
    content: str


@dataclass(frozen=True)
class AssistantTurn:
    text: str | None
    tool_calls: list[ToolCall]


@dataclass(frozen=True)
class ToolResultsTurn:
    results: list[ToolResult]


Turn = UserTurn | AssistantTurn | ToolResultsTurn


@runtime_checkable
class LLMClient(Protocol):
    """What the agent needs from a model provider."""

    model: str

    def complete(
        self, system: str, transcript: list[Turn], tools: list[dict]
    ) -> LLMResponse:
        """Send the conversation so far and return the model's next reply.

        `tools` are the neutral definitions from `tools.TOOL_DEFINITIONS`; each
        adapter reshapes them for its own API.
        """
        ...


class LLMError(RuntimeError):
    """A provider call failed. Raised so the agent can decide how to degrade."""


# --- Adapters ---


class OpenAIClient:
    """Adapter for the OpenAI chat completions API."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        # Imported lazily so the package is only required in LLM mode.
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    @staticmethod
    def _tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

    @staticmethod
    def _messages(system: str, transcript: list[Turn]) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system}]
        for turn in transcript:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.content})
            elif isinstance(turn, AssistantTurn):
                message: dict = {"role": "assistant", "content": turn.text}
                if turn.tool_calls:
                    message["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in turn.tool_calls
                    ]
                messages.append(message)
            else:
                messages.extend(
                    {"role": "tool", "tool_call_id": r.id, "content": r.content}
                    for r in turn.results
                )
        return messages

    def complete(
        self, system: str, transcript: list[Turn], tools: list[dict]
    ) -> LLMResponse:
        try:
            response = self._get_client().chat.completions.create(
                model=self.model,
                messages=self._messages(system, transcript),
                tools=self._tools(tools),
                tool_choice="auto",
            )
        except Exception as e:
            raise LLMError(f"OpenAI request failed: {e}") from e

        message = response.choices[0].message
        calls = []
        for tc in message.tool_calls or []:
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError) as e:
                # A malformed arguments blob is the model's error, not a crash.
                raise LLMError(
                    f"OpenAI returned unparseable arguments for {tc.function.name}: {e}"
                ) from e
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))

        return LLMResponse(text=message.content, tool_calls=calls)


class AnthropicClient:
    """Adapter for the Anthropic messages API."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514",
                 max_tokens: int = 1024):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    @staticmethod
    def _tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

    @staticmethod
    def _messages(transcript: list[Turn]) -> list[dict]:
        messages: list[dict] = []
        for turn in transcript:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.content})
            elif isinstance(turn, AssistantTurn):
                blocks: list[dict] = []
                if turn.text:
                    blocks.append({"type": "text", "text": turn.text})
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                    for tc in turn.tool_calls
                )
                messages.append({"role": "assistant", "content": blocks})
            else:
                # Anthropic carries tool results on a user turn.
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r.id,
                            "content": r.content,
                        }
                        for r in turn.results
                    ],
                })
        return messages

    def complete(
        self, system: str, transcript: list[Turn], tools: list[dict]
    ) -> LLMResponse:
        try:
            response = self._get_client().messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=self._messages(transcript),
                tools=self._tools(tools),
            )
        except Exception as e:
            raise LLMError(f"Anthropic request failed: {e}") from e

        text_parts = [b.text for b in response.content if b.type == "text"]
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        ]
        return LLMResponse(
            text=" ".join(text_parts) if text_parts else None,
            tool_calls=calls,
        )


class FakeClient:
    """Scripted client for tests. Replays `responses` in order, no network.

    Records every (system, transcript, tools) it was handed in `calls`, so a
    test can assert what the loop actually sent -- that tool results were fed
    back, that the transcript grew correctly, that the loop stopped when it
    should have.
    """

    def __init__(self, responses: list[LLMResponse], model: str = "fake-model"):
        self.responses = list(responses)
        self.model = model
        self.calls: list[dict] = []

    def complete(
        self, system: str, transcript: list[Turn], tools: list[dict]
    ) -> LLMResponse:
        self.calls.append(
            {"system": system, "transcript": list(transcript), "tools": tools}
        )
        if not self.responses:
            raise LLMError("FakeClient ran out of scripted responses")
        return self.responses.pop(0)


class FailingClient:
    """Client that always raises, for exercising the degradation path."""

    model = "failing-model"

    def __init__(self, message: str = "provider unavailable"):
        self.message = message

    def complete(self, system: str, transcript: list[Turn], tools: list[dict]):
        raise LLMError(self.message)


# --- Construction ---

# Defaults, not commitments. Model names go stale faster than the code around
# them, so OPENAI_MODEL / ANTHROPIC_MODEL override these without an edit --
# see model_from_env(). Both entries are cheap, fast models chosen because
# this workload is short tool-calling turns, not long-form generation.
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
}


def model_from_env(provider: str, env: dict) -> str | None:
    """Model override for `provider`, if the environment names one.

    A provider-specific variable wins over the catch-all, so you can pin one
    provider while leaving the other at its default.
    """
    return env.get(f"{provider.upper()}_MODEL") or env.get("PAWPAL_MODEL")


def build_client(
    provider: str, api_key: str | None, model: str | None = None
) -> LLMClient | None:
    """Build the adapter for `provider`, or None if it cannot be used.

    Returns None rather than raising when there is no key or the provider is
    unknown: the agent treats that as "run in rule-based mode".
    """
    if not api_key or provider not in DEFAULT_MODELS:
        return None
    model = model or DEFAULT_MODELS[provider]
    if provider == "openai":
        return OpenAIClient(api_key=api_key, model=model)
    return AnthropicClient(api_key=api_key, model=model)


PROVIDER_KEY_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def key_for_provider(provider: str, env: dict) -> str | None:
    """The key belonging to `provider`, or None.

    Provider and key must always be read together. Choosing them by separate
    rules is what sent an OpenAI key to Anthropic's endpoint: one expression
    picked the first key that existed, another picked the provider by whether
    a different variable was set.
    """
    var = PROVIDER_KEY_VARS.get(provider)
    return env.get(var) if var else None


def provider_from_env(env: dict) -> tuple[str, str | None]:
    """Pick a provider and its own key from the environment.

    Anthropic is checked first, so setting ANTHROPIC_API_KEY alone selects
    Anthropic. When both are set, pass an explicit provider to choose.
    """
    for provider in PROVIDER_KEY_VARS:
        key = key_for_provider(provider, env)
        if key:
            return provider, key
    return "openai", None
