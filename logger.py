"""
Logging system for PawPal+ Agent.

Records every Agent interaction including user input, reasoning steps,
tool calls, guardrail checks, and final responses. Supports both
file-based and in-memory logging for testing.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass
class ToolCall:
    """Record of a single tool invocation."""

    tool_name: str
    arguments: dict
    result: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class GuardrailLog:
    """Record of guardrail checks for one interaction."""

    passed: bool
    warnings: list[str]
    confidence: float = 0.0
    response_modified: bool = False


@dataclass
class InteractionLog:
    """Complete log of one user-agent interaction."""

    user_message: str
    intent: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    guardrail: GuardrailLog | None = None
    agent_response: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: str | None = None

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        data = {
            "timestamp": self.timestamp,
            "user_message": self.user_message,
            "intent": self.intent,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": tc.result[:500],  # Truncate long results
                    "timestamp": tc.timestamp,
                }
                for tc in self.tool_calls
            ],
            "agent_response": self.agent_response[:1000],  # Truncate long responses
        }
        if self.guardrail:
            data["guardrail"] = {
                "passed": self.guardrail.passed,
                "warnings": self.guardrail.warnings,
                "confidence": self.guardrail.confidence,
                "response_modified": self.guardrail.response_modified,
            }
        if self.error:
            data["error"] = self.error
        return data


class AgentLogger:
    """Logger that records Agent interactions to file and memory."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.history: list[InteractionLog] = []
        os.makedirs(log_dir, exist_ok=True)

    def start_interaction(self, user_message: str, intent: str) -> InteractionLog:
        """Begin logging a new interaction. Returns the log object to populate."""
        log = InteractionLog(user_message=user_message, intent=intent)
        self.history.append(log)
        return log

    def log_tool_call(
        self, interaction: InteractionLog, tool_name: str, arguments: dict, result: str
    ) -> None:
        """Record a tool call within an interaction."""
        interaction.tool_calls.append(
            ToolCall(tool_name=tool_name, arguments=arguments, result=result)
        )

    def log_guardrail(
        self,
        interaction: InteractionLog,
        passed: bool,
        warnings: list[str],
        confidence: float = 0.0,
        response_modified: bool = False,
    ) -> None:
        """Record guardrail check results."""
        interaction.guardrail = GuardrailLog(
            passed=passed,
            warnings=warnings,
            confidence=confidence,
            response_modified=response_modified,
        )

    def log_response(self, interaction: InteractionLog, response: str) -> None:
        """Record the final Agent response."""
        interaction.agent_response = response

    def log_error(self, interaction: InteractionLog, error: str) -> None:
        """Record an error that occurred during the interaction."""
        interaction.error = error

    def save_to_file(self) -> str:
        """Save all logged interactions to a JSON file. Returns the filepath."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.log_dir, f"agent_log_{timestamp}.json")
        data = [log.to_dict() for log in self.history]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return filepath

    def get_summary(self) -> dict:
        """Generate a summary of all logged interactions."""
        if not self.history:
            return {"total_interactions": 0}

        total = len(self.history)
        tool_calls = sum(len(log.tool_calls) for log in self.history)
        errors = sum(1 for log in self.history if log.error)
        guardrail_triggers = sum(
            1
            for log in self.history
            if log.guardrail and not log.guardrail.passed
        )

        # Count tool usage frequency
        tool_freq: dict[str, int] = {}
        for log in self.history:
            for tc in log.tool_calls:
                tool_freq[tc.tool_name] = tool_freq.get(tc.tool_name, 0) + 1

        return {
            "total_interactions": total,
            "total_tool_calls": tool_calls,
            "errors": errors,
            "guardrail_triggers": guardrail_triggers,
            "tool_usage": tool_freq,
        }

    def format_summary(self) -> str:
        """Return a human-readable summary string."""
        s = self.get_summary()
        lines = [
            f"Agent Log Summary",
            f"  Total interactions: {s['total_interactions']}",
            f"  Total tool calls:   {s['total_tool_calls']}",
            f"  Errors:             {s['errors']}",
            f"  Guardrail triggers: {s['guardrail_triggers']}",
        ]
        if s.get("tool_usage"):
            lines.append("  Tool usage:")
            for tool, count in sorted(
                s["tool_usage"].items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"    {tool}: {count}")
        return "\n".join(lines)
