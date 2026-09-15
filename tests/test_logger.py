"""Tests for the interaction logger."""

import json

from logger import AgentLogger


class TestAgentLogger:
    """Recording, summarising, and exporting interactions."""

    def test_start_interaction(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path))
        log = logger.start_interaction("Hello", "general_chat")
        assert log.user_message == "Hello"
        assert log.intent == "general_chat"
        assert len(logger.history) == 1

    def test_log_tool_call(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path))
        log = logger.start_interaction("test", "test")
        logger.log_tool_call(log, "get_schedule", {}, "No tasks")
        assert len(log.tool_calls) == 1
        assert log.tool_calls[0].tool_name == "get_schedule"

    def test_summary_counts_interactions_and_tool_usage(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path))
        log1 = logger.start_interaction("msg1", "add_pet")
        logger.log_tool_call(log1, "add_pet", {"name": "Mochi"}, "Added")
        log2 = logger.start_interaction("msg2", "get_schedule")
        logger.log_tool_call(log2, "get_schedule", {}, "No tasks")

        summary = logger.get_summary()
        assert summary["total_interactions"] == 2
        assert summary["total_tool_calls"] == 2
        assert summary["tool_usage"]["add_pet"] == 1

    def test_summary_on_empty_history(self, tmp_path):
        assert AgentLogger(log_dir=str(tmp_path)).get_summary() == {"total_interactions": 0}

    def test_summary_counts_errors_and_guardrail_triggers(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path))
        log = logger.start_interaction("boom", "general_chat")
        logger.log_error(log, "something exploded")
        logger.log_guardrail(log, passed=False, warnings=["w"])

        summary = logger.get_summary()
        assert summary["errors"] == 1
        assert summary["guardrail_triggers"] == 1

    def test_serialization(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path))
        log = logger.start_interaction("Hello", "general_chat")
        logger.log_response(log, "Hi there!")
        data = log.to_dict()
        assert data["user_message"] == "Hello"
        assert data["agent_response"] == "Hi there!"

    def test_save_to_file_writes_valid_json(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path))
        log = logger.start_interaction("Hello", "general_chat")
        logger.log_tool_call(log, "get_schedule", {}, "No tasks")
        logger.log_response(log, "Hi there!")

        filepath = logger.save_to_file()
        with open(filepath) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["user_message"] == "Hello"
        assert data[0]["tool_calls"][0]["tool_name"] == "get_schedule"

    def test_long_fields_are_truncated_on_export(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path))
        log = logger.start_interaction("Hello", "general_chat")
        logger.log_tool_call(log, "search_care_info", {}, "x" * 2000)
        logger.log_response(log, "y" * 5000)

        data = log.to_dict()
        assert len(data["tool_calls"][0]["result"]) == 500
        assert len(data["agent_response"]) == 1000
