"""Tests for the command-line entry point."""

import json

import pytest

import cli
from cli import build_agent, describe_mode, main
from pawpal_system import Owner


@pytest.fixture
def data_file(tmp_path) -> str:
    """Path to a data file inside a per-test temp directory."""
    return str(tmp_path / "data.json")


class TestSeed:
    """The `seed` subcommand."""

    def test_writes_sample_data(self, data_file, capsys):
        assert main(["--data", data_file, "seed"]) == 0
        with open(data_file) as f:
            data = json.load(f)
        assert [p["name"] for p in data["pets"]] == ["Mochi", "Luna"]
        assert "Wrote sample data" in capsys.readouterr().out

    def test_refuses_to_clobber_without_force(self, data_file, capsys):
        main(["--data", data_file, "seed"])
        assert main(["--data", data_file, "seed"]) == 1
        assert "already exists" in capsys.readouterr().err

    def test_force_overwrites(self, data_file):
        main(["--data", data_file, "seed"])
        assert main(["--data", data_file, "seed", "--force"]) == 0


class TestReadCommands:
    """The `schedule` and `pets` subcommands."""

    def test_schedule_reports_seeded_tasks_and_conflicts(self, data_file, capsys):
        main(["--data", data_file, "seed"])
        assert main(["--data", data_file, "schedule"]) == 0
        out = capsys.readouterr().out
        assert "Morning walk" in out
        assert "Conflict at 08:00" in out

    def test_schedule_on_empty_data(self, data_file, capsys):
        assert main(["--data", data_file, "schedule"]) == 0
        assert "No tasks scheduled" in capsys.readouterr().out

    def test_pets_lists_pending_counts(self, data_file, capsys):
        main(["--data", data_file, "seed"])
        assert main(["--data", data_file, "pets"]) == 0
        out = capsys.readouterr().out
        assert "Mochi (dog) - 3 pending" in out

    def test_pets_on_empty_data(self, data_file, capsys):
        assert main(["--data", data_file, "pets"]) == 0
        assert "No pets registered" in capsys.readouterr().out


class TestAsk:
    """The `ask` subcommand, in rule-based mode."""

    def test_answers_a_care_question(self, data_file, capsys):
        assert main(["--data", data_file, "ask", "How often should I bathe my dog?"]) == 0
        assert "Bathe dogs every 4-8 weeks" in capsys.readouterr().out

    def test_emergency_override(self, data_file, capsys):
        assert main(["--data", data_file, "ask", "My dog is not breathing!"]) == 0
        captured = capsys.readouterr()
        assert "emergency" in captured.out.lower()
        assert "Emergency keyword detected" in captured.err

    def test_trace_goes_to_stderr(self, data_file, capsys):
        main(["--data", data_file, "ask", "--trace", "How much should I feed my dog?"])
        captured = capsys.readouterr()
        assert "search_care_info" in captured.err
        assert "search_care_info" not in captured.out

    def test_writes_state_to_the_requested_path_only(self, tmp_path, monkeypatch):
        """--data must win over the tool layer's hardcoded "data.json" default."""
        monkeypatch.chdir(tmp_path)
        custom = str(tmp_path / "elsewhere.json")

        assert main(["--data", custom, "ask", "Add Mochi, a dog"]) == 0

        with open(custom) as f:
            assert json.load(f)["pets"][0]["name"] == "Mochi"
        assert not (tmp_path / "data.json").exists(), "state leaked to the cwd default"


class TestProviderFlag:
    """--provider exists so both providers can be verified without editing .env.

    These check selection only. Running a turn would reach the network, and the
    suite is hermetic by design -- the adapters themselves are covered in
    test_llm.py against captured response shapes.
    """

    @pytest.fixture(autouse=True)
    def _both_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-claude")
        for var in ("OPENAI_MODEL", "ANTHROPIC_MODEL", "PAWPAL_MODEL"):
            monkeypatch.delenv(var, raising=False)

    def test_is_offered_in_help(self, capsys):
        with pytest.raises(SystemExit):
            main(["--help"])
        assert "--provider" in capsys.readouterr().out

    def test_rejects_an_unknown_provider(self, capsys):
        with pytest.raises(SystemExit):
            main(["--provider", "ollama", "pets"])
        assert "invalid choice" in capsys.readouterr().err

    def test_default_picks_anthropic_when_both_keys_are_set(self):
        assert build_agent(Owner(name="T")).api_provider == "anthropic"

    def test_openai_can_be_forced(self):
        agent = build_agent(Owner(name="T"), "openai")
        assert agent.api_provider == "openai"
        assert agent.use_llm is True, "forcing a provider must still find its key"

    def test_describe_mode_names_the_forced_provider(self):
        assert describe_mode(build_agent(Owner(name="T"), "openai")).startswith("openai:")

    def test_the_flag_reaches_build_agent(self, data_file, monkeypatch):
        """The argparse wiring, with the provider neutralised so nothing dials out."""
        seen = []
        monkeypatch.setattr(
            cli, "build_agent",
            lambda owner, provider=None: (seen.append(provider), build_agent(owner, "nope"))[1],
        )

        main(["--data", data_file, "--provider", "openai", "ask", "hi"])

        assert seen == ["openai"]
