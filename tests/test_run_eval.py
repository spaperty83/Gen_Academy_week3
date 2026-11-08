"""MOCKED evaluation runner tests: no credentials or network."""

import json
from types import SimpleNamespace

import pytest

from scripts import run_eval as evaluation


def screen(kind, version=1, critique=None):
    """Minimal public API result."""
    return {"screen": {"kind": kind, "draft_version": version, "reason": "research unavailable"},
            "__interrupt__": [True] if kind in {"approval", "recovery", "export_failed"} else [],
            "draft_version": version, "critique_version": version if critique is None else critique,
            "itinerary": {"city": "Porto", "version": version}}


def simulate(monkeypatch, states):
    """Drive the public API surface with scripted snapshots."""
    actions = []
    remaining = iter(states)
    current = {}
    monkeypatch.setattr(evaluation, "build_graph", lambda saver: object())

    def stream(graph, **kwargs):
        nonlocal current
        actions.append(kwargs)
        current = next(remaining)
        yield {"orchestrator": {"dispatched": ["sights", "food"], "research_round": 1}}
        yield {"sights_researcher": {}, "food_researcher": {},
               "planner": {"itinerary": current.get("itinerary")}, "critic": {}}

    monkeypatch.setattr(evaluation, "stream_trip", stream)
    monkeypatch.setattr(evaluation, "get_trip", lambda graph, thread: current)
    result = evaluation.target({"user_input": "Porto", "revision": "Relaxed pace", "kind": "normal"})
    return result, actions


def test_revision_then_current_approval(monkeypatch):
    result, actions = simulate(monkeypatch, [screen("approval"), screen("approval", 2), screen("exported", 2)])
    assert [action.get("decision") for action in actions] == [None, "revise", "approve"]
    assert actions[-1]["draft_version"] == 2
    assert result["approved_itinerary"]["version"] == 2
    assert result["first_draft_s"] <= result["agent_time_s"]
    assert result["pause_kinds"] == ["approval", "approval"]
    assert result["dispatched_history"]


@pytest.mark.parametrize("kind", ["recovery", "approval"])
def test_recovery_and_unchecked_approval_never_approve(monkeypatch, kind):
    result, actions = simulate(monkeypatch, [screen(kind, critique=0), screen("cancelled")])
    assert actions[-1]["decision"] == "cancel"
    assert result["failure_reasons"]
    assert result["approved_itinerary"] is None


def test_one_export_retry_then_cancel(monkeypatch):
    result, actions = simulate(monkeypatch, [screen("approval"), screen("approval", 2),
                                           screen("export_failed", 2), screen("export_failed", 2),
                                           screen("cancelled", 2)])
    assert [action.get("decision") for action in actions] == [None, "revise", "approve", "retry_export", "cancel"]
    assert result["failure_reasons"] == ["export failed again"]


def test_six_pause_cap(monkeypatch):
    result, actions = simulate(monkeypatch, [screen("approval")] * 7)
    assert len(result["pause_kinds"]) == 6
    assert len(actions) == 7  # initial run and at most six resume actions
    assert "six-pause safety cap reached" in result["failure_reasons"]


def test_vague_clarification(monkeypatch):
    result, _ = simulate(monkeypatch, [screen("clarification") | {"itinerary": None}])
    assert evaluation.code_scores({"kind": "vague"}, result)["graceful_failure"]


def test_export_requires_complete_artifacts_and_approved_json(monkeypatch, tmp_path):
    path = tmp_path / "trip_v1.json"
    path.write_text('{"city": "Porto"}', encoding="utf-8")
    output = {"saved_path": str(path), "approved_itinerary": {"city": "Porto"}}
    monkeypatch.setattr(evaluation, "hard_checks", lambda state: [])
    monkeypatch.setattr(evaluation.export, "export_is_complete", lambda base: False)
    assert not evaluation.code_scores({"kind": "normal"}, output)["export_completed"]
    monkeypatch.setattr(evaluation.export, "export_is_complete", lambda base: True)
    assert evaluation.code_scores({"kind": "normal"}, output)["export_completed"]
    output["approved_itinerary"] = {"city": "Paris"}
    assert not evaluation.code_scores({"kind": "normal"}, output)["export_completed"]


def test_dataset_sync_skips_exact_examples():
    class Client:
        def __init__(self):
            self.examples = []
            self.created = 0

        def has_dataset(self, **kwargs):
            return True

        def read_dataset(self, **kwargs):
            return SimpleNamespace(id="dataset")

        def list_examples(self, **kwargs):
            return self.examples

        def create_examples(self, *, examples, **kwargs):
            self.created += len(examples)
            self.examples.extend(SimpleNamespace(inputs=item["inputs"]) for item in examples)

    client = Client()
    cases = [{"case_id": "a"}, {"case_id": "b"}]
    assert len(evaluation.sync_dataset(client, cases)) == 2
    assert len(evaluation.sync_dataset(client, cases)) == 2
    assert client.created == 2


def test_judge_reasks_once_and_shares_result(monkeypatch):
    calls = []

    class Model:
        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, prompt):
            calls.append(prompt)
            if len(calls) == 1:
                raise ValueError("invalid structured output")
            return {"revision_applied": True, "revision_reason": "Pace changed",
                    "usefulness": 4, "usefulness_reason": "Practical"}

    monkeypatch.setattr(evaluation, "get_llm", lambda **kwargs: Model())
    judges = evaluation.Judges()
    inputs, outputs = {"kind": "normal", "user_input": "Porto"}, {"thread_id": "unique"}
    assert judges.revision_applied(inputs, outputs)["score"] is True
    assert judges.usefulness(inputs, outputs)["score"] == 4
    assert len(calls) == 2
    assert "<tool_data>" in calls[0]


def test_cases_cover_required_variation():
    cases = json.loads((evaluation.ROOT / "tests/eval_cases.json").read_text())
    normal = [case for case in cases if case["kind"] == "normal"]
    assert len(normal) == 5 and len(cases) == 6
    assert all(case["revision"] for case in normal)
    assert any("Porto" in case["user_input"] for case in normal)
    assert any("7 days" in case["user_input"] and "packed" in case["user_input"] for case in normal)


def test_judge_exhausted_validation_is_unscored(monkeypatch):
    class Invalid:
        def with_structured_output(self, *args, **kwargs):
            return self

        def invoke(self, prompt):
            raise ValueError("bad output")

    monkeypatch.setattr(evaluation, "get_llm", lambda **kwargs: Invalid())
    judges = evaluation.Judges()
    inputs, outputs = {"kind": "normal", "user_input": "Porto"}, {"thread_id": "bad"}
    assert judges.revision_applied(inputs, outputs)["score"] is None
    assert judges.usefulness(inputs, outputs)["score"] is None


def test_report_primary_conjunction_and_dated_csv(tmp_path, caplog):
    class Results(list):
        url = "https://smith.langchain.com/test-experiment"

    results = Results()
    for index in range(6):
        vague = index == 5
        scores = {key: True for key in (*evaluation.METRICS, "revision_applied")}
        scores["first_draft_under_5_min"] = False  # secondary does not change success
        scores["usefulness"] = None if vague else 4
        if index == 4:
            scores["revision_applied"] = False
        if vague:
            scores = {key: None for key in scores} | {"graceful_failure": True}
        else:
            scores["graceful_failure"] = None
        results.append({"example": SimpleNamespace(inputs={"case_id": str(index), "kind": "vague" if vague else "normal"}),
                        "run": SimpleNamespace(outputs={"pause_kinds": ["approval", "approval"],
                                                         "agent_time_s": 10, "first_draft_s": 2}),
                        "evaluation_results": {"results": [SimpleNamespace(key=key, score=value)
                                                              for key, value in scores.items()]}})
    output = tmp_path / "eval_results.csv"
    with caplog.at_level("INFO"):
        assert evaluation.report(results, output) == 4
    text = output.read_text(encoding="utf-8")
    assert "4/5" in text and evaluation.LABEL in text and evaluation.SAMPLE in text
    assert "revision_applied" in text and "SUMMARY" in text
    assert "Successes: 4/5" in caplog.text
    assert "usefulness" in caplog.text and Results.url in caplog.text


def test_real_graph_revision_and_manifest_with_fake_providers(monkeypatch, tmp_path):
    from tests.test_failures import FailureRun

    for key in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(key, raising=False)
    FailureRun(monkeypatch, tmp_path / "outputs")
    inputs = {"case_id": "integration", "kind": "normal",
              "user_input": "1 day in Lisbon, mid, relaxed, history",
              "revision": "Change Lisbon to Paris"}
    result = evaluation.target(inputs)
    assert result["pause_kinds"] == ["approval", "approval"]
    assert result["initial_itinerary"]["city"] == "Lisbon"
    assert result["approved_itinerary"]["city"] == "Paris"
    scores = evaluation.code_scores(inputs, result)
    assert scores["export_completed"]
    assert scores["hard_checks_pass"]
    assert scores["delegation_sensible"]


@pytest.mark.parametrize("status, expected", [(401, "LANGSMITH_API_KEY"), (403, "LANGSMITH_WORKSPACE_ID"),
                                            (404, "region"), (429, "rate limit")])
def test_safe_langsmith_http_diagnostics(status, expected):
    import requests

    response = requests.Response()
    response.status_code = status
    error = requests.HTTPError("secret-api-key should never be logged", response=response)
    message = evaluation.safe_langsmith_error("dataset synchronization", error)
    assert f"HTTP {status}" in message
    assert expected in message
    assert "dataset synchronization" in message
    assert "secret-api-key" not in message


def test_langsmith_wrapper_preserves_safe_operation():
    with (
        pytest.raises(evaluation.EvaluationSetupError, match="HTTP 403") as caught,
        evaluation.langsmith_operation("dataset synchronization"),
    ):
        raise RuntimeError("403 Forbidden: private-response-data")
    assert "private-response-data" not in str(caught.value)


def test_read_only_langsmith_check(monkeypatch):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            calls.append("client")

        def list_datasets(self, *, limit):
            assert limit == 1
            calls.append("read")
            return iter([])

    monkeypatch.setattr(evaluation, "Client", Client)
    monkeypatch.setattr(evaluation.Settings, "load", lambda: SimpleNamespace(langsmith_api_key="fake"))
    monkeypatch.setattr(evaluation, "evaluate", lambda *args, **kwargs: pytest.fail("Must not run evaluation"))
    monkeypatch.setattr(evaluation, "sync_dataset", lambda *args: pytest.fail("Must not write dataset"))
    assert evaluation.main(["--check-langsmith"]) == 0
    assert calls == ["client", "read"]
