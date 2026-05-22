from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command, Interrupt

from scripts.evaluate_caseflow import (
    EvaluationError,
    _build_evaluation_report,
    _evaluate_live_cases,
    _non_negative_int,
)


def _case(
    case_id: str,
    *,
    intent: str = "咨询类",
    priority: str = "low",
    approval: bool = False,
) -> dict:
    return {
        "id": case_id,
        "user_query": "客户问怎么重置账号密码",
        "expected_intent": intent,
        "expected_priority": priority,
        "expected_needs_human_approval": approval,
        "expected_evidence_keywords": ["密码"],
        "draft_response_must_include": ["重置密码"],
        "evaluation_focus": ["classification"],
    }


def _result(
    *,
    intent: str = "咨询类",
    priority: str = "low",
    approval: bool = False,
    model_used: str = "openai-compatible",
) -> dict:
    return {
        "intent": intent,
        "priority": priority,
        "needs_human_approval": approval,
        "evidence": ["FAQ：密码重置说明"],
        "draft_response": "请在登录页点击重置密码。",
        "model_used": model_used,
    }


def test_live_report_includes_fallback_rate_models_and_bounded_examples():
    cases = [
        _case("eval-001"),
        _case("eval-002", intent="退款 / 升级类", priority="high", approval=True),
    ]
    results = [
        _result(model_used="openai-compatible"),
        _result(
            intent="投诉类",
            priority="medium",
            approval=True,
            model_used="openai-compatible:fallback",
        ),
    ]
    live_records = [
        {"interrupt_count": 0, "approval_resume": None},
        {"interrupt_count": 1, "approval_resume": "approve"},
    ]

    report = _build_evaluation_report(
        cases,
        results,
        mode="live-llm",
        live_records=live_records,
        examples_limit=1,
    )

    assert report["mode"] == "live-llm"
    assert report["fallback_rate"] == 0.5
    assert report["models_used"] == ["openai-compatible", "openai-compatible:fallback"]
    assert len(report["qualitative_examples"]) == 1
    example = report["qualitative_examples"][0]
    assert example["id"] == "eval-002"
    assert example["fallback_used"] is True
    assert example["interrupt_count"] == 1
    assert example["approval_resume"] == "approve"
    assert "intent" in example["mismatches"]
    assert "priority" in example["mismatches"]
    assert report["live_model_verification"] == {
        "status": "verified",
        "verified": True,
        "fallback_cases": 1,
        "non_fallback_cases": 1,
        "diagnostic": "At least one live eval case used non-fallback model output.",
    }


def test_live_report_marks_all_fallback_runs_as_not_verified():
    report = _build_evaluation_report(
        [_case("eval-001")],
        [_result(model_used="openai-compatible:fallback")],
        mode="live-llm",
        live_records=[{"interrupt_count": 0, "approval_resume": None}],
        examples_limit=1,
    )

    assert report["fallback_rate"] == 1.0
    assert report["live_model_verification"]["status"] == "fallback-only"
    assert report["live_model_verification"]["verified"] is False
    assert "before marking Story 5.2 done" in report["live_model_verification"]["diagnostic"]


def test_live_report_preserves_human_review_example_when_limit_is_small():
    cases = [
        _case("eval-001"),
        _case("eval-002"),
        _case("eval-003", intent="退款 / 升级类", priority="high", approval=True),
    ]
    results = [
        _result(intent="异常类", model_used="openai-compatible"),
        _result(intent="投诉类", model_used="openai-compatible"),
        _result(
            intent="退款 / 升级类",
            priority="high",
            approval=True,
            model_used="openai-compatible",
        ),
    ]
    live_records = [
        {"interrupt_count": 0, "approval_resume": None},
        {"interrupt_count": 0, "approval_resume": None},
        {"interrupt_count": 1, "approval_resume": "approve"},
    ]

    report = _build_evaluation_report(
        cases,
        results,
        mode="live-llm",
        live_records=live_records,
        examples_limit=1,
    )

    assert report["qualitative_examples"][0]["id"] == "eval-003"


class FinalGraph:
    def __init__(self, final_custom_data: dict):
        self.final_custom_data = final_custom_data
        self.calls = []
        self.checkpointer = object()

    async def ainvoke(self, input, config, stream_mode):
        self.calls.append(
            {
                "input": input,
                "thread_id": config["configurable"]["thread_id"],
                "stream_mode": stream_mode,
            }
        )
        return [
            (
                "values",
                {
                    "messages": [
                        AIMessage(
                            content="CaseFlow 处理结果",
                            additional_kwargs={"custom_data": self.final_custom_data},
                        ),
                        {"content": "non-standard trailing message"},
                    ]
                },
            )
        ]


@pytest.mark.asyncio
async def test_live_evaluation_handles_normal_non_interrupt_final_custom_data():
    graph = FinalGraph(_result(model_used="openai-compatible"))

    results, records = await _evaluate_live_cases(
        [_case("eval-001")],
        graph=graph,
        approval_response="approve",
    )

    assert results[0]["model_used"] == "openai-compatible"
    assert records[0]["interrupt_count"] == 0
    assert records[0]["approval_resume"] is None
    assert graph.calls[0]["thread_id"] == "eval-eval-001"
    assert graph.calls[0]["stream_mode"] == ["updates", "values"]


class InterruptThenFinalGraph:
    def __init__(self, final_custom_data: dict):
        self.final_custom_data = final_custom_data
        self.calls = []
        self.checkpointer = object()

    async def ainvoke(self, input, config, stream_mode):
        self.calls.append(
            {
                "input": input,
                "thread_id": config["configurable"]["thread_id"],
                "stream_mode": stream_mode,
            }
        )
        if len(self.calls) == 1:
            return [("updates", {"__interrupt__": [Interrupt(value="需要人工审批")]} )]
        return [
            (
                "values",
                {
                    "messages": [
                        AIMessage(
                            content="CaseFlow 处理结果",
                            additional_kwargs={"custom_data": self.final_custom_data},
                        )
                    ]
                },
            )
        ]


@pytest.mark.asyncio
async def test_live_evaluation_resumes_interrupt_with_same_thread_and_response():
    graph = InterruptThenFinalGraph(
        _result(
            intent="退款 / 升级类",
            priority="high",
            approval=True,
            model_used="openai-compatible",
        )
    )
    cases = [
        _case("eval-005", intent="退款 / 升级类", priority="high", approval=True)
    ]

    results, records = await _evaluate_live_cases(
        cases,
        graph=graph,
        approval_response="approve-for-eval",
    )

    assert results[0]["intent"] == "退款 / 升级类"
    assert records[0]["interrupt_count"] == 1
    assert records[0]["approval_resume"] == "approve-for-eval"
    assert graph.calls[0]["thread_id"] == "eval-eval-005"
    assert graph.calls[1]["thread_id"] == "eval-eval-005"
    assert graph.calls[0]["stream_mode"] == ["updates", "values"]
    assert isinstance(graph.calls[1]["input"], Command)
    assert graph.calls[1]["input"].resume == "approve-for-eval"


class MissingCustomDataGraph:
    checkpointer = object()

    async def ainvoke(self, input, config, stream_mode):
        return [("values", {"messages": [AIMessage(content="No custom data")]})]


@pytest.mark.asyncio
async def test_live_evaluation_missing_custom_data_raises_clear_error():
    with pytest.raises(EvaluationError, match="custom_data"):
        await _evaluate_live_cases(
            [_case("eval-001")],
            graph=MissingCustomDataGraph(),
            approval_response="approve",
        )


def test_live_report_missing_required_custom_data_fields_raises_clear_error():
    bad_result = _result()
    bad_result.pop("intent")

    with pytest.raises(EvaluationError, match="missing fields"):
        _build_evaluation_report(
            [_case("eval-001")],
            [bad_result],
            mode="live-llm",
            live_records=[{"interrupt_count": 0, "approval_resume": None}],
        )


def test_live_report_empty_case_set_raises_clear_error():
    with pytest.raises(EvaluationError, match="at least one case"):
        _build_evaluation_report([], [], mode="live-llm")


@pytest.mark.parametrize("bad_value", ["-1", "-5"])
def test_examples_limit_rejects_negative_values(bad_value: str):
    with pytest.raises(Exception, match="greater than or equal to 0"):
        _non_negative_int(bad_value)


def test_live_report_output_path_writes_utf8_json(tmp_path: Path):
    output_path = tmp_path / "live-report.json"
    report = _build_evaluation_report(
        [_case("eval-001")],
        [_result()],
        mode="live-llm",
        live_records=[{"interrupt_count": 0, "approval_resume": None}],
        examples_limit=5,
        output_path=output_path,
    )

    assert output_path.read_text(encoding="utf-8").startswith("{")
    assert report["qualitative_examples"][0]["draft_excerpt"] == "请在登录页点击重置密码。"
