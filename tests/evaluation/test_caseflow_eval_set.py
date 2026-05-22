import json
import re
from collections import Counter
from pathlib import Path

INTENT_LABELS = {"咨询类", "投诉类", "异常类", "退款 / 升级类", "信息不全待补充类"}
PRIORITY_VALUES = {"low", "medium", "high"}
REQUIRED_FIELDS = [
    "id",
    "user_query",
    "expected_intent",
    "expected_priority",
    "expected_needs_human_approval",
    "expected_evidence_keywords",
    "draft_response_must_include",
    "evaluation_focus",
]


def _load_eval_cases() -> list[dict]:
    eval_path = Path("data/caseflow/eval_cases.json")
    return json.loads(eval_path.read_text(encoding="utf-8"))


def _hit_any(text: str, keywords: list[str]) -> bool:
    return any(keyword.lower() in text.lower() for keyword in keywords)


def _coverage_groups() -> dict[tuple[str, ...], str]:
    """Keyword groups matching what the evaluator code checks for.
    Maps keyword tuple -> coverage label.  The single 'supervisor/escalation'
    label replaces the two separate 'supervisor' and 'escalation' items.
    """
    return {
        ("退款", "退费", "重复扣费"): "refund",
        ("账单", "扣费", "重复扣费"): "duplicate fee",
        ("账单", "费用", "付款"): "billing",
        ("补偿", "赔偿", "减免"): "compensation",
        ("投诉", "不满", "差评"): "complaint",
        ("主管", "升级", "介入"): "supervisor/escalation",
    }


def test_caseflow_eval_set_has_minimum_case_count():
    """AC #1: At least 14 deterministic cases required."""
    cases = _load_eval_cases()
    assert len(cases) >= 14, f"Expected >=14 cases, got {len(cases)}"


def test_caseflow_eval_set_ids_are_unique_and_sequential():
    """AC #1: Unique eval-### ids with no missing required fields."""
    cases = _load_eval_cases()

    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"
    assert all(re.fullmatch(r"eval-\d{3}", case_id) for case_id in ids)

    nums = sorted(int(case_id.removeprefix("eval-")) for case_id in ids)
    expected = list(range(1, len(cases) + 1))
    assert nums == expected, f"IDs not sequential: {nums} vs expected {expected}"

    for case in cases:
        for field in REQUIRED_FIELDS:
            assert field in case, f"Case {case['id']} missing field '{field}'"
        assert case["user_query"].strip(), f"Case {case['id']} has empty user_query"
        assert case["expected_intent"] in INTENT_LABELS
        assert case["expected_priority"] in PRIORITY_VALUES
        assert isinstance(case["expected_needs_human_approval"], bool)
        assert isinstance(case["expected_evidence_keywords"], list)
        assert case["expected_evidence_keywords"], f"Case {case['id']} has empty evidence keywords"
        assert isinstance(case["draft_response_must_include"], list)
        assert case["draft_response_must_include"], f"Case {case['id']} has empty draft_response_must_include"
        assert isinstance(case["evaluation_focus"], list)
        assert case["evaluation_focus"], f"Case {case['id']} has empty evaluation_focus"


def test_caseflow_eval_set_intent_coverage():
    """AC #2: Every supported Chinese intent label appears at least twice."""
    cases = _load_eval_cases()

    intent_counts = Counter(case["expected_intent"] for case in cases)
    missing_or_sparse = [
        label for label in INTENT_LABELS if intent_counts.get(label, 0) < 2
    ]
    assert not missing_or_sparse, (
        f"Intent labels with <2 cases: {dict(intent_counts)}; "
        f"missing/sparse: {missing_or_sparse}"
    )


def test_caseflow_eval_set_high_risk_guardrail_coverage():
    """AC #3: High-risk guardrail coverage for refund, billing, compensation,
    complaint, supervisor/escalation, and approval paths."""
    from agents.caseflow_agent import analyze_case

    cases = _load_eval_cases()

    covered_focuses: set[str] = set()
    for case in cases:
        user_query_text = case.get("user_query", "").lower()
        focus_text = " ".join(case.get("evaluation_focus", [])).lower()
        for keyword_group, focus_name in _coverage_groups().items():
            if any(kw in user_query_text or kw in focus_text for kw in keyword_group):
                covered_focuses.add(focus_name)
        result = analyze_case(case["user_query"], f"coverage-{case['id']}", "cust-001")
        if result["needs_human_approval"]:
            covered_focuses.add("approval")
        if result["needs_human_approval"] and result["priority"] == "high":
            covered_focuses.add("high-priority approval")

    required = set(_coverage_groups().values()) | {"approval", "high-priority approval"}
    missing = required - covered_focuses
    assert not missing, f"High-risk guardrail gaps: {sorted(missing)}"


def test_caseflow_eval_set_passes_current_requirements():
    """AC #1 (legacy): Original coverage test retained for backward compatibility."""
    cases = _load_eval_cases()

    assert len(cases) >= 8
    assert INTENT_LABELS.issubset({case["expected_intent"] for case in cases})
    assert any(case["expected_needs_human_approval"] for case in cases)
    assert all(case["evaluation_focus"] for case in cases)
    assert all(case["expected_evidence_keywords"] for case in cases)
    assert all(case["draft_response_must_include"] for case in cases)


def test_caseflow_deterministic_eval_metrics_are_reported():
    """AC #4/5: Deterministic metrics pass and priority_accuracy is enforced."""
    from scripts.evaluate_caseflow import evaluate_cases

    metrics = evaluate_cases(live_llm=False)

    assert metrics["mode"] == "deterministic"
    assert metrics["total_cases"] >= 14, f"Expected >=14 cases, got {metrics['total_cases']}"
    assert metrics["intent_accuracy"] >= 0.75
    assert metrics["approval_accuracy"] == 1.0
    assert metrics["evidence_hit_rate"] >= 0.75
    assert "draft_groundedness_rate" in metrics
    assert metrics["priority_accuracy"] >= 0.75


def test_caseflow_deterministic_eval_matches_each_case_expectation():
    """Each deterministic case must match its own expected output contract."""
    from agents.caseflow_agent import analyze_case

    misses = []
    for case in _load_eval_cases():
        result = analyze_case(case["user_query"], f"eval-{case['id']}", "cust-001")
        evidence_text = "\n".join(result.get("evidence", []))
        draft_response = result.get("draft_response", "")
        case_misses = []
        if result["intent"] != case["expected_intent"]:
            case_misses.append(f"intent {result['intent']} != {case['expected_intent']}")
        if result["priority"] != case["expected_priority"]:
            case_misses.append(f"priority {result['priority']} != {case['expected_priority']}")
        if result["needs_human_approval"] != case["expected_needs_human_approval"]:
            case_misses.append(
                "approval "
                f"{result['needs_human_approval']} != {case['expected_needs_human_approval']}"
            )
        if not _hit_any(evidence_text, case["expected_evidence_keywords"]):
            case_misses.append("evidence keywords miss")
        if not all(
            keyword.lower() in draft_response.lower()
            for keyword in case["draft_response_must_include"]
        ):
            case_misses.append("draft keywords miss")
        if case_misses:
            misses.append(f"{case['id']}: {'; '.join(case_misses)}")

    assert not misses, "\n".join(misses)
