"""命题逻辑推理内核的单元测试。"""

from logic_qa.engine import InferenceEngine
from logic_qa.models import ImplicationRule, Literal, VerificationStatus


def test_derives_a_multi_step_conclusion() -> None:
    """系统应能沿多条蕴含规则完成前向推理。"""
    result = InferenceEngine().verify(
        facts=[Literal.parse("A")],
        rules=[
            ImplicationRule(Literal.parse("A"), Literal.parse("B")),
            ImplicationRule(Literal.parse("B"), Literal.parse("C")),
        ],
        query=Literal.parse("C"),
    )

    assert result.status is VerificationStatus.PROVED
    assert [step.derived.display() for step in result.proof_steps] == ["A", "B", "C"]


def test_derives_a_contrapositive_conclusion() -> None:
    """系统只使用逻辑等价的逆否规则，不错误使用逆命题。"""
    result = InferenceEngine().verify(
        facts=[Literal.parse("!B")],
        rules=[ImplicationRule(Literal.parse("A"), Literal.parse("B"))],
        query=Literal.parse("!A"),
    )

    assert result.status is VerificationStatus.PROVED
    assert result.proof_steps[-1].reason == "逆否推理"
    assert result.proof_steps[-1].derived.display() == "¬A"


def test_does_not_infer_the_converse() -> None:
    """B 为真不能仅由 A→B 反向推出 A 为真。"""
    result = InferenceEngine().verify(
        facts=[Literal.parse("B")],
        rules=[ImplicationRule(Literal.parse("A"), Literal.parse("B"))],
        query=Literal.parse("A"),
    )

    assert result.status is VerificationStatus.UNKNOWN
    assert result.proof_steps == ()


def test_reports_disproved_when_opposite_is_proved() -> None:
    """当查询的否定可被推出时，系统应明确返回反证状态。"""
    result = InferenceEngine().verify(
        facts=[Literal.parse("A")],
        rules=[ImplicationRule(Literal.parse("A"), Literal.parse("!B"))],
        query=Literal.parse("B"),
    )

    assert result.status is VerificationStatus.DISPROVED
    assert result.proof_steps[-1].derived.display() == "¬B"


def test_reports_inconsistent_knowledge() -> None:
    """当正反结论均可推出时，系统不能静默返回任意一侧。"""
    result = InferenceEngine().verify(
        facts=[Literal.parse("A"), Literal.parse("C")],
        rules=[
            ImplicationRule(Literal.parse("A"), Literal.parse("B")),
            ImplicationRule(Literal.parse("C"), Literal.parse("!B")),
        ],
        query=Literal.parse("B"),
    )

    assert result.status is VerificationStatus.INCONSISTENT
    assert result.conflict is not None
    assert result.is_consistent is False


def test_reports_global_conflict_for_unrelated_query() -> None:
    """与查询无关的条件矛盾也必须被上报，不能伪装成未知。"""
    result = InferenceEngine().verify(
        facts=[Literal.parse("A"), Literal.parse("!A")],
        rules=[],
        query=Literal.parse("C"),
    )

    assert result.status is VerificationStatus.INCONSISTENT
    assert result.conflict is not None
    assert {item.display() for item in result.conflict} == {"A", "¬A"}
    derived = {step.derived.display() for step in result.proof_steps}
    assert {"A", "¬A"}.issubset(derived)


def test_reports_conflict_derived_only_from_rules() -> None:
    """矛盾由规则推导产生时同样需要被检测并定位来源。"""
    result = InferenceEngine().verify(
        facts=[Literal.parse("A"), Literal.parse("B")],
        rules=[
            ImplicationRule(Literal.parse("A"), Literal.parse("C")),
            ImplicationRule(Literal.parse("B"), Literal.parse("!C")),
        ],
        query=Literal.parse("C"),
    )

    assert result.status is VerificationStatus.INCONSISTENT
    assert result.conflict is not None
    assert {item.display() for item in result.conflict} == {"C", "¬C"}


def test_circular_rules_terminate() -> None:
    """循环规则不应导致推理死循环，且查询仍可被证明。"""
    result = InferenceEngine().verify(
        facts=[Literal.parse("A")],
        rules=[
            ImplicationRule(Literal.parse("A"), Literal.parse("B")),
            ImplicationRule(Literal.parse("B"), Literal.parse("A")),
        ],
        query=Literal.parse("B"),
    )

    assert result.status is VerificationStatus.PROVED
    assert result.is_consistent
