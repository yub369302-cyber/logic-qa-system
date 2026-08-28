"""结构化解法批改与错因诊断的回归测试。"""

from logic_qa.models import ImplicationRule, Literal
from logic_qa.solution_reviewer import (
    DiagnosticCode,
    ReasoningDirection,
    ReasoningStep,
    ReviewStatus,
    SolutionReviewer,
)


def _chain_rules() -> tuple[ImplicationRule, ...]:
    return (
        ImplicationRule(Literal.parse("A"), Literal.parse("B")),
        ImplicationRule(Literal.parse("B"), Literal.parse("C")),
    )


def test_accepts_complete_correct_solution() -> None:
    """正确的链式前向推理应完整通过。"""
    result = SolutionReviewer().review(
        facts=(Literal.parse("A"),),
        rules=_chain_rules(),
        target=Literal.parse("C"),
        steps=(
            ReasoningStep(
                rule_id=1,
                direction=ReasoningDirection.FORWARD,
                premise=Literal.parse("A"),
                conclusion=Literal.parse("B"),
            ),
            ReasoningStep(
                rule_id=2,
                direction=ReasoningDirection.FORWARD,
                premise=Literal.parse("B"),
                conclusion=Literal.parse("C"),
            ),
        ),
    )

    assert result.status is ReviewStatus.CORRECT
    assert result.checked_step_count == 2
    assert result.diagnostic is None


def test_identifies_invalid_converse_as_first_error() -> None:
    """将 A→B 错写成 B→A 时应定位逆命题错误。"""
    result = SolutionReviewer().review(
        facts=(Literal.parse("A"), Literal.parse("B")),
        rules=_chain_rules(),
        target=Literal.parse("C"),
        steps=(
            ReasoningStep(
                rule_id=1,
                direction=ReasoningDirection.FORWARD,
                premise=Literal.parse("B"),
                conclusion=Literal.parse("A"),
            ),
        ),
    )

    assert result.status is ReviewStatus.FIRST_ERROR
    assert result.checked_step_count == 0
    assert result.diagnostic is not None
    assert result.diagnostic.code is DiagnosticCode.INVALID_CONVERSE
    assert result.diagnostic.step_index == 1


def test_identifies_premise_not_established() -> None:
    """引用正确规则但前提未成立时应标记前提核验错误。"""
    result = SolutionReviewer().review(
        facts=(Literal.parse("A"),),
        rules=_chain_rules(),
        target=Literal.parse("C"),
        steps=(
            ReasoningStep(
                rule_id=2,
                direction=ReasoningDirection.FORWARD,
                premise=Literal.parse("B"),
                conclusion=Literal.parse("C"),
            ),
        ),
    )

    assert result.status is ReviewStatus.FIRST_ERROR
    assert result.diagnostic is not None
    assert result.diagnostic.code is DiagnosticCode.PREMISE_NOT_ESTABLISHED


def test_accepts_contrapositive_reasoning() -> None:
    """用户明确使用逆否方向时，等价推理应被接受。"""
    result = SolutionReviewer().review(
        facts=(Literal.parse("!B"),),
        rules=(ImplicationRule(Literal.parse("A"), Literal.parse("B")),),
        target=Literal.parse("!A"),
        steps=(
            ReasoningStep(
                rule_id=1,
                direction=ReasoningDirection.CONTRAPOSITIVE,
                premise=Literal.parse("!B"),
                conclusion=Literal.parse("!A"),
            ),
        ),
    )

    assert result.status is ReviewStatus.CORRECT


def test_reports_incomplete_when_correct_steps_do_not_reach_target() -> None:
    """所有已提交步骤正确但未推出目标时应提示漏关键步骤。"""
    result = SolutionReviewer().review(
        facts=(Literal.parse("A"),),
        rules=_chain_rules(),
        target=Literal.parse("C"),
        steps=(
            ReasoningStep(
                rule_id=1,
                direction=ReasoningDirection.FORWARD,
                premise=Literal.parse("A"),
                conclusion=Literal.parse("B"),
            ),
        ),
    )

    assert result.status is ReviewStatus.INCOMPLETE
    assert result.diagnostic is not None
    assert result.diagnostic.code is DiagnosticCode.MISSING_KEY_STEP


def test_blocks_review_for_inconsistent_conditions() -> None:
    """题干自身矛盾时必须阻断对用户步骤的评判。"""
    result = SolutionReviewer().review(
        facts=(Literal.parse("A"), Literal.parse("!A")),
        rules=(),
        target=Literal.parse("C"),
        steps=(),
    )

    assert result.status is ReviewStatus.BLOCKED_INCONSISTENT
    assert result.diagnostic is not None
    assert result.diagnostic.code is DiagnosticCode.INCONSISTENT_CONDITIONS


def test_rejects_target_that_cannot_be_proved() -> None:
    """目标本身无法由题干推出时，不应把用户步骤误判为漏步。"""
    result = SolutionReviewer().review(
        facts=(Literal.parse("A"),),
        rules=(),
        target=Literal.parse("B"),
        steps=(),
    )

    assert result.status is ReviewStatus.TARGET_NOT_PROVABLE
    assert result.diagnostic is not None
    assert result.diagnostic.code is DiagnosticCode.TARGET_NOT_PROVABLE
