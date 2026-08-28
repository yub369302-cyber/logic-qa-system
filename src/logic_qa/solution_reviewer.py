"""结构化命题逻辑解法批改与首错诊断。

批改器只核验用户明确提交的规则编号、推理方向、前提和结论，不把自由文本
猜测为证明。题干的可推导性始终由同一个 InferenceEngine 作为基准裁决。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from logic_qa.engine import InferenceEngine
from logic_qa.models import (
    ImplicationRule,
    Literal,
    VerificationResult,
    VerificationStatus,
)


class ReasoningDirection(StrEnum):
    """用户声称使用原规则还是其等价逆否规则。"""

    FORWARD = "forward"
    CONTRAPOSITIVE = "contrapositive"


class ReviewStatus(StrEnum):
    """解法批改的总体状态。"""

    CORRECT = "correct"
    FIRST_ERROR = "first_error"
    INCOMPLETE = "incomplete"
    TARGET_NOT_PROVABLE = "target_not_provable"
    BLOCKED_INCONSISTENT = "blocked_inconsistent"


class DiagnosticCode(StrEnum):
    """第一版支持的结构化错因标签。"""

    INVALID_CONVERSE = "invalid_converse"
    WRONG_RULE_APPLICATION = "wrong_rule_application"
    PREMISE_NOT_ESTABLISHED = "premise_not_established"
    INVALID_RULE_REFERENCE = "invalid_rule_reference"
    MISSING_KEY_STEP = "missing_key_step"
    TARGET_NOT_PROVABLE = "target_not_provable"
    INCONSISTENT_CONDITIONS = "inconsistent_conditions"


@dataclass(frozen=True, slots=True)
class ReasoningStep:
    """用户提交的一步结构化推理。规则编号从 1 开始。"""

    rule_id: int
    direction: ReasoningDirection
    premise: Literal
    conclusion: Literal


@dataclass(frozen=True, slots=True)
class ReviewDiagnostic:
    """首个发现的问题及其可教学的解释。"""

    code: DiagnosticCode
    message: str
    knowledge_tags: tuple[str, ...]
    step_index: int | None = None


@dataclass(frozen=True, slots=True)
class SolutionReviewResult:
    """逐步批改结果、首错诊断和基准验证状态。"""

    status: ReviewStatus
    checked_step_count: int
    established_literals: tuple[Literal, ...]
    diagnostic: ReviewDiagnostic | None
    baseline: VerificationResult

    @property
    def is_correct(self) -> bool:
        """用户步骤是否完整且成功证明了目标。"""
        return self.status is ReviewStatus.CORRECT


class SolutionReviewer:
    """逐步核验结构化推理，并在第一个错误处停止。"""

    def __init__(self, engine: InferenceEngine | None = None) -> None:
        self._engine = engine or InferenceEngine()

    def review(
        self,
        facts: tuple[Literal, ...],
        rules: tuple[ImplicationRule, ...],
        target: Literal,
        steps: tuple[ReasoningStep, ...],
    ) -> SolutionReviewResult:
        """批改用户步骤并使用逻辑内核确认目标是否本可证明。"""
        baseline = self._engine.verify(facts=facts, rules=rules, query=target)
        if baseline.status is VerificationStatus.INCONSISTENT:
            return self._blocked_result(facts, baseline)
        if baseline.status is not VerificationStatus.PROVED:
            return self._target_not_provable_result(facts, baseline)

        established = set(facts)
        for index, step in enumerate(steps, start=1):
            diagnostic = self._validate_step(step, rules, established, index)
            if diagnostic is not None:
                return SolutionReviewResult(
                    status=ReviewStatus.FIRST_ERROR,
                    checked_step_count=index - 1,
                    established_literals=_sorted_literals(established),
                    diagnostic=diagnostic,
                    baseline=baseline,
                )
            established.add(step.conclusion)

        if target not in established:
            return SolutionReviewResult(
                status=ReviewStatus.INCOMPLETE,
                checked_step_count=len(steps),
                established_literals=_sorted_literals(established),
                diagnostic=ReviewDiagnostic(
                    code=DiagnosticCode.MISSING_KEY_STEP,
                    message=f"当前步骤尚未推出目标 {target.display()}。",
                    knowledge_tags=("条件推理", "证明完整性"),
                    step_index=None,
                ),
                baseline=baseline,
            )

        return SolutionReviewResult(
            status=ReviewStatus.CORRECT,
            checked_step_count=len(steps),
            established_literals=_sorted_literals(established),
            diagnostic=None,
            baseline=baseline,
        )

    def _validate_step(
        self,
        step: ReasoningStep,
        rules: tuple[ImplicationRule, ...],
        established: set[Literal],
        step_index: int,
    ) -> ReviewDiagnostic | None:
        if not 1 <= step.rule_id <= len(rules):
            return ReviewDiagnostic(
                code=DiagnosticCode.INVALID_RULE_REFERENCE,
                message=f"第 {step_index} 步引用的规则编号 {step.rule_id} 不存在。",
                knowledge_tags=("规则引用",),
                step_index=step_index,
            )

        rule = rules[step.rule_id - 1]
        expected_premise, expected_conclusion = _expected_literals(rule, step.direction)
        if step.premise != expected_premise or step.conclusion != expected_conclusion:
            if _is_direct_converse(step, rule):
                return ReviewDiagnostic(
                    code=DiagnosticCode.INVALID_CONVERSE,
                    message=(
                        f"第 {step_index} 步把 {rule.display()} 反向当作规则使用了。"
                        "原命题不能直接逆推；如需反向推理，应使用逆否命题。"
                    ),
                    knowledge_tags=("逆命题与逆否命题",),
                    step_index=step_index,
                )
            return ReviewDiagnostic(
                code=DiagnosticCode.WRONG_RULE_APPLICATION,
                message=(
                    f"第 {step_index} 步与规则 {step.rule_id} 的可用方向不一致。"
                    f"该方向应为 {expected_premise.display()} → "
                    f"{expected_conclusion.display()}。"
                ),
                knowledge_tags=("条件推理", "规则方向"),
                step_index=step_index,
            )

        if step.premise not in established:
            return ReviewDiagnostic(
                code=DiagnosticCode.PREMISE_NOT_ESTABLISHED,
                message=(
                    f"第 {step_index} 步的前提 {step.premise.display()} 尚未由题干"
                    "或前序正确步骤建立。"
                ),
                knowledge_tags=("前提核验",),
                step_index=step_index,
            )
        return None

    @staticmethod
    def _blocked_result(
        facts: tuple[Literal, ...],
        baseline: VerificationResult,
    ) -> SolutionReviewResult:
        return SolutionReviewResult(
            status=ReviewStatus.BLOCKED_INCONSISTENT,
            checked_step_count=0,
            established_literals=_sorted_literals(set(facts)),
            diagnostic=ReviewDiagnostic(
                code=DiagnosticCode.INCONSISTENT_CONDITIONS,
                message="题干条件存在矛盾，当前不能可靠地评价用户推理过程。",
                knowledge_tags=("条件一致性",),
                step_index=None,
            ),
            baseline=baseline,
        )

    @staticmethod
    def _target_not_provable_result(
        facts: tuple[Literal, ...],
        baseline: VerificationResult,
    ) -> SolutionReviewResult:
        return SolutionReviewResult(
            status=ReviewStatus.TARGET_NOT_PROVABLE,
            checked_step_count=0,
            established_literals=_sorted_literals(set(facts)),
            diagnostic=ReviewDiagnostic(
                code=DiagnosticCode.TARGET_NOT_PROVABLE,
                message=(
                    f"根据当前题干，目标 {baseline.query.display()} 不能被严格推出，"
                    "因此无法按“证明该目标”评判步骤。"
                ),
                knowledge_tags=("结论可推导性",),
                step_index=None,
            ),
            baseline=baseline,
        )


def _expected_literals(
    rule: ImplicationRule,
    direction: ReasoningDirection,
) -> tuple[Literal, Literal]:
    """返回指定方向下允许使用的前提和结论。"""
    if direction is ReasoningDirection.FORWARD:
        return rule.premise, rule.conclusion
    return rule.conclusion.opposite(), rule.premise.opposite()


def _is_direct_converse(step: ReasoningStep, rule: ImplicationRule) -> bool:
    """识别将 P→Q 直接写成 Q→P 的常见错误。"""
    return step.premise == rule.conclusion and step.conclusion == rule.premise


def _sorted_literals(literals: set[Literal]) -> tuple[Literal, ...]:
    """为 API 和测试提供稳定的文字输出顺序。"""
    return tuple(sorted(literals, key=lambda literal: literal.display()))
