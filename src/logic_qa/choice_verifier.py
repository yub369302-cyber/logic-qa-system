"""基于完整真值模型的命题逻辑选择题验证器。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from logic_qa.model_enumerator import (
    ModelEnumerationResult,
    ModelEnumerationStatus,
    ModelEnumerator,
    TruthModel,
)
from logic_qa.models import ImplicationRule, Literal


class ChoiceQuestionType(StrEnum):
    """第一版支持的选择题设问类型。"""

    MUST_BE_TRUE = "must_be_true"
    MAY_BE_TRUE = "may_be_true"
    CANNOT_BE_TRUE = "cannot_be_true"
    CANNOT_BE_INFERRED = "cannot_be_inferred"


class ChoiceVerificationStatus(StrEnum):
    """单个选项的验证状态。"""

    VERIFIED = "verified"
    NOT_VERIFIED = "not_verified"
    BLOCKED_UNSATISFIABLE = "blocked_unsatisfiable"
    BLOCKED_SYMBOL_LIMIT = "blocked_symbol_limit"


@dataclass(frozen=True, slots=True)
class ChoiceVerificationResult:
    """一个选项的验证结论与可展示的正例或反例。"""

    question_type: ChoiceQuestionType
    option: Literal
    status: ChoiceVerificationStatus
    witness_model: TruthModel | None
    model_count: int
    enumeration_status: ModelEnumerationStatus

    @property
    def is_verified(self) -> bool:
        """当前选项是否满足题目设问。"""
        return self.status is ChoiceVerificationStatus.VERIFIED


class ChoiceVerifier:
    """在枚举完成的真值模型集合上验证单个选项。"""

    def __init__(self, enumerator: ModelEnumerator | None = None) -> None:
        self._enumerator = enumerator or ModelEnumerator()

    def verify(
        self,
        facts: tuple[Literal, ...],
        rules: tuple[ImplicationRule, ...],
        option: Literal,
        question_type: ChoiceQuestionType,
    ) -> ChoiceVerificationResult:
        """验证一个选项是否符合指定设问，并提供关键实例。"""
        enumeration = self._enumerator.enumerate(facts=facts, rules=rules)
        blocked = self._blocked_result(question_type, option, enumeration)
        if blocked is not None:
            return blocked

        models = enumeration.models
        match question_type:
            case ChoiceQuestionType.MUST_BE_TRUE:
                witness = self._first_model(models, option, expected=False)
                status = (
                    ChoiceVerificationStatus.VERIFIED
                    if witness is None
                    else ChoiceVerificationStatus.NOT_VERIFIED
                )
            case ChoiceQuestionType.MAY_BE_TRUE:
                witness = self._first_model(models, option, expected=True)
                status = (
                    ChoiceVerificationStatus.VERIFIED
                    if witness is not None
                    else ChoiceVerificationStatus.NOT_VERIFIED
                )
            case ChoiceQuestionType.CANNOT_BE_TRUE:
                witness = self._first_model(models, option, expected=True)
                status = (
                    ChoiceVerificationStatus.VERIFIED
                    if witness is None
                    else ChoiceVerificationStatus.NOT_VERIFIED
                )
            case ChoiceQuestionType.CANNOT_BE_INFERRED:
                witness = self._first_model(models, option, expected=False)
                status = (
                    ChoiceVerificationStatus.VERIFIED
                    if witness is not None
                    else ChoiceVerificationStatus.NOT_VERIFIED
                )

        return ChoiceVerificationResult(
            question_type=question_type,
            option=option,
            status=status,
            witness_model=witness,
            model_count=len(models),
            enumeration_status=enumeration.status,
        )

    @staticmethod
    def _first_model(
        models: tuple[TruthModel, ...],
        option: Literal,
        expected: bool,
    ) -> TruthModel | None:
        return next(
            (model for model in models if model.satisfies(option) is expected),
            None,
        )

    @staticmethod
    def _blocked_result(
        question_type: ChoiceQuestionType,
        option: Literal,
        enumeration: ModelEnumerationResult,
    ) -> ChoiceVerificationResult | None:
        if enumeration.status is ModelEnumerationStatus.UNSATISFIABLE:
            return ChoiceVerificationResult(
                question_type=question_type,
                option=option,
                status=ChoiceVerificationStatus.BLOCKED_UNSATISFIABLE,
                witness_model=None,
                model_count=0,
                enumeration_status=enumeration.status,
            )
        if enumeration.status is ModelEnumerationStatus.SYMBOL_LIMIT_EXCEEDED:
            return ChoiceVerificationResult(
                question_type=question_type,
                option=option,
                status=ChoiceVerificationStatus.BLOCKED_SYMBOL_LIMIT,
                witness_model=None,
                model_count=0,
                enumeration_status=enumeration.status,
            )
        return None
