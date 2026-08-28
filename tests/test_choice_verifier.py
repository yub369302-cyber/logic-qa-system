"""命题逻辑选择题验证器的回归测试。"""

import pytest

from logic_qa.choice_verifier import (
    ChoiceQuestionType,
    ChoiceVerificationStatus,
    ChoiceVerifier,
)
from logic_qa.model_enumerator import ModelEnumerator
from logic_qa.models import ImplicationRule, Literal


@pytest.mark.parametrize(
    ("question_type", "option", "expected_status"),
    [
        (ChoiceQuestionType.MUST_BE_TRUE, "B", ChoiceVerificationStatus.VERIFIED),
        (ChoiceQuestionType.MAY_BE_TRUE, "B", ChoiceVerificationStatus.VERIFIED),
        (
            ChoiceQuestionType.CANNOT_BE_TRUE,
            "!B",
            ChoiceVerificationStatus.VERIFIED,
        ),
        (
            ChoiceQuestionType.CANNOT_BE_INFERRED,
            "!B",
            ChoiceVerificationStatus.VERIFIED,
        ),
    ],
)
def test_verifies_each_question_type_against_all_models(
    question_type: ChoiceQuestionType,
    option: str,
    expected_status: ChoiceVerificationStatus,
) -> None:
    """四类设问应按各自语义在完整模型集合上判定。"""
    result = ChoiceVerifier().verify(
        facts=(Literal.parse("A"),),
        rules=(ImplicationRule(Literal.parse("A"), Literal.parse("B")),),
        option=Literal.parse(option),
        question_type=question_type,
    )

    assert result.status is expected_status
    assert result.model_count == 1


def test_must_be_true_returns_counterexample_when_option_is_not_necessary() -> None:
    """“一定为真”失败时必须返回违反选项的反例。"""
    result = ChoiceVerifier().verify(
        facts=(),
        rules=(ImplicationRule(Literal.parse("A"), Literal.parse("B")),),
        option=Literal.parse("B"),
        question_type=ChoiceQuestionType.MUST_BE_TRUE,
    )

    assert result.status is ChoiceVerificationStatus.NOT_VERIFIED
    assert result.witness_model is not None
    assert result.witness_model.satisfies(Literal.parse("!B"))


def test_may_be_true_returns_no_witness_when_option_is_impossible() -> None:
    """“可能为真”失败时不应伪造正例。"""
    result = ChoiceVerifier().verify(
        facts=(Literal.parse("A"),),
        rules=(ImplicationRule(Literal.parse("A"), Literal.parse("B")),),
        option=Literal.parse("!B"),
        question_type=ChoiceQuestionType.MAY_BE_TRUE,
    )

    assert result.status is ChoiceVerificationStatus.NOT_VERIFIED
    assert result.witness_model is None


def test_cannot_be_true_returns_positive_witness_when_option_is_possible() -> None:
    """“不可能为真”失败时必须返回满足选项的正例。"""
    result = ChoiceVerifier().verify(
        facts=(),
        rules=(ImplicationRule(Literal.parse("A"), Literal.parse("B")),),
        option=Literal.parse("B"),
        question_type=ChoiceQuestionType.CANNOT_BE_TRUE,
    )

    assert result.status is ChoiceVerificationStatus.NOT_VERIFIED
    assert result.witness_model is not None
    assert result.witness_model.satisfies(Literal.parse("B"))


def test_cannot_be_inferred_fails_without_a_counterexample() -> None:
    """所有模型都满足选项时，“无法推出”不得通过。"""
    result = ChoiceVerifier().verify(
        facts=(Literal.parse("A"),),
        rules=(ImplicationRule(Literal.parse("A"), Literal.parse("B")),),
        option=Literal.parse("B"),
        question_type=ChoiceQuestionType.CANNOT_BE_INFERRED,
    )

    assert result.status is ChoiceVerificationStatus.NOT_VERIFIED
    assert result.witness_model is None


@pytest.mark.parametrize(
    ("facts", "max_symbols", "expected_status"),
    [
        ((Literal.parse("A"), Literal.parse("!A")), 12, "blocked_unsatisfiable"),
        (
            tuple(Literal.parse(f"P{index}") for index in range(13)),
            12,
            "blocked_symbol_limit",
        ),
    ],
)
def test_blocks_verification_when_models_are_not_available(
    facts: tuple[Literal, ...],
    max_symbols: int,
    expected_status: str,
) -> None:
    """无解或超出规模上限时，选择题结论必须被阻断。"""
    verifier = ChoiceVerifier(enumerator=ModelEnumerator(max_symbols=max_symbols))
    result = verifier.verify(
        facts=facts,
        rules=(),
        option=Literal.parse("B"),
        question_type=ChoiceQuestionType.MUST_BE_TRUE,
    )

    assert result.status.value == expected_status
    assert result.witness_model is None
