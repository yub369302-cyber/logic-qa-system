"""受控中文条件解析器的回归测试。"""

import pytest

from logic_qa.chinese_parser import (
    ChineseParseError,
    ConfirmedChineseSentence,
    ControlledChineseParser,
)
from logic_qa.models import ImplicationRule, Literal


@pytest.mark.parametrize(
    ("sentence", "premise", "conclusion"),
    [
        ("如果甲参加，那么乙通过。", "甲参加", "乙通过"),
        ("若甲参加，则乙通过。", "甲参加", "乙通过"),
        ("只要甲参加，就乙通过。", "甲参加", "乙通过"),
        ("如果甲参加，那么乙不通过。", "甲参加", "¬乙通过"),
    ],
)
def test_parses_supported_sufficient_condition_forms(
    sentence: str,
    premise: str,
    conclusion: str,
) -> None:
    """三种充分条件句式与否定结论应转换为同一逻辑规则。"""
    result = ControlledChineseParser().parse(sentence)

    assert result.facts == ()
    assert len(result.rules) == 1
    assert result.rules[0].premise.display() == premise
    assert result.rules[0].conclusion.display() == conclusion
    assert result.rules[0].source_text == sentence.removesuffix("。")


def test_parses_necessary_condition_in_reverse_direction() -> None:
    """“只有 A，B 才 C”必须转换为“B C → A”。"""
    result = ControlledChineseParser().parse("只有甲参加，乙才通过。")

    rule = result.rules[0]
    assert rule.premise.display() == "乙通过"
    assert rule.conclusion.display() == "甲参加"


def test_parses_positive_and_negative_facts() -> None:
    """普通事实及单个“不”否定事实应被识别。"""
    result = ControlledChineseParser().parse("甲参加。乙不通过。")

    assert [fact.display() for fact in result.facts] == ["甲参加", "¬乙通过"]
    assert result.rules == ()


def test_preserves_multiple_source_sentences() -> None:
    """多条件输入应保留每条原始条件，供后续 API 回显。"""
    result = ControlledChineseParser().parse("甲参加；如果甲参加，那么乙通过。")

    assert result.source_sentences == ("甲参加", "如果甲参加，那么乙通过")
    assert result.rules[0].source_text == "如果甲参加，那么乙通过"


def test_parses_explicit_biconditional_into_two_rules() -> None:
    """语法完整的当且仅当应安全展开为两个方向的蕴含规则。"""
    result = ControlledChineseParser().parse("当且仅当甲参加，乙通过。")

    assert [rule.display() for rule in result.rules] == [
        "甲参加 → 乙通过",
        "乙通过 → 甲参加",
    ]


def test_complex_semantics_are_returned_for_human_confirmation() -> None:
    """除非、析取、合取和量词不能自动推理，必须返回确认请求。"""
    analysis = ControlledChineseParser().analyze(
        "甲参加。除非乙通过，否则丙入选。丁参加或者戊参加。至少两人通过。"
    )

    assert [fact.display() for fact in analysis.parsed.facts] == ["甲参加"]
    assert [request.source_sentence for request in analysis.confirmation_requests] == [
        "除非乙通过，否则丙入选",
        "丁参加或者戊参加",
        "至少两人通过",
    ]
    assert analysis.confirmation_requests[0].codes == ("unless",)
    assert analysis.confirmation_requests[1].codes == ("disjunction",)
    assert analysis.confirmation_requests[2].codes == ("quantifier",)


def test_parse_with_confirmations_only_uses_explicit_structured_translation() -> None:
    """复杂原句只能通过人工确认给出的结构化规则进入逻辑内核。"""
    parser = ControlledChineseParser()
    result = parser.parse_with_confirmations(
        "甲参加。除非乙通过，否则丙入选。",
        (
            ConfirmedChineseSentence(
                source_sentence="除非乙通过，否则丙入选",
                rules=(
                    ImplicationRule(
                        premise=Literal.parse("!乙通过"),
                        conclusion=Literal.parse("丙入选"),
                    ),
                ),
            ),
        ),
    )

    assert [fact.display() for fact in result.facts] == ["甲参加"]
    assert result.rules[0].display() == "¬乙通过 → 丙入选"
    assert result.rules[0].source_text == "除非乙通过，否则丙入选（人工确认）"


def test_parse_with_confirmations_rejects_missing_or_unrelated_confirmation() -> None:
    """确认不完整或引用非待确认原句时不得形成可验证输入。"""
    parser = ControlledChineseParser()

    with pytest.raises(ChineseParseError, match="尚未人工确认"):
        parser.parse_with_confirmations("甲参加或者乙参加。", ())

    with pytest.raises(ChineseParseError, match="不是当前题干"):
        parser.parse_with_confirmations(
            "甲参加。",
            (
                ConfirmedChineseSentence(
                    source_sentence="甲参加",
                    facts=(Literal.parse("甲参加"),),
                ),
            ),
        )


@pytest.mark.parametrize(
    "sentence",
    [
        "如果甲参加，乙通过。",
        "除非甲参加，否则乙通过。",
        "甲参加且乙通过。",
        "甲不是乙。",
        "只有甲参加，乙通过。",
        "甲 不参加。",
        "。；\n",
    ],
)
def test_rejects_unsupported_or_ambiguous_forms(sentence: str) -> None:
    """不在受控语法范围内的表达必须明确拒绝，不能静默猜测。"""
    with pytest.raises(ChineseParseError):
        ControlledChineseParser().parse(sentence)
