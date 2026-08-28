"""命题逻辑模型枚举器的回归测试。"""

from logic_qa.model_enumerator import ModelEnumerationStatus, ModelEnumerator
from logic_qa.models import ImplicationRule, Literal


def test_enumerates_only_models_that_satisfy_facts_and_rules() -> None:
    """枚举结果必须排除违反已知事实或蕴含规则的赋值。"""
    result = ModelEnumerator().enumerate(
        facts=(Literal.parse("A"),),
        rules=(ImplicationRule(Literal.parse("A"), Literal.parse("B")),),
    )

    assert result.status is ModelEnumerationStatus.COMPLETE
    assert result.symbols == ("A", "B")
    assert len(result.models) == 1
    literals = [item.display() for item in result.models[0].display_literals()]
    assert literals == ["A", "B"]


def test_enumerates_models_that_leave_irrelevant_symbol_open() -> None:
    """未被规则约束的命题应保留其所有允许真值。"""
    result = ModelEnumerator().enumerate(
        facts=(),
        rules=(ImplicationRule(Literal.parse("A"), Literal.parse("B")),),
    )

    assert result.status is ModelEnumerationStatus.COMPLETE
    assert len(result.models) == 3
    assert any(model.satisfies(Literal.parse("!A")) for model in result.models)
    assert any(model.satisfies(Literal.parse("A")) for model in result.models)


def test_reports_unsatisfiable_conditions() -> None:
    """互相冲突的事实应导致不存在合法真值模型。"""
    result = ModelEnumerator().enumerate(
        facts=(Literal.parse("A"), Literal.parse("!A")),
        rules=(),
    )

    assert result.status is ModelEnumerationStatus.UNSATISFIABLE
    assert result.models == ()


def test_rejects_excessive_symbol_count_without_partial_result() -> None:
    """超过上限时必须安全拒绝，不得返回不完整的模型集合。"""
    facts = tuple(Literal.parse(f"P{index}") for index in range(13))
    result = ModelEnumerator(max_symbols=12).enumerate(facts=facts, rules=())

    assert result.status is ModelEnumerationStatus.SYMBOL_LIMIT_EXCEEDED
    assert len(result.symbols) == 13
    assert result.models == ()


def test_empty_constraints_have_one_empty_model() -> None:
    """空条件集在命题空间为空时仍有唯一的空模型。"""
    result = ModelEnumerator().enumerate(facts=(), rules=())

    assert result.status is ModelEnumerationStatus.COMPLETE
    assert result.symbols == ()
    assert len(result.models) == 1
    assert result.models[0].assignments == ()
