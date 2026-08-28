"""小规模命题逻辑真值模型枚举器。

该模块用于选择题验证：它枚举满足所有事实和蕴含规则的模型，而不是仅查看
前向推理已经导出的结论。由于枚举规模指数增长，默认严格限制为 12 个命题。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from logic_qa.models import ImplicationRule, Literal


class ModelEnumerationStatus(StrEnum):
    """模型枚举的完成状态。"""

    COMPLETE = "complete"
    UNSATISFIABLE = "unsatisfiable"
    SYMBOL_LIMIT_EXCEEDED = "symbol_limit_exceeded"


@dataclass(frozen=True, slots=True)
class TruthModel:
    """满足题干约束的一组原子命题真值赋值。"""

    assignments: tuple[tuple[str, bool], ...]

    def satisfies(self, literal: Literal) -> bool:
        """判断当前模型是否满足指定正负文字。"""
        values = dict(self.assignments)
        value = values[literal.symbol]
        return not value if literal.negated else value

    def display_literals(self) -> tuple[Literal, ...]:
        """将模型转换为用于 API 和讲解展示的正负文字列表。"""
        return tuple(
            Literal(symbol=symbol, negated=not value)
            for symbol, value in self.assignments
        )


@dataclass(frozen=True, slots=True)
class ModelEnumerationResult:
    """模型枚举的状态和所有合法模型。"""

    status: ModelEnumerationStatus
    symbols: tuple[str, ...]
    models: tuple[TruthModel, ...]

    @property
    def is_complete(self) -> bool:
        """是否在限制内完成了所有模型搜索。"""
        return self.status is ModelEnumerationStatus.COMPLETE


class ModelEnumerator:
    """通过真值表枚举满足事实与单前提蕴含规则的全部模型。"""

    def __init__(self, max_symbols: int = 12) -> None:
        if max_symbols < 1:
            raise ValueError("max_symbols 必须至少为 1")
        self._max_symbols = max_symbols

    def enumerate(
        self,
        facts: tuple[Literal, ...],
        rules: tuple[ImplicationRule, ...],
    ) -> ModelEnumerationResult:
        """返回所有满足题干条件的模型或无法安全枚举的状态。"""
        symbols = self._collect_symbols(facts, rules)
        if len(symbols) > self._max_symbols:
            return ModelEnumerationResult(
                status=ModelEnumerationStatus.SYMBOL_LIMIT_EXCEEDED,
                symbols=symbols,
                models=(),
            )

        models = tuple(
            model
            for model in self._candidate_models(symbols)
            if self._satisfies_all(model, facts, rules)
        )
        status = (
            ModelEnumerationStatus.COMPLETE
            if models
            else ModelEnumerationStatus.UNSATISFIABLE
        )
        return ModelEnumerationResult(status=status, symbols=symbols, models=models)

    @staticmethod
    def _collect_symbols(
        facts: tuple[Literal, ...],
        rules: tuple[ImplicationRule, ...],
    ) -> tuple[str, ...]:
        symbols = {fact.symbol for fact in facts}
        for rule in rules:
            symbols.add(rule.premise.symbol)
            symbols.add(rule.conclusion.symbol)
        return tuple(sorted(symbols))

    @staticmethod
    def _candidate_models(symbols: tuple[str, ...]) -> tuple[TruthModel, ...]:
        return tuple(
            TruthModel(assignments=tuple(zip(symbols, values, strict=True)))
            for values in product((False, True), repeat=len(symbols))
        )

    @staticmethod
    def _satisfies_all(
        model: TruthModel,
        facts: tuple[Literal, ...],
        rules: tuple[ImplicationRule, ...],
    ) -> bool:
        if any(not model.satisfies(fact) for fact in facts):
            return False
        return all(
            not model.satisfies(rule.premise) or model.satisfies(rule.conclusion)
            for rule in rules
        )
