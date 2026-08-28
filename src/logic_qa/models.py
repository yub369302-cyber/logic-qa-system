"""领域模型：使用显式结构承载可验证的逻辑条件。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class VerificationStatus(StrEnum):
    """查询结论的验证状态。"""

    PROVED = "proved"
    DISPROVED = "disproved"
    UNKNOWN = "unknown"
    INCONSISTENT = "inconsistent"


@dataclass(frozen=True, slots=True)
class Literal:
    """一个原子命题或其否定，例如 A 或 ¬A。"""

    symbol: str
    negated: bool = False

    @classmethod
    def parse(cls, value: str) -> Literal:
        """从简洁文本表示解析文字。

        支持 `A`、`!A`、`¬A` 和 `not A` 四种表示；命题名称会被去除首尾空格。
        """
        normalized = value.strip()
        if not normalized:
            raise ValueError("命题不能为空")

        for prefix in ("!", "¬"):
            if normalized.startswith(prefix):
                return cls(symbol=cls._validate_symbol(normalized[1:]), negated=True)

        if normalized.lower().startswith("not "):
            return cls(symbol=cls._validate_symbol(normalized[4:]), negated=True)

        return cls(symbol=cls._validate_symbol(normalized))

    @staticmethod
    def _validate_symbol(value: str) -> str:
        symbol = value.strip()
        if not symbol:
            raise ValueError("否定符号后必须包含命题名称")
        return symbol

    def opposite(self) -> Literal:
        """返回当前文字的否定。"""
        return Literal(symbol=self.symbol, negated=not self.negated)

    def display(self) -> str:
        """返回面向用户展示的逻辑文字。"""
        return f"¬{self.symbol}" if self.negated else self.symbol


@dataclass(frozen=True, slots=True)
class ImplicationRule:
    """单前提蕴含规则：premise 为真时，conclusion 必为真。"""

    premise: Literal
    conclusion: Literal
    source_text: str | None = None

    def display(self) -> str:
        """返回规则的逻辑表达。"""
        return f"{self.premise.display()} → {self.conclusion.display()}"


@dataclass(frozen=True, slots=True)
class ProofStep:
    """导出一个文字时所使用的依据。"""

    derived: Literal
    reason: str
    source_rule: ImplicationRule | None = None
    dependencies: tuple[Literal, ...] = ()


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """对单一查询文字的可追溯验证结果。"""

    query: Literal
    status: VerificationStatus
    proof_steps: tuple[ProofStep, ...]
    known_literals: tuple[Literal, ...]
    conflict: tuple[Literal, Literal] | None = None

    @property
    def is_proved(self) -> bool:
        """查询是否已被严格证明。"""
        return self.status is VerificationStatus.PROVED

    @property
    def is_consistent(self) -> bool:
        """题干事实与规则是否彼此自洽。"""
        return self.conflict is None
