"""小规模排序题的完全枚举求解器。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import permutations


class OrderingConstraintType(StrEnum):
    """第一版排序题支持的约束类型。"""

    BEFORE = "before"
    ADJACENT = "adjacent"
    NOT_ADJACENT = "not_adjacent"
    FIXED_POSITION = "fixed_position"


class OrderingSolveStatus(StrEnum):
    """排序求解的完成状态。"""

    COMPLETE = "complete"
    UNSATISFIABLE = "unsatisfiable"
    ITEM_LIMIT_EXCEEDED = "item_limit_exceeded"


@dataclass(frozen=True, slots=True)
class OrderingConstraint:
    """一个两两关系或固定位置约束。"""

    constraint_type: OrderingConstraintType
    item: str
    other_item: str | None = None
    position: int | None = None


@dataclass(frozen=True, slots=True)
class OrderingSolveResult:
    """排序题的完整解空间统计和有限展示样本。"""

    status: OrderingSolveStatus
    solution_count: int
    sample_solutions: tuple[tuple[str, ...], ...]

    @property
    def is_complete(self) -> bool:
        """是否完成了全部排列的确定性检查。"""
        return self.status is OrderingSolveStatus.COMPLETE


class OrderingSolver:
    """使用完全排列枚举求解不超过指定对象数量的排序题。"""

    def __init__(self, max_items: int = 8, sample_limit: int = 5) -> None:
        if max_items < 1:
            raise ValueError("max_items 必须至少为 1")
        if sample_limit < 1:
            raise ValueError("sample_limit 必须至少为 1")
        self._max_items = max_items
        self._sample_limit = sample_limit

    def solve(
        self,
        items: tuple[str, ...],
        constraints: tuple[OrderingConstraint, ...],
    ) -> OrderingSolveResult:
        """验证输入并枚举所有满足约束的排列。"""
        self._validate_items(items)
        self._validate_constraints(items, constraints)
        if len(items) > self._max_items:
            return OrderingSolveResult(
                status=OrderingSolveStatus.ITEM_LIMIT_EXCEEDED,
                solution_count=0,
                sample_solutions=(),
            )

        solutions = tuple(
            ordering
            for ordering in permutations(items)
            if self._satisfies_all(ordering, constraints)
        )
        if not solutions:
            return OrderingSolveResult(
                status=OrderingSolveStatus.UNSATISFIABLE,
                solution_count=0,
                sample_solutions=(),
            )
        return OrderingSolveResult(
            status=OrderingSolveStatus.COMPLETE,
            solution_count=len(solutions),
            sample_solutions=solutions[: self._sample_limit],
        )

    @staticmethod
    def _validate_items(items: tuple[str, ...]) -> None:
        if not items:
            raise ValueError("排序对象不能为空")
        if any(not item.strip() for item in items):
            raise ValueError("排序对象不能包含空文本")
        if len(set(items)) != len(items):
            raise ValueError("排序对象不能重复")

    @staticmethod
    def _validate_constraints(
        items: tuple[str, ...],
        constraints: tuple[OrderingConstraint, ...],
    ) -> None:
        item_set = set(items)
        for constraint in constraints:
            if constraint.item not in item_set:
                raise ValueError(f"约束对象不存在：“{constraint.item}”")
            if constraint.constraint_type is OrderingConstraintType.FIXED_POSITION:
                if constraint.other_item is not None or constraint.position is None:
                    raise ValueError("固定位置约束必须仅包含 position")
                if not 1 <= constraint.position <= len(items):
                    raise ValueError("固定位置必须位于对象数量范围内")
                continue

            if constraint.other_item is None or constraint.position is not None:
                raise ValueError("两两关系约束必须仅包含 other_item")
            if constraint.other_item not in item_set:
                raise ValueError(f"约束对象不存在：“{constraint.other_item}”")
            if constraint.item == constraint.other_item:
                raise ValueError("排序关系不能引用同一对象")

    @staticmethod
    def _satisfies_all(
        ordering: tuple[str, ...],
        constraints: tuple[OrderingConstraint, ...],
    ) -> bool:
        positions = {item: index + 1 for index, item in enumerate(ordering)}
        for constraint in constraints:
            match constraint.constraint_type:
                case OrderingConstraintType.BEFORE:
                    if positions[constraint.item] >= positions[constraint.other_item]:
                        return False
                case OrderingConstraintType.ADJACENT:
                    distance = abs(
                        positions[constraint.item] - positions[constraint.other_item]
                    )
                    if distance != 1:
                        return False
                case OrderingConstraintType.NOT_ADJACENT:
                    distance = abs(
                        positions[constraint.item] - positions[constraint.other_item]
                    )
                    if distance == 1:
                        return False
                case OrderingConstraintType.FIXED_POSITION:
                    if positions[constraint.item] != constraint.position:
                        return False
        return True
