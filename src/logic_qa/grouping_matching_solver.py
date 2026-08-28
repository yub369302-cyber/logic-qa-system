"""小规模分组与一对一匹配题的完全枚举求解器。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import permutations, product


class GroupConstraintType(StrEnum):
    """第一版分组题支持的两两约束。"""

    SAME_GROUP = "same_group"
    DIFFERENT_GROUP = "different_group"


class GroupingSolveStatus(StrEnum):
    """分组求解的完成状态。"""

    COMPLETE = "complete"
    UNSATISFIABLE = "unsatisfiable"
    SEARCH_LIMIT_EXCEEDED = "search_limit_exceeded"


@dataclass(frozen=True, slots=True)
class GroupConstraint:
    """要求两个对象同组或不同组的关系约束。"""

    constraint_type: GroupConstraintType
    item: str
    other_item: str


@dataclass(frozen=True, slots=True)
class GroupAssignment:
    """一组对象到分组名称的完整分配。"""

    assignments: tuple[tuple[str, str], ...]

    def group_of(self, item: str) -> str:
        """返回指定对象所属的分组。"""
        return dict(self.assignments)[item]


@dataclass(frozen=True, slots=True)
class GroupingSolveResult:
    """分组题的精确解空间统计和可读样例。"""

    status: GroupingSolveStatus
    solution_count: int
    sample_solutions: tuple[GroupAssignment, ...]


class MatchConstraintType(StrEnum):
    """第一版一对一匹配题支持的约束。"""

    FIXED_MATCH = "fixed_match"
    FORBIDDEN_MATCH = "forbidden_match"


class MatchingSolveStatus(StrEnum):
    """一对一匹配求解的完成状态。"""

    COMPLETE = "complete"
    UNSATISFIABLE = "unsatisfiable"
    ITEM_LIMIT_EXCEEDED = "item_limit_exceeded"


@dataclass(frozen=True, slots=True)
class MatchConstraint:
    """固定对象到目标的匹配，或禁止该对象匹配某目标。"""

    constraint_type: MatchConstraintType
    item: str
    target: str


@dataclass(frozen=True, slots=True)
class MatchingAssignment:
    """对象与目标的一对一匹配结果。"""

    assignments: tuple[tuple[str, str], ...]

    def target_of(self, item: str) -> str:
        """返回指定对象所匹配的目标。"""
        return dict(self.assignments)[item]


@dataclass(frozen=True, slots=True)
class MatchingSolveResult:
    """匹配题的精确解空间统计和可读样例。"""

    status: MatchingSolveStatus
    solution_count: int
    sample_solutions: tuple[MatchingAssignment, ...]


class GroupingSolver:
    """在可控候选空间内完整枚举对象分组方案。"""

    def __init__(
        self,
        max_candidate_assignments: int = 100_000,
        sample_limit: int = 5,
    ) -> None:
        if max_candidate_assignments < 1:
            raise ValueError("max_candidate_assignments 必须至少为 1")
        if sample_limit < 1:
            raise ValueError("sample_limit 必须至少为 1")
        self._max_candidate_assignments = max_candidate_assignments
        self._sample_limit = sample_limit

    def solve(
        self,
        items: tuple[str, ...],
        groups: tuple[str, ...],
        max_group_size: int,
        constraints: tuple[GroupConstraint, ...],
    ) -> GroupingSolveResult:
        """验证输入后枚举满足容量和关系条件的全部分组方案。"""
        self._validate_input(items, groups, max_group_size, constraints)
        candidate_count = len(groups) ** len(items)
        if candidate_count > self._max_candidate_assignments:
            return GroupingSolveResult(
                status=GroupingSolveStatus.SEARCH_LIMIT_EXCEEDED,
                solution_count=0,
                sample_solutions=(),
            )

        solutions = tuple(
            assignment
            for assignment in self._candidate_assignments(items, groups)
            if self._satisfies_all(assignment, groups, max_group_size, constraints)
        )
        if not solutions:
            return GroupingSolveResult(
                status=GroupingSolveStatus.UNSATISFIABLE,
                solution_count=0,
                sample_solutions=(),
            )
        return GroupingSolveResult(
            status=GroupingSolveStatus.COMPLETE,
            solution_count=len(solutions),
            sample_solutions=solutions[: self._sample_limit],
        )

    @staticmethod
    def _validate_input(
        items: tuple[str, ...],
        groups: tuple[str, ...],
        max_group_size: int,
        constraints: tuple[GroupConstraint, ...],
    ) -> None:
        _validate_unique_nonempty(items, "分组对象")
        _validate_unique_nonempty(groups, "分组名称")
        if max_group_size < 1:
            raise ValueError("每组最大容量必须至少为 1")

        item_set = set(items)
        for constraint in constraints:
            if constraint.item not in item_set:
                raise ValueError(f"约束对象不存在：“{constraint.item}”")
            if constraint.other_item not in item_set:
                raise ValueError(f"约束对象不存在：“{constraint.other_item}”")
            if constraint.item == constraint.other_item:
                raise ValueError("分组关系不能引用同一对象")

    @staticmethod
    def _candidate_assignments(
        items: tuple[str, ...],
        groups: tuple[str, ...],
    ) -> tuple[GroupAssignment, ...]:
        return tuple(
            GroupAssignment(assignments=tuple(zip(items, choices, strict=True)))
            for choices in product(groups, repeat=len(items))
        )

    @staticmethod
    def _satisfies_all(
        assignment: GroupAssignment,
        groups: tuple[str, ...],
        max_group_size: int,
        constraints: tuple[GroupConstraint, ...],
    ) -> bool:
        group_by_item = dict(assignment.assignments)
        group_sizes = {group: 0 for group in groups}
        for group in group_by_item.values():
            group_sizes[group] += 1
        if any(size > max_group_size for size in group_sizes.values()):
            return False

        for constraint in constraints:
            same_group = (
                group_by_item[constraint.item] == group_by_item[constraint.other_item]
            )
            if constraint.constraint_type is GroupConstraintType.SAME_GROUP:
                if not same_group:
                    return False
            elif same_group:
                return False
        return True


class MatchingSolver:
    """在对象和目标数量受控时完整枚举一对一匹配。"""

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
        targets: tuple[str, ...],
        constraints: tuple[MatchConstraint, ...],
    ) -> MatchingSolveResult:
        """验证输入后枚举全部满足约束的一对一匹配方案。"""
        self._validate_input(items, targets, constraints)
        if len(items) > self._max_items:
            return MatchingSolveResult(
                status=MatchingSolveStatus.ITEM_LIMIT_EXCEEDED,
                solution_count=0,
                sample_solutions=(),
            )

        solutions = tuple(
            assignment
            for assignment in self._candidate_assignments(items, targets)
            if self._satisfies_all(assignment, constraints)
        )
        if not solutions:
            return MatchingSolveResult(
                status=MatchingSolveStatus.UNSATISFIABLE,
                solution_count=0,
                sample_solutions=(),
            )
        return MatchingSolveResult(
            status=MatchingSolveStatus.COMPLETE,
            solution_count=len(solutions),
            sample_solutions=solutions[: self._sample_limit],
        )

    @staticmethod
    def _validate_input(
        items: tuple[str, ...],
        targets: tuple[str, ...],
        constraints: tuple[MatchConstraint, ...],
    ) -> None:
        _validate_unique_nonempty(items, "匹配对象")
        _validate_unique_nonempty(targets, "匹配目标")
        if len(items) != len(targets):
            raise ValueError("匹配对象与目标数量必须相等")

        item_set = set(items)
        target_set = set(targets)
        for constraint in constraints:
            if constraint.item not in item_set:
                raise ValueError(f"匹配对象不存在：“{constraint.item}”")
            if constraint.target not in target_set:
                raise ValueError(f"匹配目标不存在：“{constraint.target}”")

    @staticmethod
    def _candidate_assignments(
        items: tuple[str, ...],
        targets: tuple[str, ...],
    ) -> tuple[MatchingAssignment, ...]:
        return tuple(
            MatchingAssignment(assignments=tuple(zip(items, choices, strict=True)))
            for choices in permutations(targets)
        )

    @staticmethod
    def _satisfies_all(
        assignment: MatchingAssignment,
        constraints: tuple[MatchConstraint, ...],
    ) -> bool:
        target_by_item = dict(assignment.assignments)
        for constraint in constraints:
            matches = target_by_item[constraint.item] == constraint.target
            if constraint.constraint_type is MatchConstraintType.FIXED_MATCH:
                if not matches:
                    return False
            elif matches:
                return False
        return True


def _validate_unique_nonempty(values: tuple[str, ...], label: str) -> None:
    """校验对象集合非空、无空文本且没有重复值。"""
    if not values:
        raise ValueError(f"{label}不能为空")
    if any(not value.strip() for value in values):
        raise ValueError(f"{label}不能包含空文本")
    if len(set(values)) != len(values):
        raise ValueError(f"{label}不能重复")
