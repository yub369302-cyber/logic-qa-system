"""分组与一对一匹配求解器的回归测试。"""

import pytest

from logic_qa.grouping_matching_solver import (
    GroupConstraint,
    GroupConstraintType,
    GroupingSolver,
    GroupingSolveStatus,
    MatchConstraint,
    MatchConstraintType,
    MatchingSolver,
    MatchingSolveStatus,
)


def test_grouping_solver_respects_capacity_and_group_relations() -> None:
    """分组器应同时处理容量、同组与不同组约束。"""
    result = GroupingSolver().solve(
        items=("A", "B", "C"),
        groups=("G1", "G2"),
        max_group_size=2,
        constraints=(
            GroupConstraint(GroupConstraintType.SAME_GROUP, "A", "B"),
            GroupConstraint(GroupConstraintType.DIFFERENT_GROUP, "A", "C"),
        ),
    )

    assert result.status is GroupingSolveStatus.COMPLETE
    assert result.solution_count == 2
    for assignment in result.sample_solutions:
        assert assignment.group_of("A") == assignment.group_of("B")
        assert assignment.group_of("A") != assignment.group_of("C")


def test_grouping_solver_reports_conflicting_constraints() -> None:
    """同组与不同组同时要求时应返回无解。"""
    result = GroupingSolver().solve(
        items=("A", "B"),
        groups=("G1", "G2"),
        max_group_size=2,
        constraints=(
            GroupConstraint(GroupConstraintType.SAME_GROUP, "A", "B"),
            GroupConstraint(GroupConstraintType.DIFFERENT_GROUP, "A", "B"),
        ),
    )

    assert result.status is GroupingSolveStatus.UNSATISFIABLE
    assert result.solution_count == 0


def test_grouping_solver_stops_before_excessive_search() -> None:
    """候选分配数量超过限制时不得产生局部解。"""
    result = GroupingSolver(max_candidate_assignments=10).solve(
        items=("A", "B", "C"),
        groups=("G1", "G2", "G3"),
        max_group_size=3,
        constraints=(),
    )

    assert result.status is GroupingSolveStatus.SEARCH_LIMIT_EXCEEDED
    assert result.solution_count == 0


@pytest.mark.parametrize(
    ("items", "groups", "max_size", "constraints", "message"),
    [
        (("A", "A"), ("G1",), 1, (), "分组对象不能重复"),
        (("A",), ("G1", "G1"), 1, (), "分组名称不能重复"),
        (("A",), ("G1",), 0, (), "每组最大容量必须至少为 1"),
        (
            ("A", "B"),
            ("G1",),
            2,
            (GroupConstraint(GroupConstraintType.SAME_GROUP, "A", "C"),),
            "约束对象不存在：“C”",
        ),
    ],
)
def test_grouping_solver_rejects_invalid_input(
    items: tuple[str, ...],
    groups: tuple[str, ...],
    max_size: int,
    constraints: tuple[GroupConstraint, ...],
    message: str,
) -> None:
    """分组输入必须在枚举前完成严格校验。"""
    with pytest.raises(ValueError, match=message):
        GroupingSolver().solve(items, groups, max_size, constraints)


def test_matching_solver_respects_fixed_and_forbidden_matches() -> None:
    """匹配器应正确处理固定和禁止配对。"""
    result = MatchingSolver().solve(
        items=("A", "B", "C"),
        targets=("X", "Y", "Z"),
        constraints=(
            MatchConstraint(MatchConstraintType.FIXED_MATCH, "A", "X"),
            MatchConstraint(MatchConstraintType.FORBIDDEN_MATCH, "B", "Y"),
        ),
    )

    assert result.status is MatchingSolveStatus.COMPLETE
    assert result.solution_count == 1
    assignment = result.sample_solutions[0]
    assert assignment.target_of("A") == "X"
    assert assignment.target_of("B") == "Z"


def test_matching_solver_reports_impossible_matches() -> None:
    """同一对象同时固定到不同目标时应返回无解。"""
    result = MatchingSolver().solve(
        items=("A", "B"),
        targets=("X", "Y"),
        constraints=(
            MatchConstraint(MatchConstraintType.FIXED_MATCH, "A", "X"),
            MatchConstraint(MatchConstraintType.FIXED_MATCH, "A", "Y"),
        ),
    )

    assert result.status is MatchingSolveStatus.UNSATISFIABLE
    assert result.solution_count == 0


def test_matching_solver_blocks_excessive_item_count() -> None:
    """匹配对象数量超过上限时不得返回局部搜索结论。"""
    items = tuple(f"I{index}" for index in range(9))
    targets = tuple(f"T{index}" for index in range(9))
    result = MatchingSolver(max_items=8).solve(items, targets, ())

    assert result.status is MatchingSolveStatus.ITEM_LIMIT_EXCEEDED
    assert result.solution_count == 0


@pytest.mark.parametrize(
    ("items", "targets", "constraints", "message"),
    [
        (("A",), ("X", "Y"), (), "匹配对象与目标数量必须相等"),
        (
            ("A",),
            ("X",),
            (MatchConstraint(MatchConstraintType.FIXED_MATCH, "B", "X"),),
            "匹配对象不存在：“B”",
        ),
        (
            ("A",),
            ("X",),
            (MatchConstraint(MatchConstraintType.FORBIDDEN_MATCH, "A", "Y"),),
            "匹配目标不存在：“Y”",
        ),
    ],
)
def test_matching_solver_rejects_invalid_input(
    items: tuple[str, ...],
    targets: tuple[str, ...],
    constraints: tuple[MatchConstraint, ...],
    message: str,
) -> None:
    """匹配输入必须在枚举前完成严格校验。"""
    with pytest.raises(ValueError, match=message):
        MatchingSolver().solve(items, targets, constraints)
