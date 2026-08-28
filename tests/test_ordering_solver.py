"""排序题完全枚举求解器的回归测试。"""

import pytest

from logic_qa.ordering_solver import (
    OrderingConstraint,
    OrderingConstraintType,
    OrderingSolver,
    OrderingSolveStatus,
)


def test_solves_before_adjacent_and_fixed_position_constraints() -> None:
    """求解器应同时满足先后、相邻和固定位置关系。"""
    result = OrderingSolver().solve(
        items=("A", "B", "C", "D"),
        constraints=(
            OrderingConstraint(OrderingConstraintType.BEFORE, "A", "B"),
            OrderingConstraint(OrderingConstraintType.ADJACENT, "B", "C"),
            OrderingConstraint(
                OrderingConstraintType.FIXED_POSITION,
                "D",
                position=4,
            ),
        ),
    )

    assert result.status is OrderingSolveStatus.COMPLETE
    assert result.solution_count == 2
    assert result.sample_solutions == (("A", "B", "C", "D"), ("A", "C", "B", "D"))


def test_solves_not_adjacent_constraint() -> None:
    """不相邻约束应从全排列中排除相邻位置。"""
    result = OrderingSolver().solve(
        items=("A", "B", "C"),
        constraints=(
            OrderingConstraint(OrderingConstraintType.NOT_ADJACENT, "A", "B"),
        ),
    )

    assert result.status is OrderingSolveStatus.COMPLETE
    assert result.solution_count == 2
    assert result.sample_solutions == (("A", "C", "B"), ("B", "C", "A"))


def test_reports_unsatisfiable_conflicting_constraints() -> None:
    """冲突的先后关系应返回无解，而不是无限搜索。"""
    result = OrderingSolver().solve(
        items=("A", "B"),
        constraints=(
            OrderingConstraint(OrderingConstraintType.BEFORE, "A", "B"),
            OrderingConstraint(OrderingConstraintType.BEFORE, "B", "A"),
        ),
    )

    assert result.status is OrderingSolveStatus.UNSATISFIABLE
    assert result.solution_count == 0
    assert result.sample_solutions == ()


@pytest.mark.parametrize(
    ("items", "constraint", "message"),
    [
        (
            ("A", "A"),
            None,
            "排序对象不能重复",
        ),
        (
            ("A", "B"),
            OrderingConstraint(OrderingConstraintType.BEFORE, "A", "C"),
            "约束对象不存在：“C”",
        ),
        (
            ("A", "B"),
            OrderingConstraint(
                OrderingConstraintType.FIXED_POSITION,
                "A",
                position=3,
            ),
            "固定位置必须位于对象数量范围内",
        ),
        (
            ("A", "B"),
            OrderingConstraint(OrderingConstraintType.ADJACENT, "A", "A"),
            "排序关系不能引用同一对象",
        ),
    ],
)
def test_rejects_invalid_ordering_inputs(
    items: tuple[str, ...],
    constraint: OrderingConstraint | None,
    message: str,
) -> None:
    """输入对象和约束必须在枚举前被严格校验。"""
    constraints = (constraint,) if constraint else ()

    with pytest.raises(ValueError, match=message):
        OrderingSolver().solve(items=items, constraints=constraints)


def test_blocks_excessive_item_count() -> None:
    """对象数超过安全上限时不得返回局部搜索结论。"""
    result = OrderingSolver(max_items=8).solve(
        items=tuple(f"I{index}" for index in range(9)),
        constraints=(),
    )

    assert result.status is OrderingSolveStatus.ITEM_LIMIT_EXCEEDED
    assert result.solution_count == 0
    assert result.sample_solutions == ()
