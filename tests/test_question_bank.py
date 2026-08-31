"""内容绑定审核题库发布与个人练习推荐的回归测试。"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from logic_qa.grouping_matching_solver import (
    GroupConstraint,
    GroupConstraintType,
    GroupingSolveStatus,
    MatchConstraint,
    MatchConstraintType,
    MatchingSolveStatus,
)
from logic_qa.learning_profile import LearningProfile, LearningRecommendation
from logic_qa.models import VerificationStatus
from logic_qa.ordering_solver import (
    OrderingConstraint,
    OrderingConstraintType,
    OrderingSolveStatus,
)
from logic_qa.quality_operations import (
    QuestionReviewInput,
    QuestionReviewStatus,
    QuestionReviewStore,
)
from logic_qa.question_bank import (
    FormalizationRule,
    GroupingFormalization,
    MatchingFormalization,
    OptionAssertion,
    OrderingFormalization,
    PropositionalFormalization,
    QuestionBankStore,
    QuestionPublicationInput,
)


def _review_store(tmp_path: Path) -> QuestionReviewStore:
    return QuestionReviewStore(tmp_path / "reviews.sqlite3")


def _question_store(tmp_path: Path) -> QuestionBankStore:
    return QuestionBankStore(tmp_path / "questions.sqlite3")


def _formalization(
    expected_status: VerificationStatus = VerificationStatus.PROVED,
    expected_answer: str = "B",
) -> PropositionalFormalization:
    return PropositionalFormalization(
        facts=("A",),
        rules=(FormalizationRule("A", "B", "如果 A，那么 B"),),
        query="B",
        expected_status=expected_status,
        expected_answer=expected_answer,
        option_assertions=(
            OptionAssertion("A", "disproved"),
            OptionAssertion("B", "proved"),
        ),
    )


def _publication(
    question_id: str,
    version: str,
    stem: str | None = None,
    formalization: (
        PropositionalFormalization
        | OrderingFormalization
        | GroupingFormalization
        | MatchingFormalization
        | None
    ) = None,
    question_type: str = "propositional",
) -> QuestionPublicationInput:
    return QuestionPublicationInput(
        question_id=question_id,
        content_version=version,
        question_type=question_type,
        stem=stem or f"{question_id} 的题干",
        options=("A", "B"),
        error_tags=("invalid_converse",),
        knowledge_tags=("逆命题与逆否命题",),
        formalization_version="logic-v1",
        formalization=formalization or _formalization(),
    )


def _approved_review(
    review_store: QuestionReviewStore,
    question_store: QuestionBankStore,
    publication: QuestionPublicationInput,
):
    candidate = question_store.submit_candidate(publication)
    review = review_store.upsert_review(
        QuestionReviewInput(
            question_id=candidate.publication.question_id,
            content_version=candidate.publication.content_version,
            content_hash=candidate.content_hash,
            reviewer_id="reviewer-a",
            status=QuestionReviewStatus.APPROVED,
            verified_answer="B",
            formalization_version=candidate.publication.formalization_version,
        )
    )
    return candidate, review


def _profile() -> LearningProfile:
    return LearningProfile(
        user_id="user-a",
        total_attempts=2,
        correct_attempts=1,
        accuracy=0.5,
        error_counts=(("invalid_converse", 2),),
        knowledge_mastery=(("逆命题与逆否命题", 0.0),),
        recommendations=(
            LearningRecommendation(
                focus_type="error_tag",
                label="invalid_converse",
                reason="test",
                suggested_practice="test",
            ),
        ),
    )


def test_submitted_candidate_is_immutable_and_retrievable(tmp_path: Path) -> None:
    """候选快照按题号、版本和摘要保存，重复提交同内容保持幂等。"""
    question_store = _question_store(tmp_path)
    publication = _publication("q-1", "content-v1")

    first = question_store.submit_candidate(publication)
    second = question_store.submit_candidate(publication)
    changed = question_store.submit_candidate(
        _publication("q-1", "content-v1", stem="同版本的另一份候选题干")
    )
    changed_formalization = question_store.submit_candidate(
        _publication(
            "q-1",
            "content-v1",
            formalization=_formalization(expected_answer="A"),
        )
    )
    changed_option_semantics = question_store.submit_candidate(
        _publication(
            "q-1",
            "content-v1",
            formalization=replace(
                _formalization(),
                option_assertions=(
                    OptionAssertion("A", "proved"),
                    OptionAssertion("B", "disproved"),
                ),
            ),
        )
    )

    assert first == second
    assert first.content_hash != changed.content_hash
    assert first.content_hash != changed_formalization.content_hash
    assert first.content_hash != changed_option_semantics.content_hash
    assert first.publication.formalization is not None
    assert first.publication.formalization.query == "B"
    assert (
        question_store.get_candidate(
            first.publication.question_id,
            first.publication.content_version,
            first.content_hash,
        )
        == first
    )
    assert (
        question_store.get_candidate(
            changed.publication.question_id,
            changed.publication.content_version,
            changed.content_hash,
        )
        == changed
    )
    assert (
        question_store.get_candidate(
            changed_formalization.publication.question_id,
            changed_formalization.publication.content_version,
            changed_formalization.content_hash,
        )
        == changed_formalization
    )
    assert (
        question_store.get_candidate(
            changed_option_semantics.publication.question_id,
            changed_option_semantics.publication.content_version,
            changed_option_semantics.content_hash,
        )
        == changed_option_semantics
    )


def test_publishes_only_when_matching_content_is_approved(tmp_path: Path) -> None:
    """发布必须精确匹配已保存候选与同一题号、版本、摘要和形式化版本的审核。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    candidate, review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1"),
    )

    published = question_store.publish(candidate, "publisher-a", review)

    assert published.question_id == "q-1"
    assert published.content_hash == candidate.content_hash
    assert published.formalization.expected_status is VerificationStatus.PROVED
    assert question_store.active_questions() == (published,)
    verification = question_store.get_formalization_verification(
        published.question_id,
        published.content_version,
        published.content_hash,
    )
    assert verification is not None
    assert verification.selected_option == "B"
    assert verification.matching_options == ("B",)

    with question_store._connect() as connection:
        event = connection.execute(
            """
            SELECT expected_status, actual_status, proof_steps, known_literals,
                   selected_option, matching_options
            FROM question_formalization_verification_events
            WHERE question_id = ? AND content_hash = ?
            """,
            (published.question_id, published.content_hash),
        ).fetchone()
    assert event is not None
    assert event["expected_status"] == "proved"
    assert event["actual_status"] == "proved"
    assert event["selected_option"] == "B"
    assert event["matching_options"] == '["B"]'
    assert "B" in event["known_literals"]
    assert "条件推理" in event["proof_steps"]


@pytest.mark.parametrize(
    (
        "question_type",
        "formalization",
        "expected_status",
        "expected_solution_count",
        "evidence_fragment",
    ),
    [
        (
            "ordering",
            OrderingFormalization(
                items=("A", "B", "C"),
                constraints=(
                    OrderingConstraint(OrderingConstraintType.BEFORE, "A", "B"),
                ),
                expected_status=OrderingSolveStatus.COMPLETE,
                expected_solution_count=3,
                expected_answer="B",
                option_assertions=(
                    OptionAssertion("A", "complete", 2),
                    OptionAssertion("B", "complete", 3),
                ),
            ),
            "complete",
            3,
            "A",
        ),
        (
            "grouping",
            GroupingFormalization(
                items=("A", "B"),
                groups=("G1", "G2"),
                max_group_size=2,
                constraints=(
                    GroupConstraint(GroupConstraintType.SAME_GROUP, "A", "B"),
                ),
                expected_status=GroupingSolveStatus.COMPLETE,
                expected_solution_count=2,
                expected_answer="B",
                option_assertions=(
                    OptionAssertion("A", "complete", 1),
                    OptionAssertion("B", "complete", 2),
                ),
            ),
            "complete",
            2,
            "G1",
        ),
        (
            "matching",
            MatchingFormalization(
                items=("A", "B"),
                targets=("X", "Y"),
                constraints=(
                    MatchConstraint(MatchConstraintType.FIXED_MATCH, "A", "X"),
                ),
                expected_status=MatchingSolveStatus.COMPLETE,
                expected_solution_count=1,
                expected_answer="B",
                option_assertions=(
                    OptionAssertion("A", "complete", 2),
                    OptionAssertion("B", "complete", 1),
                ),
            ),
            "complete",
            1,
            "X",
        ),
    ],
)
def test_publishes_constraint_question_with_reproducible_verification(
    tmp_path: Path,
    question_type: str,
    formalization: (
        OrderingFormalization | GroupingFormalization | MatchingFormalization
    ),
    expected_status: str,
    expected_solution_count: int,
    evidence_fragment: str,
) -> None:
    """排序、分组与匹配题均应通过对应完整枚举器复核后才可发布。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    candidate, review = _approved_review(
        review_store,
        question_store,
        _publication(
            f"q-{question_type}",
            "content-v1",
            formalization=formalization,
            question_type=question_type,
        ),
    )

    published = question_store.publish(candidate, "publisher-a", review)
    verification = question_store.get_formalization_verification(
        published.question_id,
        published.content_version,
        published.content_hash,
    )

    assert verification is not None
    assert verification.expected_status.value == expected_status
    assert verification.actual_status.value == expected_status
    assert verification.expected_solution_count == expected_solution_count
    assert verification.actual_solution_count == expected_solution_count
    assert verification.evidence is not None
    assert evidence_fragment in verification.evidence


@pytest.mark.parametrize(
    "formalization",
    [
        OrderingFormalization(
            items=("A", "B"),
            constraints=(),
            expected_status=OrderingSolveStatus.COMPLETE,
            expected_solution_count=1,
            expected_answer="B",
            option_assertions=(
                OptionAssertion("A", "complete", 2),
                OptionAssertion("B", "complete", 1),
            ),
        ),
        GroupingFormalization(
            items=("A", "B"),
            groups=("G1", "G2"),
            max_group_size=2,
            constraints=(),
            expected_status=GroupingSolveStatus.COMPLETE,
            expected_solution_count=1,
            expected_answer="B",
            option_assertions=(
                OptionAssertion("A", "complete", 4),
                OptionAssertion("B", "complete", 1),
            ),
        ),
        MatchingFormalization(
            items=("A", "B"),
            targets=("X", "Y"),
            constraints=(),
            expected_status=MatchingSolveStatus.COMPLETE,
            expected_solution_count=1,
            expected_answer="B",
            option_assertions=(
                OptionAssertion("A", "complete", 2),
                OptionAssertion("B", "complete", 1),
            ),
        ),
    ],
)
def test_rejects_constraint_question_with_wrong_solution_count(
    tmp_path: Path,
    formalization: (
        OrderingFormalization | GroupingFormalization | MatchingFormalization
    ),
) -> None:
    """声明的完整解数量与求解器复算不一致时，发布必须被拒绝。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    question_type = (
        type(formalization).__name__.removesuffix("Formalization").lower()
    )
    candidate, review = _approved_review(
        review_store,
        question_store,
        _publication(
            f"q-{question_type}-wrong-count",
            "content-v1",
            formalization=formalization,
            question_type=question_type,
        ),
    )

    with pytest.raises(ValueError, match="解空间数量"):
        question_store.publish(candidate, "publisher-a", review)


def test_rejects_publication_for_unsubmitted_candidate(tmp_path: Path) -> None:
    """仅计算摘要而未提交快照的内容不能绕过候选提交门禁。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    candidate = question_store.prepare_candidate(_publication("q-1", "content-v1"))
    review = review_store.upsert_review(
        QuestionReviewInput(
            question_id=candidate.publication.question_id,
            content_version=candidate.publication.content_version,
            content_hash=candidate.content_hash,
            reviewer_id="reviewer-a",
            status=QuestionReviewStatus.APPROVED,
            verified_answer="B",
            formalization_version=candidate.publication.formalization_version,
        )
    )

    with pytest.raises(ValueError, match="候选内容尚未提交"):
        question_store.publish(candidate, "publisher-a", review)


def test_rejects_non_unique_or_mismatched_option_semantics(tmp_path: Path) -> None:
    """必须恰好一个选项与复算结果匹配，且该选项必须是预期答案。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    non_unique_candidate, non_unique_review = _approved_review(
        review_store,
        question_store,
        _publication(
            "q-non-unique",
            "content-v1",
            formalization=PropositionalFormalization(
                facts=("A",),
                rules=(FormalizationRule("A", "B"),),
                query="B",
                expected_status=VerificationStatus.PROVED,
                expected_answer="B",
                option_assertions=(
                    OptionAssertion("A", "proved"),
                    OptionAssertion("B", "proved"),
                ),
            ),
        ),
    )
    mismatched_candidate, mismatched_review = _approved_review(
        review_store,
        question_store,
        _publication(
            "q-mismatch",
            "content-v1",
            formalization=PropositionalFormalization(
                facts=("A",),
                rules=(FormalizationRule("A", "B"),),
                query="B",
                expected_status=VerificationStatus.PROVED,
                expected_answer="B",
                option_assertions=(
                    OptionAssertion("A", "proved"),
                    OptionAssertion("B", "disproved"),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="唯一命中"):
        question_store.publish(
            non_unique_candidate,
            "publisher-a",
            non_unique_review,
        )
    with pytest.raises(ValueError, match="预期答案与选项语义"):
        question_store.publish(
            mismatched_candidate,
            "publisher-a",
            mismatched_review,
        )


def test_rejects_review_answer_mismatched_with_formalization(tmp_path: Path) -> None:
    """审核结论的答案必须与形式化资产的预期答案精确一致。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    candidate = question_store.submit_candidate(_publication("q-1", "content-v1"))
    review = review_store.upsert_review(
        QuestionReviewInput(
            question_id=candidate.publication.question_id,
            content_version=candidate.publication.content_version,
            content_hash=candidate.content_hash,
            reviewer_id="reviewer-a",
            status=QuestionReviewStatus.APPROVED,
            verified_answer="A",
            formalization_version=candidate.publication.formalization_version,
        )
    )

    with pytest.raises(ValueError, match="审核核验答案"):
        question_store.publish(candidate, "publisher-a", review)


def test_rejects_incorrect_expected_formalization_status(tmp_path: Path) -> None:
    """形式化资产的预期结论与内核复算不一致时，发布必须被门禁拦截。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    candidate, review = _approved_review(
        review_store,
        question_store,
        _publication(
            "q-1",
            "content-v1",
            formalization=_formalization(VerificationStatus.DISPROVED),
        ),
    )

    with pytest.raises(ValueError, match="形式化验证结果与预期不一致"):
        question_store.publish(candidate, "publisher-a", review)


def test_rejects_unresolved_or_inconsistent_formalization(tmp_path: Path) -> None:
    """无法确定或全局矛盾的形式化条件不能被误发布为已验证题目。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    unresolved_candidate, unresolved_review = _approved_review(
        review_store,
        question_store,
        _publication(
            "q-unknown",
            "content-v1",
            formalization=PropositionalFormalization(
                facts=(),
                rules=(),
                query="B",
                expected_status=VerificationStatus.UNKNOWN,
                expected_answer="B",
                option_assertions=(
                    OptionAssertion("A", "disproved"),
                    OptionAssertion("B", "unknown"),
                ),
            ),
        ),
    )
    inconsistent_candidate, inconsistent_review = _approved_review(
        review_store,
        question_store,
        _publication(
            "q-inconsistent",
            "content-v1",
            formalization=PropositionalFormalization(
                facts=("A", "!A"),
                rules=(),
                query="B",
                expected_status=VerificationStatus.INCONSISTENT,
                expected_answer="B",
                option_assertions=(
                    OptionAssertion("A", "disproved"),
                    OptionAssertion("B", "inconsistent"),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="无法确定"):
        question_store.publish(
            unresolved_candidate,
            "publisher-a",
            unresolved_review,
        )
    with pytest.raises(ValueError, match="存在矛盾"):
        question_store.publish(
            inconsistent_candidate,
            "publisher-a",
            inconsistent_review,
        )


def test_rejects_changed_content_even_when_version_and_review_are_reused(
    tmp_path: Path,
) -> None:
    """题干、标签或选项改变会改变摘要，旧审核不得被重新用于发布。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    original_candidate, original_review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1", stem="原始题干"),
    )
    changed_candidate = question_store.submit_candidate(
        _publication("q-1", "content-v1", stem="已修改的题干")
    )

    assert original_candidate.content_hash != changed_candidate.content_hash
    with pytest.raises(ValueError, match="内容摘要"):
        question_store.publish(changed_candidate, "publisher-a", original_review)


def test_rejects_publication_without_matching_approval(tmp_path: Path) -> None:
    """缺少审核、审核未通过或版本不一致均不得发布题目。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    candidate = question_store.submit_candidate(_publication("q-1", "content-v1"))

    with pytest.raises(ValueError, match="尚无审核记录"):
        question_store.publish(candidate, "publisher-a", None)

    pending_review = review_store.upsert_review(
        QuestionReviewInput(
            question_id=candidate.publication.question_id,
            content_version=candidate.publication.content_version,
            content_hash=candidate.content_hash,
            reviewer_id="reviewer-a",
            status=QuestionReviewStatus.PENDING,
            verified_answer=None,
            formalization_version=candidate.publication.formalization_version,
        )
    )
    with pytest.raises(ValueError, match="未通过审核"):
        question_store.publish(candidate, "publisher-a", pending_review)

    approved_review = review_store.upsert_review(
        QuestionReviewInput(
            question_id=candidate.publication.question_id,
            content_version=candidate.publication.content_version,
            content_hash=candidate.content_hash,
            reviewer_id="reviewer-a",
            status=QuestionReviewStatus.APPROVED,
            verified_answer="B",
            formalization_version=candidate.publication.formalization_version,
        )
    )
    wrong_version_candidate = question_store.submit_candidate(
        _publication("q-1", "content-v2")
    )
    with pytest.raises(ValueError, match="内容版本"):
        question_store.publish(wrong_version_candidate, "publisher-a", approved_review)


def test_new_version_deactivates_old_version_but_keeps_history(tmp_path: Path) -> None:
    """同题新版本发布后只保留新版本为活动状态。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    first_candidate, first_review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1"),
    )
    second_candidate, second_review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v2"),
    )
    first = question_store.publish(first_candidate, "publisher-a", first_review)
    second = question_store.publish(second_candidate, "publisher-a", second_review)

    assert first.content_version == "content-v1"
    assert question_store.active_questions() == (second,)
    supersession_events = question_store.list_question_version_lifecycle_events("q-1")
    assert len(supersession_events) == 1
    superseded = supersession_events[0]
    assert superseded.question_id == first.question_id
    assert superseded.content_version == first.content_version
    assert superseded.content_hash == first.content_hash
    assert superseded.action.value == "superseded"
    assert superseded.actor_id == "publisher-a"
    assert superseded.replaced_content_version == second.content_version
    assert superseded.reason == "发布新的已审核内容版本"
    assert superseded.created_at == second.published_at


def test_publish_rejects_a_corrupted_question_with_multiple_active_versions(
    tmp_path: Path,
) -> None:
    """发布不会猜测多个活动版本的替代关系或写入部分审计事实。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    first_candidate, first_review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1"),
    )
    second_candidate, second_review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v2", stem="q-1 的第二版本题干"),
    )
    third_candidate, third_review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v3", stem="q-1 的第三版本题干"),
    )
    first = question_store.publish(first_candidate, "publisher-a", first_review)
    second = question_store.publish(second_candidate, "publisher-a", second_review)
    with question_store._connect() as connection:
        connection.execute(
            """
            UPDATE question_versions
            SET is_active = 1
            WHERE question_id = ? AND content_version = ?
            """,
            (first.question_id, first.content_version),
        )

    with pytest.raises(ValueError, match="多个活动版本"):
        question_store.publish(third_candidate, "publisher-a", third_review)

    assert question_store.get_published_question("q-1", "content-v3") is None
    assert question_store.active_questions() == (first, second)
    lifecycle_events = question_store.list_question_version_lifecycle_events("q-1")
    assert lifecycle_events[0].action.value == "superseded"


def test_deactivation_and_reactivation_preserve_history_and_append_events(
    tmp_path: Path,
) -> None:
    """下线和回滚只变更活动指针，保留发布事实并追加可审计复验记录。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    first_candidate, first_review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1"),
    )
    second_candidate, second_review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v2", stem="q-1 的新版本题干"),
    )
    first = question_store.publish(first_candidate, "publisher-a", first_review)
    second = question_store.publish(second_candidate, "publisher-a", second_review)

    deactivated = question_store.deactivate_active_version(
        "q-1",
        "content-v2",
        actor_id="admin-a",
        reason="发现题干表述需要暂时下线",
    )

    assert deactivated is not None
    assert deactivated.action.value == "deactivated"
    assert deactivated.content_hash == second.content_hash
    assert deactivated.actor_id == "admin-a"
    assert deactivated.replaced_content_version is None
    assert question_store.active_questions() == ()
    assert question_store.get_published_question("q-1", "content-v1") == first
    assert question_store.get_published_question("q-1", "content-v2") == second
    assert question_store.get_active_learner_question("q-1", "content-v2") is None
    assert question_store.grade_active_learner_answer("q-1", "content-v2", "B") is None

    reactivated = question_store.reactivate_published_version(
        "q-1",
        "content-v1",
        actor_id="admin-b",
        reason="复验历史版本后执行受控回滚",
        review=review_store.get_review(
            first.question_id,
            first.content_version,
            first.content_hash,
        ),
    )

    assert reactivated is not None
    assert reactivated.action.value == "reactivated"
    assert reactivated.content_hash == first.content_hash
    assert reactivated.actor_id == "admin-b"
    assert reactivated.replaced_content_version is None
    assert question_store.active_questions() == (first,)
    assert question_store.get_active_learner_question("q-1", "content-v1") is not None
    assert (
        question_store.grade_active_learner_answer("q-1", "content-v1", "B")
        is not None
    )
    assert question_store.get_published_question("q-1", "content-v2") == second
    lifecycle_events = question_store.list_question_version_lifecycle_events("q-1")
    assert lifecycle_events[0].action.value == "superseded"
    assert lifecycle_events[0].content_version == first.content_version
    assert lifecycle_events[0].replaced_content_version == second.content_version
    assert lifecycle_events[1:] == (deactivated, reactivated)
    with question_store._connect() as connection:
        verification_count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM question_formalization_verification_events
            WHERE question_id = ? AND content_version = ? AND content_hash = ?
            """,
            (first.question_id, first.content_version, first.content_hash),
        ).fetchone()["count"]
    assert verification_count == 2


def test_deactivation_is_atomic_under_concurrent_administrators(
    tmp_path: Path,
) -> None:
    """竞争下线同一活动版本时，只能提交一条下线事实。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    candidate, review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1"),
    )
    published = question_store.publish(candidate, "publisher-a", review)

    def deactivate() -> str:
        try:
            event = question_store.deactivate_active_version(
                published.question_id,
                published.content_version,
                actor_id="admin-a",
                reason="并发下线治理测试",
            )
        except ValueError as error:
            assert "当前未活动" in str(error)
            return "already_inactive"
        assert event is not None
        return event.event_id

    with ThreadPoolExecutor(max_workers=4) as executor:
        outcomes = tuple(executor.map(lambda _: deactivate(), range(4)))

    successful_event_ids = tuple(
        outcome for outcome in outcomes if outcome != "already_inactive"
    )
    assert len(successful_event_ids) == 1
    assert outcomes.count("already_inactive") == 3
    assert question_store.active_questions() == ()
    lifecycle_events = question_store.list_question_version_lifecycle_events("q-1")
    assert len(lifecycle_events) == 1
    assert lifecycle_events[0].event_id == successful_event_ids[0]
    assert lifecycle_events[0].action.value == "deactivated"


def test_reactivation_replaces_current_version_after_revalidation(
    tmp_path: Path,
) -> None:
    """回滚可替换另一活动版本，但必须保留其历史记录并在事件中指明替代版本。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    first_candidate, first_review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1"),
    )
    second_candidate, second_review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v2", stem="q-1 的新版本题干"),
    )
    first = question_store.publish(first_candidate, "publisher-a", first_review)
    second = question_store.publish(second_candidate, "publisher-a", second_review)

    reactivated = question_store.reactivate_published_version(
        "q-1",
        "content-v1",
        actor_id="admin-a",
        reason="审核确认回滚到历史版本",
        review=review_store.get_review(
            first.question_id,
            first.content_version,
            first.content_hash,
        ),
    )

    assert reactivated is not None
    assert reactivated.action.value == "reactivated"
    assert reactivated.replaced_content_version == second.content_version
    assert question_store.active_questions() == (first,)
    assert question_store.get_active_published_question("q-1", "content-v2") is None
    assert question_store.get_published_question("q-1", "content-v2") == second
    lifecycle_events = question_store.list_question_version_lifecycle_events("q-1")
    assert [event.action.value for event in lifecycle_events] == [
        "superseded",
        "superseded",
        "reactivated",
    ]
    superseded_by_reactivation = lifecycle_events[1]
    assert superseded_by_reactivation.content_version == second.content_version
    assert superseded_by_reactivation.content_hash == second.content_hash
    assert superseded_by_reactivation.actor_id == "admin-a"
    assert superseded_by_reactivation.replaced_content_version == first.content_version
    assert superseded_by_reactivation.reason == "重新激活已审核历史版本"
    assert superseded_by_reactivation.created_at == reactivated.created_at


def test_reactivation_requires_current_exact_approval_and_immutable_candidate(
    tmp_path: Path,
) -> None:
    """历史版本重新上线不能绕过当前审核状态或精确候选快照。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    first_candidate, first_review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1"),
    )
    second_candidate, second_review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v2"),
    )
    first = question_store.publish(first_candidate, "publisher-a", first_review)
    question_store.publish(second_candidate, "publisher-a", second_review)
    stale_review = review_store.upsert_review(
        QuestionReviewInput(
            question_id=first.question_id,
            content_version=first.content_version,
            content_hash=first.content_hash,
            reviewer_id="reviewer-b",
            status=QuestionReviewStatus.NEEDS_REVISION,
            verified_answer=None,
            formalization_version=first.formalization_version,
        )
    )

    with pytest.raises(ValueError, match="未通过审核"):
        question_store.reactivate_published_version(
            "q-1",
            "content-v1",
            actor_id="admin-a",
            reason="尝试绕过已更新审核",
            review=stale_review,
        )
    assert question_store.active_questions()[0].content_version == "content-v2"

    with question_store._connect() as connection:
        connection.execute(
            """
            DELETE FROM question_candidates
            WHERE question_id = ? AND content_version = ? AND content_hash = ?
            """,
            (first.question_id, first.content_version, first.content_hash),
        )
    with pytest.raises(ValueError, match="缺少精确候选快照"):
        question_store.reactivate_published_version(
            "q-1",
            "content-v1",
            actor_id="admin-a",
            reason="尝试绕过候选快照",
            review=stale_review,
        )
    assert question_store.active_questions()[0].content_version == "content-v2"


def test_version_lifecycle_operations_validate_state_and_identifiers(
    tmp_path: Path,
) -> None:
    """生命周期治理拒绝空标识、空理由、缺失版本和重复状态切换。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    candidate, review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1"),
    )
    question_store.publish(candidate, "publisher-a", review)

    with pytest.raises(ValueError, match="下线原因不能为空"):
        question_store.deactivate_active_version(
            "q-1",
            "content-v1",
            actor_id="admin-a",
            reason=" ",
        )
    assert question_store.deactivate_active_version(
        "q-1",
        "content-v1",
        actor_id="admin-a",
        reason="受控下线",
    ) is not None
    with pytest.raises(ValueError, match="当前未活动"):
        question_store.deactivate_active_version(
            "q-1",
            "content-v1",
            actor_id="admin-a",
            reason="重复下线",
        )
    with pytest.raises(ValueError, match="重新激活原因不能为空"):
        question_store.reactivate_published_version(
            "q-1",
            "content-v1",
            actor_id="admin-a",
            reason=" ",
            review=review,
        )
    assert question_store.reactivate_published_version(
        "missing",
        "content-v1",
        actor_id="admin-a",
        reason="不存在版本",
        review=review,
    ) is None


def test_recommendation_excludes_attempted_questions_and_hides_internal_tags(
    tmp_path: Path,
) -> None:
    """推荐应排除已做题，并向学习者仅返回最小展示题目视图。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    candidate_one, review_one = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1"),
    )
    candidate_two, review_two = _approved_review(
        review_store,
        question_store,
        _publication("q-2", "content-v1"),
    )
    question_store.publish(candidate_one, "publisher-a", review_one)
    question_store.publish(candidate_two, "publisher-a", review_two)

    recommendations = question_store.recommend(
        profile=_profile(),
        attempted_question_versions=(("q-1", "content-v1"),),
        limit=3,
    )

    assert len(recommendations) == 1
    recommendation = recommendations[0]
    assert recommendation.question.question_id == "q-2"
    assert recommendation.matched_tags == ("invalid_converse", "逆命题与逆否命题")
    assert "高频错因" in recommendation.reason
    assert not hasattr(recommendation.question, "content_hash")
    assert not hasattr(recommendation.question, "error_tags")


def test_get_active_published_question_returns_current_audited_version(
    tmp_path: Path,
) -> None:
    """治理链路只能读取当前活动版本，且保留其可信内容摘要和形式化资产。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    candidate, review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1"),
    )
    published = question_store.publish(candidate, "publisher-a", review)

    active = question_store.get_active_published_question("q-1", "content-v1")

    assert active == published
    assert active.content_hash == candidate.content_hash
    assert active.formalization.expected_answer == "B"
    assert question_store.get_published_question("q-1", "content-v1") == published
    assert question_store.get_published_question("missing", "content-v1") is None
    assert question_store.get_active_published_question("missing", "content-v1") is None
    with pytest.raises(ValueError, match="题目标识不能为空"):
        question_store.get_active_published_question(" ", "content-v1")
    with pytest.raises(ValueError, match="内容版本不能为空"):
        question_store.get_active_published_question("q-1", " ")
    with pytest.raises(ValueError, match="内容版本不能为空"):
        question_store.get_published_question("q-1", " ")


def test_published_question_verification_snapshot_keeps_history_and_activity_together(
    tmp_path: Path,
) -> None:
    """跨库核验快照必须在同一次题库读取中返回历史事实及活动状态。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    candidate, review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1"),
    )
    published = question_store.publish(candidate, "publisher-a", review)

    active_snapshot = question_store.get_published_question_verification_snapshot(
        "q-1",
        "content-v1",
    )

    assert active_snapshot.question == published
    assert active_snapshot.is_active is True
    assert question_store.deactivate_active_version(
        "q-1",
        "content-v1",
        actor_id="admin-a",
        reason="核验快照状态测试",
    ) is not None
    inactive_snapshot = question_store.get_published_question_verification_snapshot(
        "q-1",
        "content-v1",
    )
    missing_snapshot = question_store.get_published_question_verification_snapshot(
        "missing",
        "content-v1",
    )

    assert inactive_snapshot.question == published
    assert inactive_snapshot.is_active is False
    assert missing_snapshot.question is None
    assert missing_snapshot.is_active is None
    with pytest.raises(ValueError, match="题目标识不能为空"):
        question_store.get_published_question_verification_snapshot(" ", "content-v1")
    with pytest.raises(ValueError, match="内容版本不能为空"):
        question_store.get_published_question_verification_snapshot("q-1", " ")


def test_get_active_learner_question_hides_internal_fields_and_history(
    tmp_path: Path,
) -> None:
    """学习者只能读取当前活动版本，且领域对象不携带内部发布信息。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    first_candidate, first_review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1"),
    )
    second_candidate, second_review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v2", stem="q-1 的新版题干"),
    )
    question_store.publish(first_candidate, "publisher-a", first_review)
    question_store.publish(second_candidate, "publisher-a", second_review)

    active = question_store.get_active_learner_question("q-1", "content-v2")

    assert active is not None
    assert active.stem == "q-1 的新版题干"
    assert not hasattr(active, "content_hash")
    assert not hasattr(active, "formalization")
    assert question_store.get_active_learner_question("q-1", "content-v1") is None
    assert question_store.get_active_learner_question("missing", "content-v1") is None


def test_grade_active_learner_answer_uses_audited_option_and_keeps_tags_internal(
    tmp_path: Path,
) -> None:
    """练习判分必须使用发布审计选项，而不是客户端声称的对错或答案。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    candidate, review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1"),
    )
    question_store.publish(candidate, "publisher-a", review)

    correct = question_store.grade_active_learner_answer("q-1", "content-v1", "B")
    incorrect = question_store.grade_active_learner_answer("q-1", "content-v1", "A")

    assert correct is not None
    assert correct.is_correct is True
    assert correct.error_tags == ("invalid_converse",)
    assert correct.knowledge_tags == ("逆命题与逆否命题",)
    assert incorrect is not None
    assert incorrect.is_correct is False
    assert (
        question_store.grade_active_learner_answer("missing", "content-v1", "A")
        is None
    )
    with pytest.raises(ValueError, match="所选选项不属于该题目"):
        question_store.grade_active_learner_answer("q-1", "content-v1", "not-an-option")


@pytest.mark.parametrize(
    ("question_id", "content_version", "message"),
    [
        (" ", "content-v1", "题目标识不能为空"),
        ("q-1", " ", "内容版本不能为空"),
    ],
)
def test_get_active_learner_question_rejects_invalid_identifiers(
    tmp_path: Path,
    question_id: str,
    content_version: str,
    message: str,
) -> None:
    """学习者题目读取必须校验题号与内容版本，不能执行模糊查询。"""
    with pytest.raises(ValueError, match=message):
        _question_store(tmp_path).get_active_learner_question(
            question_id,
            content_version,
        )


def test_recommendation_rejects_invalid_limit(tmp_path: Path) -> None:
    """推荐数量越界时不应返回局部或隐式默认结果。"""
    with pytest.raises(ValueError, match="推荐数量"):
        _question_store(tmp_path).recommend(_profile(), (), limit=0)


def test_question_store_upgrades_v1_to_lifecycle_governance_schema(
    tmp_path: Path,
) -> None:
    """既有 v1 题库升级时只新增治理审计表，不改写已发布版本。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    candidate, review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1"),
    )
    published = question_store.publish(candidate, "publisher-a", review)
    with question_store._connect() as connection:
        connection.execute("DROP TABLE question_version_lifecycle_events")
        connection.execute("DELETE FROM schema_migrations WHERE version IN (2, 3)")

    upgraded_store = _question_store(tmp_path)

    assert upgraded_store.schema_version() == 3
    assert upgraded_store.active_questions() == (published,)
    assert upgraded_store.list_question_version_lifecycle_events("q-1") == ()
    with sqlite3.connect(tmp_path / "questions.sqlite3") as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(question_version_lifecycle_events)"
            )
        }
    assert columns == {
        "event_id",
        "question_id",
        "content_version",
        "content_hash",
        "action",
        "actor_id",
        "replaced_content_version",
        "reason",
        "created_at",
    }


def test_question_store_upgrades_v2_lifecycle_events_for_supersession(
    tmp_path: Path,
) -> None:
    """v2 生命周期事件在扩展动作约束时必须保持逐字段不变。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    candidate, review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1"),
    )
    published = question_store.publish(candidate, "publisher-a", review)
    deactivated = question_store.deactivate_active_version(
        published.question_id,
        published.content_version,
        actor_id="admin-a",
        reason="模拟既有 v2 下线事件",
    )
    reactivated = question_store.reactivate_published_version(
        published.question_id,
        published.content_version,
        actor_id="admin-b",
        reason="模拟既有 v2 重新激活事件",
        review=review,
    )
    assert deactivated is not None
    assert reactivated is not None

    with question_store._connect() as connection:
        connection.execute("DROP INDEX idx_question_version_lifecycle_events_q_created")
        connection.execute(
            """
            CREATE TABLE question_version_lifecycle_events_v2 (
                event_id TEXT PRIMARY KEY,
                question_id TEXT NOT NULL,
                content_version TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                action TEXT NOT NULL CHECK (action IN ('deactivated', 'reactivated')),
                actor_id TEXT NOT NULL,
                replaced_content_version TEXT,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO question_version_lifecycle_events_v2 (
                event_id, question_id, content_version, content_hash, action, actor_id,
                replaced_content_version, reason, created_at
            )
            SELECT event_id, question_id, content_version, content_hash, action,
                   actor_id, replaced_content_version, reason, created_at
            FROM question_version_lifecycle_events
            ORDER BY rowid ASC
            """
        )
        connection.execute("DROP TABLE question_version_lifecycle_events")
        connection.execute(
            """
            ALTER TABLE question_version_lifecycle_events_v2
            RENAME TO question_version_lifecycle_events
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_question_version_lifecycle_events_q_created
            ON question_version_lifecycle_events (question_id, created_at)
            """
        )
        connection.execute("DELETE FROM schema_migrations WHERE version = 3")

    upgraded_store = _question_store(tmp_path)

    assert upgraded_store.schema_version() == 3
    assert upgraded_store.list_question_version_lifecycle_events("q-1") == (
        deactivated,
        reactivated,
    )
    with sqlite3.connect(tmp_path / "questions.sqlite3") as connection:
        table_sql = connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'table' AND name = 'question_version_lifecycle_events'
            """
        ).fetchone()[0]
    assert "'superseded'" in table_sql


def test_question_store_restores_candidate_publication_and_audit_snapshot(
    tmp_path: Path,
) -> None:
    """题库恢复必须同时回退候选、活动版本及形式化验证审计。"""
    question_store = _question_store(tmp_path)
    review_store = _review_store(tmp_path)
    first_candidate, first_review = _approved_review(
        review_store,
        question_store,
        _publication("q-1", "content-v1"),
    )
    first_published = question_store.publish(
        first_candidate,
        "publisher-a",
        first_review,
    )
    first_deactivation = question_store.deactivate_active_version(
        "q-1",
        "content-v1",
        actor_id="admin-a",
        reason="验证备份中的下线审计事件",
    )
    first_reactivation = question_store.reactivate_published_version(
        "q-1",
        "content-v1",
        actor_id="admin-a",
        reason="验证备份中的重新激活审计事件",
        review=first_review,
    )
    assert first_deactivation is not None
    assert first_reactivation is not None
    backup = question_store.create_backup(tmp_path / "backups")
    second_candidate, second_review = _approved_review(
        review_store,
        question_store,
        _publication("q-2", "content-v1"),
    )
    question_store.publish(second_candidate, "publisher-a", second_review)

    question_store.restore_backup(question_store.load_backup(backup.manifest_path))

    assert question_store.schema_version() == 3
    assert question_store.active_questions() == (first_published,)
    assert question_store.list_question_version_lifecycle_events("q-1") == (
        first_deactivation,
        first_reactivation,
    )
    verification = question_store.get_formalization_verification(
        first_published.question_id,
        first_published.content_version,
        first_published.content_hash,
    )
    assert verification is not None
    assert verification.selected_option == "B"


def test_question_store_rejects_backup_from_another_database(tmp_path: Path) -> None:
    """不同题库即使结构相同，也不得接受彼此的备份恢复。"""
    source_store = QuestionBankStore(tmp_path / "source.sqlite3")
    target_store = QuestionBankStore(tmp_path / "target.sqlite3")
    backup = source_store.create_backup(tmp_path / "backups")

    with pytest.raises(ValueError, match="不属于当前数据库"):
        target_store.restore_backup(backup)
