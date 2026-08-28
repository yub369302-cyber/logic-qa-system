"""内容绑定审核题库发布与个人练习推荐的回归测试。"""

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
        attempted_question_ids=("q-1",),
        limit=3,
    )

    assert len(recommendations) == 1
    recommendation = recommendations[0]
    assert recommendation.question.question_id == "q-2"
    assert recommendation.matched_tags == ("invalid_converse", "逆命题与逆否命题")
    assert "高频错因" in recommendation.reason
    assert not hasattr(recommendation.question, "content_hash")
    assert not hasattr(recommendation.question, "error_tags")


def test_recommendation_rejects_invalid_limit(tmp_path: Path) -> None:
    """推荐数量越界时不应返回局部或隐式默认结果。"""
    with pytest.raises(ValueError, match="推荐数量"):
        _question_store(tmp_path).recommend(_profile(), (), limit=0)


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
    backup = question_store.create_backup(tmp_path / "backups")
    second_candidate, second_review = _approved_review(
        review_store,
        question_store,
        _publication("q-2", "content-v1"),
    )
    question_store.publish(second_candidate, "publisher-a", second_review)

    question_store.restore_backup(question_store.load_backup(backup.manifest_path))

    assert question_store.schema_version() == 1
    assert question_store.active_questions() == (first_published,)
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
