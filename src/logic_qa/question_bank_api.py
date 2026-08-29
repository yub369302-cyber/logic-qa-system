"""题目候选、审核与发布的管理端 API 路由。"""

from __future__ import annotations

from typing import Annotated
from typing import Literal as TypeLiteral

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from logic_qa.api_security import AdminIdentity
from logic_qa.grouping_matching_solver import (
    GroupConstraint,
    GroupConstraintType,
    GroupingSolveStatus,
    MatchConstraint,
    MatchConstraintType,
    MatchingSolveStatus,
)
from logic_qa.learning_profile import (
    LearningProfileStore,
    PracticeCorrectionRepublication,
    PracticeCorrectionRepublicationAlreadyLinkedError,
    PracticeCorrectionRepublicationNotEligibleError,
    PracticeCorrectionRepublicationVersionAlreadyLinkedError,
)
from logic_qa.models import VerificationStatus
from logic_qa.ordering_solver import (
    OrderingConstraint,
    OrderingConstraintType,
    OrderingSolveStatus,
)
from logic_qa.quality_operations import (
    QuestionReviewInput,
    QuestionReviewRecord,
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
    PublishedQuestion,
    QuestionBankStore,
    QuestionCandidate,
    QuestionFormalization,
    QuestionPublicationInput,
    QuestionVersionLifecycleAction,
    QuestionVersionLifecycleEvent,
)

router = APIRouter(prefix="/v1/admin", tags=["question-bank"])


class OrderingConstraintInput(BaseModel):
    """排序题的一条结构化约束。"""

    constraint_type: OrderingConstraintType
    item: str
    other_item: str | None = None
    position: int | None = None


class GroupConstraintInput(BaseModel):
    """分组题的一条结构化约束。"""

    constraint_type: GroupConstraintType
    item: str
    other_item: str


class MatchConstraintInput(BaseModel):
    """一对一匹配题的一条结构化约束。"""

    constraint_type: MatchConstraintType
    item: str
    target: str


class QuestionReviewRequest(BaseModel):
    """管理员对精确候选内容提交的审核结论。"""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    content_version: str
    content_hash: str
    status: QuestionReviewStatus
    verified_answer: str | None = None
    formalization_version: str
    notes: str | None = None


class FormalizationRuleInput(BaseModel):
    """可复现命题形式化资产的一条单前提蕴含规则。"""

    model_config = ConfigDict(extra="forbid")

    premise: str
    conclusion: str
    source_text: str | None = None


class OptionAssertionInput(BaseModel):
    """一个选项与其可复现求解结果之间的结构化断言。"""

    model_config = ConfigDict(extra="forbid")

    option: str
    claim_status: str
    claim_solution_count: int | None = Field(default=None, ge=0)


class PropositionalFormalizationInput(BaseModel):
    """候选题绑定的结构化命题资产和预期验证结论。"""

    model_config = ConfigDict(extra="forbid")

    kind: TypeLiteral["propositional"]
    facts: list[str] = Field(default_factory=list)
    rules: list[FormalizationRuleInput] = Field(default_factory=list)
    query: str
    expected_status: VerificationStatus
    expected_answer: str
    option_assertions: list[OptionAssertionInput]


class OrderingFormalizationInput(BaseModel):
    """候选排序题的对象、约束与完整枚举预期。"""

    model_config = ConfigDict(extra="forbid")

    kind: TypeLiteral["ordering"]
    items: list[str]
    constraints: list[OrderingConstraintInput] = Field(default_factory=list)
    expected_status: OrderingSolveStatus
    expected_solution_count: int = Field(ge=0)
    expected_answer: str
    option_assertions: list[OptionAssertionInput]


class GroupingFormalizationInput(BaseModel):
    """候选分组题的对象、组别、容量和完整枚举预期。"""

    model_config = ConfigDict(extra="forbid")

    kind: TypeLiteral["grouping"]
    items: list[str]
    groups: list[str]
    max_group_size: int = Field(ge=1)
    constraints: list[GroupConstraintInput] = Field(default_factory=list)
    expected_status: GroupingSolveStatus
    expected_solution_count: int = Field(ge=0)
    expected_answer: str
    option_assertions: list[OptionAssertionInput]


class MatchingFormalizationInput(BaseModel):
    """候选匹配题的对象、目标、约束与完整枚举预期。"""

    model_config = ConfigDict(extra="forbid")

    kind: TypeLiteral["matching"]
    items: list[str]
    targets: list[str]
    constraints: list[MatchConstraintInput] = Field(default_factory=list)
    expected_status: MatchingSolveStatus
    expected_solution_count: int = Field(ge=0)
    expected_answer: str
    option_assertions: list[OptionAssertionInput]


type QuestionFormalizationInput = (
    PropositionalFormalizationInput
    | OrderingFormalizationInput
    | GroupingFormalizationInput
    | MatchingFormalizationInput
)


class QuestionPublicationRequest(BaseModel):
    """管理员提交候选或发布已审核题目版本的请求。"""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    content_version: str
    question_type: str
    stem: str
    options: list[str] = Field(default_factory=list)
    error_tags: list[str] = Field(default_factory=list)
    knowledge_tags: list[str] = Field(default_factory=list)
    formalization_version: str
    formalization: QuestionFormalizationInput = Field(discriminator="kind")


class QuestionReviewResponse(BaseModel):
    """管理员可见的精确候选内容审核状态。"""

    question_id: str
    content_version: str
    content_hash: str
    reviewer_id: str
    status: QuestionReviewStatus
    verified_answer: str | None
    formalization_version: str
    notes: str | None
    updated_at: str


class FormalizationRuleResponse(BaseModel):
    """管理员可回查的形式化规则。"""

    premise: str
    conclusion: str
    source_text: str | None


class OptionAssertionResponse(BaseModel):
    """管理员可回查的选项语义断言。"""

    option: str
    claim_status: str
    claim_solution_count: int | None


class PropositionalFormalizationResponse(BaseModel):
    """管理员可见的规范化命题逻辑资产。"""

    kind: TypeLiteral["propositional"] = "propositional"
    facts: list[str]
    rules: list[FormalizationRuleResponse]
    query: str
    expected_status: VerificationStatus
    expected_answer: str
    option_assertions: list[OptionAssertionResponse]


class OrderingFormalizationResponse(BaseModel):
    """管理员可见的规范化排序资产。"""

    kind: TypeLiteral["ordering"] = "ordering"
    items: list[str]
    constraints: list[OrderingConstraintInput]
    expected_status: OrderingSolveStatus
    expected_solution_count: int
    expected_answer: str
    option_assertions: list[OptionAssertionResponse]


class GroupingFormalizationResponse(BaseModel):
    """管理员可见的规范化分组资产。"""

    kind: TypeLiteral["grouping"] = "grouping"
    items: list[str]
    groups: list[str]
    max_group_size: int
    constraints: list[GroupConstraintInput]
    expected_status: GroupingSolveStatus
    expected_solution_count: int
    expected_answer: str
    option_assertions: list[OptionAssertionResponse]


class MatchingFormalizationResponse(BaseModel):
    """管理员可见的规范化匹配资产。"""

    kind: TypeLiteral["matching"] = "matching"
    items: list[str]
    targets: list[str]
    constraints: list[MatchConstraintInput]
    expected_status: MatchingSolveStatus
    expected_solution_count: int
    expected_answer: str
    option_assertions: list[OptionAssertionResponse]


type QuestionFormalizationResponse = (
    PropositionalFormalizationResponse
    | OrderingFormalizationResponse
    | GroupingFormalizationResponse
    | MatchingFormalizationResponse
)


class QuestionCandidateResponse(BaseModel):
    """管理员用于审核的规范化候选内容及其 SHA-256 摘要。"""

    question_id: str
    content_version: str
    question_type: str
    stem: str
    options: list[str]
    error_tags: list[str]
    knowledge_tags: list[str]
    formalization_version: str
    formalization: QuestionFormalizationResponse = Field(discriminator="kind")
    content_hash: str


class AdminPublishedQuestionResponse(QuestionCandidateResponse):
    """管理员可见的已发布版本，保留内部标签和内容摘要。"""

    published_at: str


class QuestionVersionLifecycleRequest(BaseModel):
    """管理员触发下线或历史版本重新激活时必须提交的治理理由。"""

    model_config = ConfigDict(extra="forbid")

    reason: str


class QuestionVersionLifecycleEventResponse(BaseModel):
    """管理员可回查的版本下线或重新激活审计事实。"""

    event_id: str
    question_id: str
    content_version: str
    content_hash: str
    action: QuestionVersionLifecycleAction
    actor_id: str
    replaced_content_version: str | None
    reason: str
    created_at: str


class CorrectionRepublicationLinkRequest(BaseModel):
    """管理员将需要重新发布的申请绑定到已通过门禁的新发布版本。"""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    previous_content_version: str


class CorrectionRepublicationLinkResponse(BaseModel):
    """管理员可审计的复核申请与新发布版本关联。"""

    request_id: str
    question_id: str
    previous_content_version: str
    new_content_version: str
    new_content_hash: str
    linked_by: str
    linked_at: str


class QuestionBankDependencies:
    """题库管理路由所需的可替换存储依赖。"""

    def __init__(
        self,
        question_bank_store: QuestionBankStore,
        review_store: QuestionReviewStore,
        learning_store: LearningProfileStore,
    ) -> None:
        self.question_bank_store = question_bank_store
        self.review_store = review_store
        self.learning_store = learning_store


def get_question_bank_dependencies(request: Request) -> QuestionBankDependencies:
    """从应用状态获取当前请求使用的题库和审核存储。"""
    app_state = request.app.state
    return QuestionBankDependencies(
        question_bank_store=app_state.question_bank_store,
        review_store=app_state.review_store,
        learning_store=app_state.learning_store,
    )


QuestionBankDependenciesInput = Annotated[
    QuestionBankDependencies,
    Depends(get_question_bank_dependencies),
]


@router.post("/question-candidates", response_model=QuestionCandidateResponse)
def prepare_question_candidate(
    request: QuestionPublicationRequest,
    _: AdminIdentity,
    dependencies: QuestionBankDependenciesInput,
) -> QuestionCandidateResponse:
    """提交不可变候选快照并返回其唯一内容摘要，供审核精确绑定。"""
    try:
        candidate = dependencies.question_bank_store.submit_candidate(
            _publication_from_request(request)
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _question_candidate_to_response(candidate)


@router.get(
    "/question-candidates/{question_id}/{content_version}/{content_hash}",
    response_model=QuestionCandidateResponse,
)
def get_question_candidate(
    question_id: str,
    content_version: str,
    content_hash: str,
    _: AdminIdentity,
    dependencies: QuestionBankDependenciesInput,
) -> QuestionCandidateResponse:
    """精确回查已持久化的候选内容，供审核与发布链路核验。"""
    try:
        candidate = dependencies.question_bank_store.get_candidate(
            question_id,
            content_version,
            content_hash,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if candidate is None:
        raise HTTPException(status_code=404, detail="未找到该候选内容快照")
    return _question_candidate_to_response(candidate)


@router.post("/question-reviews", response_model=QuestionReviewResponse)
def upsert_question_review(
    request: QuestionReviewRequest,
    identity: AdminIdentity,
    dependencies: QuestionBankDependenciesInput,
) -> QuestionReviewResponse:
    """仅为已保存且形式化版本一致的候选内容写入审核结论。"""
    try:
        candidate = dependencies.question_bank_store.get_candidate(
            request.question_id,
            request.content_version,
            request.content_hash,
        )
        if candidate is None:
            raise ValueError("候选内容不存在，请先提交候选题目")
        if (
            request.formalization_version.strip()
            != candidate.publication.formalization_version
        ):
            raise ValueError("审核形式化版本与候选内容不一致")
        record = dependencies.review_store.upsert_review(
            QuestionReviewInput(
                question_id=request.question_id,
                content_version=request.content_version,
                content_hash=request.content_hash,
                reviewer_id=identity.subject,
                status=request.status,
                verified_answer=request.verified_answer,
                formalization_version=request.formalization_version,
                notes=request.notes,
            )
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _question_review_to_response(record)


@router.get(
    "/question-reviews/{question_id}/{content_version}/{content_hash}",
    response_model=QuestionReviewResponse,
)
def get_question_review(
    question_id: str,
    content_version: str,
    content_hash: str,
    _: AdminIdentity,
    dependencies: QuestionBankDependenciesInput,
) -> QuestionReviewResponse:
    """读取与题号、版本和内容摘要完全匹配的审核结论。"""
    try:
        record = dependencies.review_store.get_review(
            question_id,
            content_version,
            content_hash,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if record is None:
        raise HTTPException(status_code=404, detail="未找到该候选内容的审核记录")
    return _question_review_to_response(record)


@router.post("/questions", response_model=AdminPublishedQuestionResponse)
def publish_question(
    request: QuestionPublicationRequest,
    identity: AdminIdentity,
    dependencies: QuestionBankDependenciesInput,
) -> AdminPublishedQuestionResponse:
    """发布与审核绑定信息完全匹配的题目内容。"""
    try:
        candidate = dependencies.question_bank_store.prepare_candidate(
            _publication_from_request(request)
        )
        review = dependencies.review_store.get_review(
            candidate.publication.question_id,
            candidate.publication.content_version,
            candidate.content_hash,
        )
        question = dependencies.question_bank_store.publish(
            candidate,
            publisher_id=identity.subject,
            review=review,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _admin_published_question_to_response(question)


@router.post(
    "/questions/{question_id}/{content_version}/deactivation",
    response_model=QuestionVersionLifecycleEventResponse,
)
def deactivate_question_version(
    question_id: str,
    content_version: str,
    request: QuestionVersionLifecycleRequest,
    identity: AdminIdentity,
    dependencies: QuestionBankDependenciesInput,
) -> QuestionVersionLifecycleEventResponse:
    """仅下线当前活动版本；历史发布、验证与学习账本均不改写。"""
    try:
        event = dependencies.question_bank_store.deactivate_active_version(
            question_id,
            content_version,
            actor_id=identity.subject,
            reason=request.reason,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if event is None:
        raise HTTPException(status_code=404, detail="未找到已发布题目版本")
    return _question_version_lifecycle_event_to_response(event)


@router.post(
    "/questions/{question_id}/{content_version}/reactivation",
    response_model=QuestionVersionLifecycleEventResponse,
)
def reactivate_question_version(
    question_id: str,
    content_version: str,
    request: QuestionVersionLifecycleRequest,
    identity: AdminIdentity,
    dependencies: QuestionBankDependenciesInput,
) -> QuestionVersionLifecycleEventResponse:
    """仅在历史版本仍通过精确审核与确定性复验后重新激活。"""
    try:
        published = dependencies.question_bank_store.get_published_question(
            question_id,
            content_version,
        )
        if published is None:
            raise HTTPException(status_code=404, detail="未找到已发布题目版本")
        review = dependencies.review_store.get_review(
            published.question_id,
            published.content_version,
            published.content_hash,
        )
        event = dependencies.question_bank_store.reactivate_published_version(
            question_id,
            content_version,
            actor_id=identity.subject,
            reason=request.reason,
            review=review,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if event is None:
        raise HTTPException(status_code=404, detail="未找到已发布题目版本")
    return _question_version_lifecycle_event_to_response(event)


@router.get(
    "/questions/{question_id}/version-lifecycle-events",
    response_model=list[QuestionVersionLifecycleEventResponse],
)
def list_question_version_lifecycle_events(
    question_id: str,
    _: AdminIdentity,
    dependencies: QuestionBankDependenciesInput,
) -> list[QuestionVersionLifecycleEventResponse]:
    """回查同一题目全部下线与重新激活事件，不读取或修改学习账本。"""
    try:
        store = dependencies.question_bank_store
        events = store.list_question_version_lifecycle_events(question_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return [_question_version_lifecycle_event_to_response(event) for event in events]


@router.post(
    "/questions/{question_id}/{content_version}/correction-republication-links",
    response_model=CorrectionRepublicationLinkResponse,
)
def link_correction_republication(
    question_id: str,
    content_version: str,
    request: CorrectionRepublicationLinkRequest,
    identity: AdminIdentity,
    dependencies: QuestionBankDependenciesInput,
) -> CorrectionRepublicationLinkResponse:
    """只在新版本已发布且活动时，追加该版本与需要复核申请的关联。"""
    try:
        published = dependencies.question_bank_store.get_active_published_question(
            question_id,
            content_version,
        )
        if published is None:
            raise ValueError("未找到可关联的已发布新版本")
        republication = (
            dependencies.learning_store.link_practice_correction_republication(
                request_id=request.request_id,
                question_id=published.question_id,
                previous_content_version=request.previous_content_version,
                new_content_version=published.content_version,
                new_content_hash=published.content_hash,
                linked_by=identity.subject,
            )
        )
    except (
        PracticeCorrectionRepublicationAlreadyLinkedError,
        PracticeCorrectionRepublicationVersionAlreadyLinkedError,
    ) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except PracticeCorrectionRepublicationNotEligibleError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if republication is None:
        raise HTTPException(status_code=404, detail="未找到复核申请")
    return _correction_republication_link_to_response(republication)


def _publication_from_request(
    request: QuestionPublicationRequest,
) -> QuestionPublicationInput:
    """从管理员请求提取尚未持久化的题目内容。"""
    return QuestionPublicationInput(
        question_id=request.question_id,
        content_version=request.content_version,
        question_type=request.question_type,
        stem=request.stem,
        options=tuple(request.options),
        error_tags=tuple(request.error_tags),
        knowledge_tags=tuple(request.knowledge_tags),
        formalization_version=request.formalization_version,
        formalization=_formalization_from_request(request.formalization),
    )


def _formalization_from_request(
    formalization: QuestionFormalizationInput,
) -> QuestionFormalization:
    """将带类型判别器的 API 输入转换为领域形式化资产。"""
    match formalization:
        case PropositionalFormalizationInput():
            return PropositionalFormalization(
                facts=tuple(formalization.facts),
                rules=tuple(
                    FormalizationRule(
                        premise=rule.premise,
                        conclusion=rule.conclusion,
                        source_text=rule.source_text,
                    )
                    for rule in formalization.rules
                ),
                query=formalization.query,
                expected_status=formalization.expected_status,
                expected_answer=formalization.expected_answer,
                option_assertions=_option_assertions_from_request(
                    formalization.option_assertions
                ),
            )
        case OrderingFormalizationInput():
            return OrderingFormalization(
                items=tuple(formalization.items),
                constraints=tuple(
                    OrderingConstraint(
                        constraint_type=constraint.constraint_type,
                        item=constraint.item,
                        other_item=constraint.other_item,
                        position=constraint.position,
                    )
                    for constraint in formalization.constraints
                ),
                expected_status=formalization.expected_status,
                expected_solution_count=formalization.expected_solution_count,
                expected_answer=formalization.expected_answer,
                option_assertions=_option_assertions_from_request(
                    formalization.option_assertions
                ),
            )
        case GroupingFormalizationInput():
            return GroupingFormalization(
                items=tuple(formalization.items),
                groups=tuple(formalization.groups),
                max_group_size=formalization.max_group_size,
                constraints=tuple(
                    GroupConstraint(
                        constraint_type=constraint.constraint_type,
                        item=constraint.item,
                        other_item=constraint.other_item,
                    )
                    for constraint in formalization.constraints
                ),
                expected_status=formalization.expected_status,
                expected_solution_count=formalization.expected_solution_count,
                expected_answer=formalization.expected_answer,
                option_assertions=_option_assertions_from_request(
                    formalization.option_assertions
                ),
            )
        case MatchingFormalizationInput():
            return MatchingFormalization(
                items=tuple(formalization.items),
                targets=tuple(formalization.targets),
                constraints=tuple(
                    MatchConstraint(
                        constraint_type=constraint.constraint_type,
                        item=constraint.item,
                        target=constraint.target,
                    )
                    for constraint in formalization.constraints
                ),
                expected_status=formalization.expected_status,
                expected_solution_count=formalization.expected_solution_count,
                expected_answer=formalization.expected_answer,
                option_assertions=_option_assertions_from_request(
                    formalization.option_assertions
                ),
            )
    raise ValueError("候选题目形式化资产类型不受支持")


def _option_assertions_from_request(
    assertions: list[OptionAssertionInput],
) -> tuple[OptionAssertion, ...]:
    """将管理员输入的选项语义断言转换为不可变领域值。"""
    return tuple(
        OptionAssertion(
            option=assertion.option,
            claim_status=assertion.claim_status,
            claim_solution_count=assertion.claim_solution_count,
        )
        for assertion in assertions
    )


def _formalization_to_response(
    formalization: QuestionFormalization | None,
) -> QuestionFormalizationResponse:
    """将已规范化的形式化资产转换为管理员可审计响应。"""
    if formalization is None:
        raise ValueError("候选题目缺少可复现的形式化资产")
    match formalization:
        case PropositionalFormalization():
            return PropositionalFormalizationResponse(
                facts=list(formalization.facts),
                rules=[
                    FormalizationRuleResponse(
                        premise=rule.premise,
                        conclusion=rule.conclusion,
                        source_text=rule.source_text,
                    )
                    for rule in formalization.rules
                ],
                query=formalization.query,
                expected_status=formalization.expected_status,
                expected_answer=formalization.expected_answer,
                option_assertions=_option_assertions_to_response(
                    formalization.option_assertions
                ),
            )
        case OrderingFormalization():
            return OrderingFormalizationResponse(
                items=list(formalization.items),
                constraints=[
                    OrderingConstraintInput(
                        constraint_type=constraint.constraint_type,
                        item=constraint.item,
                        other_item=constraint.other_item,
                        position=constraint.position,
                    )
                    for constraint in formalization.constraints
                ],
                expected_status=formalization.expected_status,
                expected_solution_count=formalization.expected_solution_count,
                expected_answer=formalization.expected_answer,
                option_assertions=_option_assertions_to_response(
                    formalization.option_assertions
                ),
            )
        case GroupingFormalization():
            return GroupingFormalizationResponse(
                items=list(formalization.items),
                groups=list(formalization.groups),
                max_group_size=formalization.max_group_size,
                constraints=[
                    GroupConstraintInput(
                        constraint_type=constraint.constraint_type,
                        item=constraint.item,
                        other_item=constraint.other_item,
                    )
                    for constraint in formalization.constraints
                ],
                expected_status=formalization.expected_status,
                expected_solution_count=formalization.expected_solution_count,
                expected_answer=formalization.expected_answer,
                option_assertions=_option_assertions_to_response(
                    formalization.option_assertions
                ),
            )
        case MatchingFormalization():
            return MatchingFormalizationResponse(
                items=list(formalization.items),
                targets=list(formalization.targets),
                constraints=[
                    MatchConstraintInput(
                        constraint_type=constraint.constraint_type,
                        item=constraint.item,
                        target=constraint.target,
                    )
                    for constraint in formalization.constraints
                ],
                expected_status=formalization.expected_status,
                expected_solution_count=formalization.expected_solution_count,
                expected_answer=formalization.expected_answer,
                option_assertions=_option_assertions_to_response(
                    formalization.option_assertions
                ),
            )
    raise ValueError("候选题目形式化资产类型不受支持")


def _option_assertions_to_response(
    assertions: tuple[OptionAssertion, ...],
) -> list[OptionAssertionResponse]:
    """将不可变选项语义断言转换为管理端响应。"""
    return [
        OptionAssertionResponse(
            option=assertion.option,
            claim_status=assertion.claim_status,
            claim_solution_count=assertion.claim_solution_count,
        )
        for assertion in assertions
    ]


def _question_version_lifecycle_event_to_response(
    event: QuestionVersionLifecycleEvent,
) -> QuestionVersionLifecycleEventResponse:
    """将不可变版本状态变更事件转换为管理员响应。"""
    return QuestionVersionLifecycleEventResponse(
        event_id=event.event_id,
        question_id=event.question_id,
        content_version=event.content_version,
        content_hash=event.content_hash,
        action=event.action,
        actor_id=event.actor_id,
        replaced_content_version=event.replaced_content_version,
        reason=event.reason,
        created_at=event.created_at,
    )


def _correction_republication_link_to_response(
    republication: PracticeCorrectionRepublication,
) -> CorrectionRepublicationLinkResponse:
    """返回管理员可审计的关联，不把学习记录或题目答案写入关联。"""
    return CorrectionRepublicationLinkResponse(
        request_id=republication.request_id,
        question_id=republication.question_id,
        previous_content_version=republication.previous_content_version,
        new_content_version=republication.new_content_version,
        new_content_hash=republication.new_content_hash,
        linked_by=republication.linked_by,
        linked_at=republication.linked_at,
    )


def _question_candidate_to_response(
    candidate: QuestionCandidate,
) -> QuestionCandidateResponse:
    """将内部候选内容转换为管理员审核前可见的摘要响应。"""
    publication = candidate.publication
    return QuestionCandidateResponse(
        question_id=publication.question_id,
        content_version=publication.content_version,
        question_type=publication.question_type,
        stem=publication.stem,
        options=list(publication.options),
        error_tags=list(publication.error_tags),
        knowledge_tags=list(publication.knowledge_tags),
        formalization_version=publication.formalization_version,
        formalization=_formalization_to_response(publication.formalization),
        content_hash=candidate.content_hash,
    )


def _admin_published_question_to_response(
    question: PublishedQuestion,
) -> AdminPublishedQuestionResponse:
    """将已发布内部记录转换为管理员可见的完整版本元数据。"""
    return AdminPublishedQuestionResponse(
        question_id=question.question_id,
        content_version=question.content_version,
        question_type=question.question_type,
        stem=question.stem,
        options=list(question.options),
        error_tags=list(question.error_tags),
        knowledge_tags=list(question.knowledge_tags),
        formalization_version=question.formalization_version,
        formalization=_formalization_to_response(question.formalization),
        content_hash=question.content_hash,
        published_at=question.published_at,
    )


def _question_review_to_response(
    record: QuestionReviewRecord,
) -> QuestionReviewResponse:
    """将精确内容审核记录转换为管理员响应。"""
    return QuestionReviewResponse(
        question_id=record.question_id,
        content_version=record.content_version,
        content_hash=record.content_hash,
        reviewer_id=record.reviewer_id,
        status=record.status,
        verified_answer=record.verified_answer,
        formalization_version=record.formalization_version,
        notes=record.notes,
        updated_at=record.updated_at,
    )
