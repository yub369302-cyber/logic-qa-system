"""FastAPI 接口：暴露第一阶段的结构化命题逻辑验证能力。"""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from logic_qa.api_security import AdminIdentity, CurrentIdentity
from logic_qa.chinese_parser import (
    ChineseConfirmationRequest,
    ChineseParseError,
    ConfirmedChineseSentence,
    ControlledChineseParser,
    ParsedChineseText,
)
from logic_qa.choice_verifier import (
    ChoiceQuestionType,
    ChoiceVerificationResult,
    ChoiceVerificationStatus,
    ChoiceVerifier,
)
from logic_qa.engine import InferenceEngine
from logic_qa.grouping_matching_solver import (
    GroupConstraint,
    GroupConstraintType,
    GroupingSolver,
    GroupingSolveResult,
    GroupingSolveStatus,
    MatchConstraint,
    MatchConstraintType,
    MatchingSolver,
    MatchingSolveResult,
    MatchingSolveStatus,
)
from logic_qa.learning_profile import (
    LearningProfile,
    LearningProfileStore,
    LearningRecord,
    LearningRecordInput,
)
from logic_qa.models import ImplicationRule, Literal, VerificationResult
from logic_qa.ocr_service import (
    OcrCorrectionResult,
    OcrCorrectionService,
    OcrExtraction,
    OcrService,
    OcrUnavailableError,
)
from logic_qa.ordering_solver import (
    OrderingConstraint,
    OrderingConstraintType,
    OrderingSolver,
    OrderingSolveResult,
    OrderingSolveStatus,
)
from logic_qa.quality_operations import (
    QuestionReviewStore,
    ReviewDashboard,
    RuntimeMetrics,
    RuntimeMetricsSnapshot,
)
from logic_qa.question_bank import (
    LearnerQuestion,
    PracticeRecommendation,
    QuestionBankStore,
)
from logic_qa.question_bank_api import (
    QuestionBankDependencies,
    get_question_bank_dependencies,
)
from logic_qa.question_bank_api import router as question_bank_router
from logic_qa.solution_reviewer import (
    ReasoningDirection,
    ReasoningStep,
    ReviewDiagnostic,
    SolutionReviewer,
    SolutionReviewResult,
)

app = FastAPI(
    title="逻辑答疑系统 MVP",
    version="0.1.0",
    description=(
        "第一阶段仅接收结构化命题逻辑条件，并返回可验证的推理链。"
        "自然语言解析将在下一阶段接入。"
    ),
)
static_directory = Path(__file__).resolve().parent / "static"
app.mount("/assets", StaticFiles(directory=static_directory), name="assets")
engine = InferenceEngine()
choice_verifier = ChoiceVerifier()
chinese_parser = ControlledChineseParser()
ordering_solver = OrderingSolver()
grouping_solver = GroupingSolver()
matching_solver = MatchingSolver()
ocr_service = OcrService()
ocr_correction_service = OcrCorrectionService()
solution_reviewer = SolutionReviewer(engine=engine)
learning_store = LearningProfileStore(
    Path(
        os.environ.get(
            "LOGIC_QA_DATABASE_PATH",
            Path(__file__).resolve().parents[2] / "data" / "learning.sqlite3",
        )
    )
)
review_store = QuestionReviewStore(
    Path(
        os.environ.get(
            "LOGIC_QA_REVIEW_DATABASE_PATH",
            Path(__file__).resolve().parents[2] / "data" / "reviews.sqlite3",
        )
    )
)
question_bank_store = QuestionBankStore(
    Path(
        os.environ.get(
            "LOGIC_QA_QUESTION_DATABASE_PATH",
            Path(__file__).resolve().parents[2] / "data" / "questions.sqlite3",
        )
    )
)


def _question_bank_dependencies_override() -> QuestionBankDependencies:
    """兼容应用级存储替换，并为题库路由提供当前依赖。"""
    return QuestionBankDependencies(
        question_bank_store=question_bank_store,
        review_store=review_store,
    )


runtime_metrics = RuntimeMetrics()
app.state.learning_store = learning_store
app.state.review_store = review_store
app.state.question_bank_store = question_bank_store
app.dependency_overrides[get_question_bank_dependencies] = (
    _question_bank_dependencies_override
)
app.include_router(question_bank_router)


class RuleInput(BaseModel):
    """单前提蕴含规则的 API 入参。"""

    premise: str = Field(description="前提文字，例如 A 或 !A")
    conclusion: str = Field(description="结论文字，例如 B 或 !B")
    source_text: str | None = Field(
        default=None,
        description="题干中的原始条件，用于在解析中回溯依据",
    )


class SolveRequest(BaseModel):
    """结构化逻辑题的求解请求。"""

    facts: list[str] = Field(default_factory=list, description="已知事实")
    rules: list[RuleInput] = Field(default_factory=list, description="蕴含规则")
    query: str = Field(description="需要验证的结论")

    @field_validator("facts", mode="after")
    @classmethod
    def reject_duplicate_facts(cls, facts: list[str]) -> list[str]:
        """保留用户输入顺序的同时避免重复事实干扰阅读。"""
        return list(dict.fromkeys(facts))


class ConfirmedChineseRuleInput(BaseModel):
    """人工确认的单前提蕴含规则。"""

    premise: str
    conclusion: str


class ChineseConfirmationInput(BaseModel):
    """将一条复杂中文条件人工转写为结构化事实或规则。"""

    source_sentence: str
    facts: list[str] = Field(default_factory=list)
    rules: list[ConfirmedChineseRuleInput] = Field(default_factory=list)


class ChineseSolveRequest(BaseModel):
    """中文题干及其可选人工确认转写的求解请求。"""

    conditions: str = Field(
        description="由句号、分号或换行分隔的中文条件",
    )
    query: str = Field(description="需要验证的单个中文命题，不带句末标点")
    confirmations: list[ChineseConfirmationInput] = Field(default_factory=list)


class ChoiceVerifyRequest(BaseModel):
    """结构化命题逻辑选择题单个选项的验证请求。"""

    facts: list[str] = Field(default_factory=list, description="已知事实")
    rules: list[RuleInput] = Field(default_factory=list, description="蕴含规则")
    question_type: ChoiceQuestionType = Field(description="选择题设问类型")
    option: str = Field(description="需要验证的单个选项命题")


class ParsedRuleOutput(BaseModel):
    """中文题干解析出的规则，供用户在结论前核对。"""

    premise: str
    conclusion: str
    source_text: str


class OrderingConstraintInput(BaseModel):
    """排序题的单条结构化约束。"""

    constraint_type: OrderingConstraintType
    item: str
    other_item: str | None = None
    position: int | None = None


class GroupConstraintInput(BaseModel):
    """分组题中要求同组或不同组的关系约束。"""

    constraint_type: GroupConstraintType
    item: str
    other_item: str


class GroupingSolveRequest(BaseModel):
    """分组题的对象、分组、容量和结构化关系约束。"""

    items: list[str] = Field(description="待分组的唯一对象列表")
    groups: list[str] = Field(description="可分配的唯一分组名称")
    max_group_size: int = Field(description="每个分组的最大容量")
    constraints: list[GroupConstraintInput] = Field(
        default_factory=list,
        description="同组或不同组约束",
    )


class MatchConstraintInput(BaseModel):
    """一对一匹配题的固定或禁止配对约束。"""

    constraint_type: MatchConstraintType
    item: str
    target: str


class MatchingSolveRequest(BaseModel):
    """一对一匹配题的对象、目标及结构化约束。"""

    items: list[str] = Field(description="需要匹配的唯一对象列表")
    targets: list[str] = Field(description="可匹配的唯一目标列表")
    constraints: list[MatchConstraintInput] = Field(
        default_factory=list,
        description="固定或禁止匹配约束",
    )


class OrderingSolveRequest(BaseModel):
    """排序题的对象列表和结构化约束。"""

    items: list[str] = Field(description="待排序的唯一对象列表")
    constraints: list[OrderingConstraintInput] = Field(
        default_factory=list,
        description="先后、相邻、不相邻或固定位置约束",
    )


class OcrExtractRequest(BaseModel):
    """图片 OCR 请求，仅接受 Base64 数据而不接受文件路径。"""

    image_base64: str = Field(description="PNG、JPEG 或 WebP 图片的 Base64 或 Data URL")


class OcrCorrectionRequest(BaseModel):
    """用户提交 OCR 原文与校正后文本的核对请求。"""

    original_text: str = Field(description="OCR 返回的原始文本")
    corrected_text: str = Field(description="用户确认或修订后的文本")


class LearningRecordCreateRequest(BaseModel):
    """当前认证用户主动提交的最小学习记录。"""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question_type: str
    is_correct: bool
    error_tags: list[str] = Field(default_factory=list)
    knowledge_tags: list[str] = Field(default_factory=list)
    duration_seconds: int | None = None


class ReasoningStepInput(BaseModel):
    """用户提交的一步结构化命题逻辑推理。"""

    rule_id: int = Field(description="从 1 开始计数的原始规则编号")
    direction: ReasoningDirection
    premise: str
    conclusion: str


class SolutionReviewRequest(BaseModel):
    """结构化题干、目标结论和用户推理步骤的批改请求。"""

    facts: list[str] = Field(default_factory=list, description="已知事实")
    rules: list[RuleInput] = Field(default_factory=list, description="蕴含规则")
    target: str = Field(description="用户需要证明的目标结论")
    steps: list[ReasoningStepInput] = Field(
        default_factory=list,
        description="推理步骤",
    )


class ProofStepOutput(BaseModel):
    """单条证明步骤的响应模型。"""

    derived: str
    reason: str
    rule: str | None = None
    source_text: str | None = None
    dependencies: list[str]


class SolveResponse(BaseModel):
    """求解接口的稳定响应模型。"""

    query: str
    status: str
    conclusion: str
    verification_level: str
    proof_steps: list[ProofStepOutput]
    known_literals: list[str]
    conflict: list[str] | None = None


class ChineseConfirmationRequestOutput(BaseModel):
    """需要人工确认的复杂中文语义及其风险说明。"""

    source_sentence: str
    codes: list[str]
    message: str


class ChineseSolveResponse(SolveResponse):
    """中文题干接口的响应，额外回显解析和确认状态。"""

    parsed_facts: list[str]
    parsed_rules: list[ParsedRuleOutput]
    source_sentences: list[str]
    confirmation_required: bool
    confirmation_requests: list[ChineseConfirmationRequestOutput]


class ChoiceVerifyResponse(BaseModel):
    """选择题选项验证的稳定响应模型。"""

    question_type: ChoiceQuestionType
    option: str
    status: ChoiceVerificationStatus
    conclusion: str
    verification_level: str
    enumeration_status: str
    model_count: int
    witness_type: str | None = None
    witness_model: list[str] | None = None


class OrderingSolveResponse(BaseModel):
    """排序题完整求解状态和展示样例。"""

    status: OrderingSolveStatus
    conclusion: str
    verification_level: str
    solution_count: int
    sample_solutions: list[list[str]]


class GroupingSolveResponse(BaseModel):
    """分组题完整求解状态和展示样例。"""

    status: GroupingSolveStatus
    conclusion: str
    verification_level: str
    solution_count: int
    sample_solutions: list[dict[str, str]]


class MatchingSolveResponse(BaseModel):
    """匹配题完整求解状态和展示样例。"""

    status: MatchingSolveStatus
    conclusion: str
    verification_level: str
    solution_count: int
    sample_solutions: list[dict[str, str]]


class OcrExtractResponse(BaseModel):
    """OCR 原始文本、关键逻辑词提示和 Provider 信息。"""

    provider: str
    image_type: str
    text: str
    warnings: list[str]
    critical_terms: list[str]


class OcrCorrectionResponse(BaseModel):
    """校正文案及关键逻辑词变更确认要求。"""

    corrected_text: str
    requires_confirmation: bool
    changed_critical_terms: list[str]
    warnings: list[str]


class ReviewDashboardResponse(BaseModel):
    """审核状态看板的最小聚合输出。"""

    total_questions: int
    status_counts: list[tuple[str, int]]
    total_review_events: int


class RuntimeMetricsResponse(BaseModel):
    """不含请求正文与用户信息的运行指标输出。"""

    total_requests: int
    error_requests: int
    average_latency_ms: float
    route_counts: list[tuple[str, int]]
    status_counts: list[tuple[str, int]]


class LearnerQuestionResponse(BaseModel):
    """学习者可见的最小题目展示内容。"""

    question_id: str
    content_version: str
    question_type: str
    stem: str
    options: list[str]


class PracticeRecommendationResponse(BaseModel):
    """针对当前用户的真实题库练习推荐。"""

    question: LearnerQuestionResponse
    reason: str


class LearningRecordResponse(BaseModel):
    """当前认证用户已持久化的最小学习记录。"""

    record_id: str
    question_id: str
    question_type: str
    is_correct: bool
    error_tags: list[str]
    knowledge_tags: list[str]
    duration_seconds: int | None
    created_at: str


class LearningRecommendationResponse(BaseModel):
    """基于用户自身历史生成的练习方向。"""

    focus_type: str
    label: str
    reason: str
    suggested_practice: str


class LearningProfileResponse(BaseModel):
    """当前认证用户的学习画像与练习方向。"""

    total_attempts: int
    correct_attempts: int
    accuracy: float | None
    error_counts: list[tuple[str, int]]
    knowledge_mastery: list[tuple[str, float]]
    recommendations: list[LearningRecommendationResponse]


class ReviewDiagnosticResponse(BaseModel):
    """首错或未完成原因的结构化输出。"""

    code: str
    message: str
    knowledge_tags: list[str]
    step_index: int | None


class SolutionReviewResponse(BaseModel):
    """用户结构化解法的逐步批改结果。"""

    status: str
    checked_step_count: int
    established_literals: list[str]
    diagnostic: ReviewDiagnosticResponse | None
    baseline_status: str
    verification_level: str


@app.middleware("http")
async def collect_runtime_metrics(request: Request, call_next):
    """聚合非敏感的路由、状态和时延指标，不记录请求内容。"""
    started_at = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        runtime_metrics.record(
            request.url.path,
            status_code=500,
            latency_ms=(perf_counter() - started_at) * 1000,
        )
        raise
    route = request.scope.get("route")
    route_template = getattr(route, "path", request.url.path)
    runtime_metrics.record(
        route_template,
        status_code=response.status_code,
        latency_ms=(perf_counter() - started_at) * 1000,
    )
    return response


@app.get("/")
def learner_home() -> FileResponse:
    """返回匿名学习者可使用的同源逻辑验证页面。"""
    return FileResponse(static_directory / "index.html")


@app.get("/health")
def health_check() -> dict[str, str]:
    """返回服务健康状态。"""
    return {"status": "ok"}


@app.get(
    "/v1/admin/review-dashboard",
    response_model=ReviewDashboardResponse,
)
def get_review_dashboard(_: AdminIdentity) -> ReviewDashboardResponse:
    """读取不含题干正文的题库审核状态统计。"""
    return _review_dashboard_to_response(review_store.dashboard())


@app.get(
    "/v1/admin/runtime-metrics",
    response_model=RuntimeMetricsResponse,
)
def get_runtime_metrics(_: AdminIdentity) -> RuntimeMetricsResponse:
    """读取当前进程中的非敏感运行指标。"""
    return _runtime_metrics_to_response(runtime_metrics.snapshot())


@app.post("/v1/questions/solve", response_model=SolveResponse)
def solve_question(request: SolveRequest) -> SolveResponse:
    """验证结构化命题逻辑查询，并返回可追溯证明步骤。"""
    try:
        facts = [Literal.parse(fact) for fact in request.facts]
        rules = [
            ImplicationRule(
                premise=Literal.parse(rule.premise),
                conclusion=Literal.parse(rule.conclusion),
                source_text=rule.source_text,
            )
            for rule in request.rules
        ]
        result = engine.verify(
            facts=facts,
            rules=rules,
            query=Literal.parse(request.query),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return _to_response(result)


@app.post("/v1/questions/verify-choice", response_model=ChoiceVerifyResponse)
def verify_choice(request: ChoiceVerifyRequest) -> ChoiceVerifyResponse:
    """对一个结构化选项进行全模型验证，返回正例或反例。"""
    try:
        facts, rules = _parse_structured_conditions(request.facts, request.rules)
        result = choice_verifier.verify(
            facts=facts,
            rules=rules,
            option=Literal.parse(request.option),
            question_type=request.question_type,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return _choice_to_response(result)


@app.post("/v1/questions/solve-ordering", response_model=OrderingSolveResponse)
def solve_ordering_question(request: OrderingSolveRequest) -> OrderingSolveResponse:
    """完整枚举小规模排序题，返回精确解数和可读样例。"""
    try:
        constraints = tuple(
            OrderingConstraint(
                constraint_type=item.constraint_type,
                item=item.item,
                other_item=item.other_item,
                position=item.position,
            )
            for item in request.constraints
        )
        result = ordering_solver.solve(
            items=tuple(request.items),
            constraints=constraints,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return _ordering_to_response(result)


@app.post("/v1/questions/solve-grouping", response_model=GroupingSolveResponse)
def solve_grouping_question(request: GroupingSolveRequest) -> GroupingSolveResponse:
    """完整枚举小规模分组题，返回精确解数和分配样例。"""
    try:
        constraints = tuple(
            GroupConstraint(
                constraint_type=constraint.constraint_type,
                item=constraint.item,
                other_item=constraint.other_item,
            )
            for constraint in request.constraints
        )
        result = grouping_solver.solve(
            items=tuple(request.items),
            groups=tuple(request.groups),
            max_group_size=request.max_group_size,
            constraints=constraints,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return _grouping_to_response(result)


@app.post("/v1/questions/solve-matching", response_model=MatchingSolveResponse)
def solve_matching_question(request: MatchingSolveRequest) -> MatchingSolveResponse:
    """完整枚举小规模一对一匹配题，返回精确解数和匹配样例。"""
    try:
        constraints = tuple(
            MatchConstraint(
                constraint_type=constraint.constraint_type,
                item=constraint.item,
                target=constraint.target,
            )
            for constraint in request.constraints
        )
        result = matching_solver.solve(
            items=tuple(request.items),
            targets=tuple(request.targets),
            constraints=constraints,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return _matching_to_response(result)


@app.post("/v1/questions/ocr", response_model=OcrExtractResponse)
def extract_question_image(request: OcrExtractRequest) -> OcrExtractResponse:
    """提取图片文本并提示用户核对关键逻辑词。"""
    try:
        extraction = ocr_service.extract_from_base64(request.image_base64)
    except OcrUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _ocr_extraction_to_response(extraction)


@app.post("/v1/questions/ocr/correct", response_model=OcrCorrectionResponse)
def correct_question_ocr(request: OcrCorrectionRequest) -> OcrCorrectionResponse:
    """核对用户修订的 OCR 文本，突出影响逻辑推理的关键词修改。"""
    try:
        correction = ocr_correction_service.review(
            original_text=request.original_text,
            corrected_text=request.corrected_text,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _ocr_correction_to_response(correction)


@app.post("/v1/questions/review-solution", response_model=SolutionReviewResponse)
def review_solution(request: SolutionReviewRequest) -> SolutionReviewResponse:
    """批改用户的结构化推理，并返回首错与关联知识点。"""
    try:
        facts, rules = _parse_structured_conditions(request.facts, request.rules)
        steps = tuple(
            ReasoningStep(
                rule_id=step.rule_id,
                direction=step.direction,
                premise=Literal.parse(step.premise),
                conclusion=Literal.parse(step.conclusion),
            )
            for step in request.steps
        )
        result = solution_reviewer.review(
            facts=facts,
            rules=rules,
            target=Literal.parse(request.target),
            steps=steps,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _solution_review_to_response(result)


@app.post("/v1/learning/records", response_model=LearningRecordResponse)
def create_learning_record(
    request: LearningRecordCreateRequest,
    identity: CurrentIdentity,
) -> LearningRecordResponse:
    """以当前认证主体保存最小学习记录，不接受客户端指定用户。"""
    try:
        record = learning_store.add_record(
            LearningRecordInput(
                user_id=identity.subject,
                question_id=request.question_id,
                question_type=request.question_type,
                is_correct=request.is_correct,
                error_tags=tuple(request.error_tags),
                knowledge_tags=tuple(request.knowledge_tags),
                duration_seconds=request.duration_seconds,
            )
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _learning_record_to_response(record)


@app.get("/v1/learning/profile", response_model=LearningProfileResponse)
def get_learning_profile(identity: CurrentIdentity) -> LearningProfileResponse:
    """读取当前认证用户自己的学习画像和练习方向。"""
    try:
        profile = learning_store.get_profile(identity.subject)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _learning_profile_to_response(profile)


@app.get(
    "/v1/learning/recommendations",
    response_model=list[PracticeRecommendationResponse],
)
def get_practice_recommendations(
    identity: CurrentIdentity,
    limit: int = Query(default=5, ge=1, le=20),
) -> list[PracticeRecommendationResponse]:
    """基于当前认证用户的记录推荐未尝试且已审核发布的题目。"""
    try:
        profile = learning_store.get_profile(identity.subject)
        attempted = learning_store.attempted_question_ids(identity.subject)
        recommendations = question_bank_store.recommend(
            profile=profile,
            attempted_question_ids=attempted,
            limit=limit,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return [_practice_recommendation_to_response(item) for item in recommendations]


@app.delete("/v1/learning/records/{record_id}")
def delete_learning_record(
    record_id: str,
    identity: CurrentIdentity,
) -> dict[str, bool]:
    """仅允许当前认证用户删除属于自己的学习记录。"""
    try:
        deleted = learning_store.delete_record(
            user_id=identity.subject,
            record_id=record_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if not deleted:
        raise HTTPException(status_code=404, detail="未找到属于当前用户的学习记录")
    return {"deleted": True}


@app.post("/v1/questions/solve-chinese", response_model=ChineseSolveResponse)
def solve_chinese_question(request: ChineseSolveRequest) -> ChineseSolveResponse:
    """解析明确中文条件；复杂语义须经结构化人工确认后才进入求解。"""
    try:
        analysis = chinese_parser.analyze(request.conditions)
        query = chinese_parser.parse_query(request.query)
        confirmation_requests = _confirmation_requests_to_response(
            analysis.confirmation_requests
        )
        if analysis.confirmation_requests and not request.confirmations:
            return _chinese_confirmation_required_response(
                query=query,
                parsed=analysis.parsed,
                confirmation_requests=confirmation_requests,
            )
        parsed = chinese_parser.parse_with_confirmations(
            request.conditions,
            _confirmed_sentences_from_request(request.confirmations),
        )
        result = engine.verify(
            facts=parsed.facts,
            rules=parsed.rules,
            query=query,
        )
    except (ChineseParseError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return _chinese_solve_response(
        result=result,
        parsed=parsed,
        confirmation_required=False,
        confirmation_requests=confirmation_requests,
    )


def _confirmed_sentences_from_request(
    confirmations: list[ChineseConfirmationInput],
) -> tuple[ConfirmedChineseSentence, ...]:
    """将 API 确认输入转换为领域层的结构化事实和规则。"""
    return tuple(
        ConfirmedChineseSentence(
            source_sentence=confirmation.source_sentence,
            facts=tuple(Literal.parse(fact) for fact in confirmation.facts),
            rules=tuple(
                ImplicationRule(
                    premise=Literal.parse(rule.premise),
                    conclusion=Literal.parse(rule.conclusion),
                )
                for rule in confirmation.rules
            ),
        )
        for confirmation in confirmations
    )


def _confirmation_requests_to_response(
    requests: tuple[ChineseConfirmationRequest, ...],
) -> list[ChineseConfirmationRequestOutput]:
    """将复杂语义确认项转换为稳定的 API 输出。"""
    return [
        ChineseConfirmationRequestOutput(
            source_sentence=request.source_sentence,
            codes=list(request.codes),
            message=request.message,
        )
        for request in requests
    ]


def _chinese_confirmation_required_response(
    query: Literal,
    parsed: ParsedChineseText,
    confirmation_requests: list[ChineseConfirmationRequestOutput],
) -> ChineseSolveResponse:
    """返回已自动解析部分与待确认项，明确不输出未经验证结论。"""
    return ChineseSolveResponse(
        query=query.display(),
        status="confirmation_required",
        conclusion="题干包含复杂中文语义，请完成结构化人工确认后再验证结论。",
        verification_level="not_verified_pending_human_confirmation",
        proof_steps=[],
        known_literals=[fact.display() for fact in parsed.facts],
        conflict=None,
        parsed_facts=[fact.display() for fact in parsed.facts],
        parsed_rules=_parsed_rules_to_response(parsed),
        source_sentences=list(parsed.source_sentences),
        confirmation_required=True,
        confirmation_requests=confirmation_requests,
    )


def _chinese_solve_response(
    result: VerificationResult,
    parsed: ParsedChineseText,
    confirmation_required: bool,
    confirmation_requests: list[ChineseConfirmationRequestOutput],
) -> ChineseSolveResponse:
    """将已确认的中文条件求解结果与解析回显合并为 API 响应。"""
    base_response = _to_response(result)
    return ChineseSolveResponse(
        **base_response.model_dump(),
        parsed_facts=[fact.display() for fact in parsed.facts],
        parsed_rules=_parsed_rules_to_response(parsed),
        source_sentences=list(parsed.source_sentences),
        confirmation_required=confirmation_required,
        confirmation_requests=confirmation_requests,
    )


def _parsed_rules_to_response(
    parsed: ParsedChineseText,
) -> list[ParsedRuleOutput]:
    """回显所有自动或人工确认后进入逻辑内核的规则。"""
    return [
        ParsedRuleOutput(
            premise=rule.premise.display(),
            conclusion=rule.conclusion.display(),
            source_text=rule.source_text or rule.display(),
        )
        for rule in parsed.rules
    ]


def _learner_question_to_response(
    question: LearnerQuestion,
) -> LearnerQuestionResponse:
    """将最小学习者题目视图转换为公开推荐响应。"""
    return LearnerQuestionResponse(
        question_id=question.question_id,
        content_version=question.content_version,
        question_type=question.question_type,
        stem=question.stem,
        options=list(question.options),
    )


def _practice_recommendation_to_response(
    recommendation: PracticeRecommendation,
) -> PracticeRecommendationResponse:
    """将用户范围内推荐转换为不含内部标签的学习者响应。"""
    return PracticeRecommendationResponse(
        question=_learner_question_to_response(recommendation.question),
        reason=recommendation.reason,
    )


def _review_dashboard_to_response(
    dashboard: ReviewDashboard,
) -> ReviewDashboardResponse:
    """将审核状态计数转换为 API 输出。"""
    return ReviewDashboardResponse(
        total_questions=dashboard.total_questions,
        status_counts=list(dashboard.status_counts),
        total_review_events=dashboard.total_review_events,
    )


def _runtime_metrics_to_response(
    snapshot: RuntimeMetricsSnapshot,
) -> RuntimeMetricsResponse:
    """将进程内指标快照转换为 API 输出。"""
    return RuntimeMetricsResponse(
        total_requests=snapshot.total_requests,
        error_requests=snapshot.error_requests,
        average_latency_ms=snapshot.average_latency_ms,
        route_counts=list(snapshot.route_counts),
        status_counts=list(snapshot.status_counts),
    )


def _learning_record_to_response(record: LearningRecord) -> LearningRecordResponse:
    """将当前用户的持久化记录转换为 API 输出。"""
    return LearningRecordResponse(
        record_id=record.record_id,
        question_id=record.question_id,
        question_type=record.question_type,
        is_correct=record.is_correct,
        error_tags=list(record.error_tags),
        knowledge_tags=list(record.knowledge_tags),
        duration_seconds=record.duration_seconds,
        created_at=record.created_at,
    )


def _learning_profile_to_response(
    profile: LearningProfile,
) -> LearningProfileResponse:
    """将当前用户的学习画像转换为 API 输出。"""
    return LearningProfileResponse(
        total_attempts=profile.total_attempts,
        correct_attempts=profile.correct_attempts,
        accuracy=profile.accuracy,
        error_counts=list(profile.error_counts),
        knowledge_mastery=list(profile.knowledge_mastery),
        recommendations=[
            LearningRecommendationResponse(
                focus_type=item.focus_type,
                label=item.label,
                reason=item.reason,
                suggested_practice=item.suggested_practice,
            )
            for item in profile.recommendations
        ],
    )


def _solution_review_to_response(
    result: SolutionReviewResult,
) -> SolutionReviewResponse:
    """将批改领域结果转换为稳定 API 输出。"""
    diagnostic = (
        _review_diagnostic_to_response(result.diagnostic)
        if result.diagnostic
        else None
    )
    verification_level = (
        "inconsistent_conditions"
        if result.baseline.status.value == "inconsistent"
        else "fully_verified_by_logic_engine"
    )
    return SolutionReviewResponse(
        status=result.status.value,
        checked_step_count=result.checked_step_count,
        established_literals=[item.display() for item in result.established_literals],
        diagnostic=diagnostic,
        baseline_status=result.baseline.status.value,
        verification_level=verification_level,
    )


def _review_diagnostic_to_response(
    diagnostic: ReviewDiagnostic,
) -> ReviewDiagnosticResponse:
    """转换诊断标签，避免 API 暴露领域对象。"""
    return ReviewDiagnosticResponse(
        code=diagnostic.code.value,
        message=diagnostic.message,
        knowledge_tags=list(diagnostic.knowledge_tags),
        step_index=diagnostic.step_index,
    )


def _ocr_extraction_to_response(extraction: OcrExtraction) -> OcrExtractResponse:
    """将 OCR 领域结果转换为稳定 API 输出。"""
    return OcrExtractResponse(
        provider=extraction.provider,
        image_type=extraction.image_type,
        text=extraction.text,
        warnings=list(extraction.warnings),
        critical_terms=list(extraction.critical_terms),
    )


def _ocr_correction_to_response(
    correction: OcrCorrectionResult,
) -> OcrCorrectionResponse:
    """将 OCR 校正领域结果转换为稳定 API 输出。"""
    return OcrCorrectionResponse(
        corrected_text=correction.corrected_text,
        requires_confirmation=correction.requires_confirmation,
        changed_critical_terms=list(correction.changed_critical_terms),
        warnings=list(correction.warnings),
    )


def _grouping_to_response(result: GroupingSolveResult) -> GroupingSolveResponse:
    """将分组结果转成稳定 API 输出，显式区分无解与搜索边界。"""
    if result.status is GroupingSolveStatus.COMPLETE:
        conclusion = f"已完整验证，题干共有 {result.solution_count} 种合法分组。"
        verification_level = "fully_verified_by_enumeration"
    elif result.status is GroupingSolveStatus.UNSATISFIABLE:
        conclusion = "题干约束彼此冲突，不存在合法分组。"
        verification_level = "inconsistent_conditions"
    else:
        conclusion = "候选分组方案数量超过安全搜索上限，当前无法给出可靠结论。"
        verification_level = "not_verified_search_limit"

    sample_solutions = [dict(sample.assignments) for sample in result.sample_solutions]
    return GroupingSolveResponse(
        status=result.status,
        conclusion=conclusion,
        verification_level=verification_level,
        solution_count=result.solution_count,
        sample_solutions=sample_solutions,
    )


def _matching_to_response(result: MatchingSolveResult) -> MatchingSolveResponse:
    """将一对一匹配结果转成稳定 API 输出。"""
    if result.status is MatchingSolveStatus.COMPLETE:
        conclusion = f"已完整验证，题干共有 {result.solution_count} 种合法匹配。"
        verification_level = "fully_verified_by_enumeration"
    elif result.status is MatchingSolveStatus.UNSATISFIABLE:
        conclusion = "题干约束彼此冲突，不存在合法匹配。"
        verification_level = "inconsistent_conditions"
    else:
        conclusion = "匹配对象数量超过安全枚举上限，当前无法给出可靠结论。"
        verification_level = "not_verified_item_limit"

    sample_solutions = [dict(sample.assignments) for sample in result.sample_solutions]
    return MatchingSolveResponse(
        status=result.status,
        conclusion=conclusion,
        verification_level=verification_level,
        solution_count=result.solution_count,
        sample_solutions=sample_solutions,
    )


def _ordering_to_response(result: OrderingSolveResult) -> OrderingSolveResponse:
    """将排序求解结果转成稳定 API 输出，明确展示验证完整性。"""
    if result.status is OrderingSolveStatus.COMPLETE:
        conclusion = f"已完整验证，题干共有 {result.solution_count} 种合法排序。"
        verification_level = "fully_verified_by_enumeration"
    elif result.status is OrderingSolveStatus.UNSATISFIABLE:
        conclusion = "题干约束彼此冲突，不存在合法排序。"
        verification_level = "inconsistent_conditions"
    else:
        conclusion = "排序对象数量超过安全枚举上限，当前无法给出可靠结论。"
        verification_level = "not_verified_item_limit"

    return OrderingSolveResponse(
        status=result.status,
        conclusion=conclusion,
        verification_level=verification_level,
        solution_count=result.solution_count,
        sample_solutions=[list(solution) for solution in result.sample_solutions],
    )


def _parse_structured_conditions(
    fact_inputs: list[str],
    rule_inputs: list[RuleInput],
) -> tuple[tuple[Literal, ...], tuple[ImplicationRule, ...]]:
    """将结构化 API 输入转换为领域对象，供多个接口复用。"""
    facts = tuple(Literal.parse(fact) for fact in fact_inputs)
    rules = tuple(
        ImplicationRule(
            premise=Literal.parse(rule.premise),
            conclusion=Literal.parse(rule.conclusion),
            source_text=rule.source_text,
        )
        for rule in rule_inputs
    )
    return facts, rules


def _choice_to_response(result: ChoiceVerificationResult) -> ChoiceVerifyResponse:
    """将选择题验证结果转换为包含正例或反例的 API 响应。"""
    witness_type = _witness_type_for(result)
    witness_model = (
        [literal.display() for literal in result.witness_model.display_literals()]
        if result.witness_model
        else None
    )
    return ChoiceVerifyResponse(
        question_type=result.question_type,
        option=result.option.display(),
        status=result.status,
        conclusion=_choice_conclusion_for(result),
        verification_level=_choice_verification_level_for(result),
        enumeration_status=result.enumeration_status.value,
        model_count=result.model_count,
        witness_type=witness_type,
        witness_model=witness_model,
    )


def _witness_type_for(result: ChoiceVerificationResult) -> str | None:
    """指明返回的模型是用于支持还是反驳该选项的实例。"""
    if result.witness_model is None:
        return None
    if result.question_type in {
        ChoiceQuestionType.MUST_BE_TRUE,
        ChoiceQuestionType.CANNOT_BE_INFERRED,
    }:
        return "counterexample"
    return "example"


def _choice_verification_level_for(result: ChoiceVerificationResult) -> str:
    """区分完整枚举验证与因安全边界被阻断的情况。"""
    if result.status is ChoiceVerificationStatus.BLOCKED_UNSATISFIABLE:
        return "inconsistent_conditions"
    if result.status is ChoiceVerificationStatus.BLOCKED_SYMBOL_LIMIT:
        return "not_verified_symbol_limit"
    return "fully_verified_by_enumeration"


def _choice_conclusion_for(result: ChoiceVerificationResult) -> str:
    """按四类选择题设问生成与模型依据一致的简洁结论。"""
    option = result.option.display()
    model_count = result.model_count
    if result.status is ChoiceVerificationStatus.BLOCKED_UNSATISFIABLE:
        return "题干条件不存在合法模型，无法验证该选项。"
    if result.status is ChoiceVerificationStatus.BLOCKED_SYMBOL_LIMIT:
        return "命题数量超过安全枚举上限，当前无法验证该选项。"

    templates = {
        ChoiceQuestionType.MUST_BE_TRUE: (
            f"全部 {model_count} 个合法模型均满足 {option}，因此该选项一定为真。",
            f"存在违反 {option} 的合法模型，因此该选项不一定为真。",
        ),
        ChoiceQuestionType.MAY_BE_TRUE: (
            f"存在满足 {option} 的合法模型，因此该选项可能为真。",
            f"全部 {model_count} 个合法模型均不满足 {option}，因此该选项不可能为真。",
        ),
        ChoiceQuestionType.CANNOT_BE_TRUE: (
            f"全部 {model_count} 个合法模型均不满足 {option}，因此该选项不可能为真。",
            f"存在满足 {option} 的合法模型，因此该选项并非不可能为真。",
        ),
        ChoiceQuestionType.CANNOT_BE_INFERRED: (
            f"存在违反 {option} 的合法模型，因此题干无法推出该选项。",
            f"全部 {model_count} 个合法模型均满足 {option}，因此题干可以推出该选项。",
        ),
    }
    verified, not_verified = templates[result.question_type]
    return verified if result.is_verified else not_verified


def _to_response(result: VerificationResult) -> SolveResponse:
    """将领域结果转换为 API 输出，避免暴露内部实现对象。"""
    return SolveResponse(
        query=result.query.display(),
        status=result.status.value,
        conclusion=_conclusion_for(result),
        verification_level=_verification_level_for(result),
        proof_steps=[
            ProofStepOutput(
                derived=step.derived.display(),
                reason=step.reason,
                rule=step.source_rule.display() if step.source_rule else None,
                source_text=step.source_rule.source_text if step.source_rule else None,
                dependencies=[dependency.display() for dependency in step.dependencies],
            )
            for step in result.proof_steps
        ],
        known_literals=[literal.display() for literal in result.known_literals],
        conflict=(
            [literal.display() for literal in result.conflict]
            if result.conflict
            else None
        ),
    )


def _verification_level_for(result: VerificationResult) -> str:
    """根据验证状态给出诚实的可靠性等级，避免硬编码夸大可信度。"""
    if result.status.value == "inconsistent":
        return "inconsistent_conditions"
    return "fully_verified"


def _conclusion_for(result: VerificationResult) -> str:
    """将验证状态转成前端可直接展示的简洁结论。"""
    match result.status.value:
        case "proved":
            return f"可以由题干推出 {result.query.display()}。"
        case "disproved":
            opposite = result.query.opposite().display()
            query = result.query.display()
            return f"题干可推出 {opposite}，因此不能推出 {query}。"
        case "inconsistent":
            if result.conflict:
                left, right = (literal.display() for literal in result.conflict)
                conflict_text = f"题干条件存在矛盾（{left} 与 {right} 冲突）"
                return f"{conflict_text}，当前不能给出可靠结论。"
            return "题干条件存在矛盾，当前不能给出可靠结论。"
        case _:
            return f"现有条件不足以判断 {result.query.display()}。"
