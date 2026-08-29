const recommendationList = document.querySelector("#recommendation-list");
const recommendationStatus = document.querySelector("#recommendation-status");
const refreshButton = document.querySelector("#refresh-button");
const questionPanel = document.querySelector("#question-panel");
const questionTitle = document.querySelector("#question-title");
const questionType = document.querySelector("#question-type");
const practiceBadge = document.querySelector("#practice-badge");
const practiceReason = document.querySelector("#practice-reason");
const questionBody = document.querySelector("#question-body");
const questionStem = document.querySelector("#question-stem");
const optionGroup = document.querySelector("#option-group");
const attemptForm = document.querySelector("#attempt-form");
const submitAttemptButton = document.querySelector("#submit-attempt");
const attemptError = document.querySelector("#attempt-error");
const attemptResult = document.querySelector("#attempt-result");

let selectedRecommendation = null;
let attemptStartedAt = null;

function setLoading(button, isLoading) {
  button.disabled = isLoading;
  button.classList.toggle("is-loading", isLoading);
}

function setError(element, message) {
  element.textContent = message;
  element.hidden = !message;
}

function createElement(tagName, text = "") {
  const element = document.createElement(tagName);
  element.textContent = text;
  return element;
}

function detailFromPayload(payload, fallback) {
  if (typeof payload?.detail === "string") {
    return payload.detail;
  }
  return fallback;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const error = new Error(detailFromPayload(payload, "请求未成功，请稍后重试。"));
    error.status = response.status;
    throw error;
  }
  return payload;
}

function setQuestionState({ title, type, badge, status, reason = "" }) {
  questionTitle.textContent = title;
  questionType.textContent = type;
  practiceBadge.textContent = badge;
  practiceBadge.className = `status-badge ${status}`;
  practiceReason.textContent = reason;
}

function resetQuestion() {
  selectedRecommendation = null;
  attemptStartedAt = null;
  questionBody.hidden = true;
  attemptResult.hidden = true;
  attemptResult.replaceChildren();
  optionGroup.replaceChildren();
  setError(attemptError, "");
  setQuestionState({
    title: "选择一题开始练习",
    type: "等待取题",
    badge: "未开始",
    status: "neutral",
  });
}

function recommendationButton(recommendation) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "recommendation-item";
  button.dataset.questionId = recommendation.question.question_id;
  button.dataset.contentVersion = recommendation.question.content_version;

  const title = createElement("strong", recommendation.question.stem);
  const metadata = createElement(
    "span",
    `${questionTypeLabel(recommendation.question.question_type)} · ${recommendation.reason}`,
  );
  button.append(title, metadata);
  button.addEventListener("click", () => loadQuestion(recommendation));
  return button;
}

function renderRecommendations(recommendations) {
  recommendationList.replaceChildren();
  if (!recommendations.length) {
    const empty = document.createElement("section");
    empty.className = "empty-state";
    empty.append(
      createElement("h3", "暂时没有可练习的新题"),
      createElement("p", "完成后续审核发布，或清除自己的练习记录后再查看。"),
    );
    recommendationList.append(empty);
    return;
  }

  for (const recommendation of recommendations) {
    recommendationList.append(recommendationButton(recommendation));
  }
}

function questionTypeLabel(questionType) {
  const labels = {
    propositional: "命题推理",
    ordering: "排序约束",
    grouping: "分组约束",
    matching: "一对一匹配",
  };
  return labels[questionType] ?? "逻辑题";
}

function renderOptions(options) {
  optionGroup.replaceChildren();
  const legend = createElement("legend", "选择你的答案");
  optionGroup.append(legend);

  for (const [index, option] of options.entries()) {
    const label = document.createElement("label");
    label.className = "option-choice";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "selected-option";
    input.value = option;
    input.required = true;
    const text = createElement("span", option);
    const marker = createElement("span", String.fromCharCode(65 + index));
    marker.className = "option-marker";
    label.append(input, marker, text);
    optionGroup.append(label);
  }
}

async function loadRecommendations({ resetQuestionState = true } = {}) {
  setLoading(refreshButton, true);
  recommendationStatus.textContent = "正在读取已审核发布的练习题…";
  if (resetQuestionState) {
    resetQuestion();
  }
  try {
    const recommendations = await requestJson("/v1/learning/recommendations?limit=5");
    renderRecommendations(recommendations);
    recommendationStatus.textContent = recommendations.length
      ? `已找到 ${recommendations.length} 道未尝试的已发布题目。`
      : "当前没有可练习的新题。";
  } catch (error) {
    recommendationList.replaceChildren();
    const failure = document.createElement("section");
    failure.className = "empty-state error-state";
    failure.append(
      createElement("h3", "无法读取练习题"),
      createElement("p", error.message),
    );
    recommendationList.append(failure);
    recommendationStatus.textContent = "练习列表不可用。请确认当前部署已注入受信代理身份。";
  } finally {
    setLoading(refreshButton, false);
  }
}

async function loadQuestion(recommendation) {
  setError(attemptError, "");
  attemptResult.hidden = true;
  attemptResult.replaceChildren();
  setQuestionState({
    title: "正在读取题目",
    type: questionTypeLabel(recommendation.question.question_type),
    badge: "加载中",
    status: "pending",
    reason: recommendation.reason,
  });
  questionBody.hidden = true;

  try {
    const question = await requestJson(
      `/v1/learning/questions/${encodeURIComponent(recommendation.question.question_id)}/${encodeURIComponent(recommendation.question.content_version)}`,
    );
    selectedRecommendation = { question, reason: recommendation.reason };
    questionTitle.textContent = "开始作答";
    questionType.textContent = questionTypeLabel(question.question_type);
    practiceBadge.textContent = "作答中";
    practiceBadge.className = "status-badge pending";
    practiceReason.textContent = recommendation.reason;
    questionStem.textContent = question.stem;
    renderOptions(question.options);
    questionBody.hidden = false;
    attemptStartedAt = performance.now();
    questionPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    document.querySelector('input[name="selected-option"]')?.focus();
  } catch (error) {
    resetQuestion();
    setQuestionState({
      title: "题目暂不可用",
      type: "已发布题目",
      badge: "无法加载",
      status: "error",
      reason: error.message,
    });
  }
}

function selectedOption() {
  return document.querySelector('input[name="selected-option"]:checked')?.value ?? null;
}

function renderAttemptResult(payload) {
  attemptResult.replaceChildren();
  const heading = createElement(
    "h3",
    payload.is_correct ? "回答正确" : "本次回答不正确",
  );
  const summary = createElement(
    "p",
    payload.is_correct
      ? "结果已由服务端的发布验证审计判定，并记录在你的学习档案中。"
      : "结果已由服务端的发布验证审计判定。建议回到自主验证页，逐步检查条件方向与推理依据。",
  );
  attemptResult.className = `attempt-result ${payload.is_correct ? "correct" : "incorrect"}`;
  attemptResult.append(heading, summary);
  attemptResult.hidden = false;
  setQuestionState({
    title: payload.is_correct ? "本题完成" : "本题已记录",
    type: questionTypeLabel(selectedRecommendation.question.question_type),
    badge: payload.is_correct ? "已完成" : "待复盘",
    status: payload.is_correct ? "proved" : "disproved",
    reason: "本题不会再次出现在你的推荐列表中。",
  });
  questionBody.hidden = true;
}

async function submitAttempt(event) {
  event.preventDefault();
  setError(attemptError, "");
  const option = selectedOption();
  if (!option || !selectedRecommendation) {
    setError(attemptError, "请选择一个选项后再提交。" );
    return;
  }

  const durationSeconds = Math.max(
    0,
    Math.round((performance.now() - attemptStartedAt) / 1000),
  );
  setLoading(submitAttemptButton, true);
  try {
    const payload = await requestJson(
      `/v1/learning/questions/${encodeURIComponent(selectedRecommendation.question.question_id)}/${encodeURIComponent(selectedRecommendation.question.content_version)}/attempts`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selected_option: option,
          duration_seconds: durationSeconds,
        }),
      },
    );
    renderAttemptResult(payload);
    await loadRecommendations({ resetQuestionState: false });
  } catch (error) {
    setError(attemptError, error.message);
  } finally {
    setLoading(submitAttemptButton, false);
  }
}

refreshButton.addEventListener("click", loadRecommendations);
attemptForm.addEventListener("submit", submitAttempt);
loadRecommendations();
