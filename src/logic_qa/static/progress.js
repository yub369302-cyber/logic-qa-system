const refreshProgressButton = document.querySelector("#refresh-progress");
const progressStatus = document.querySelector("#progress-status");
const totalAttempts = document.querySelector("#total-attempts");
const correctAttempts = document.querySelector("#correct-attempts");
const accuracy = document.querySelector("#accuracy");
const focusList = document.querySelector("#focus-list");

function createElement(tagName, text = "") {
  const element = document.createElement(tagName);
  element.textContent = text;
  return element;
}

function setLoading(isLoading) {
  refreshProgressButton.disabled = isLoading;
  refreshProgressButton.classList.toggle("is-loading", isLoading);
}

function detailFromPayload(payload, fallback) {
  return typeof payload?.detail === "string" ? payload.detail : fallback;
}

async function requestProgress() {
  const response = await fetch("/v1/learning/profile");
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(detailFromPayload(payload, "学习概览暂不可用，请稍后重试。"));
  }
  return payload;
}

function renderMetrics(profile) {
  totalAttempts.textContent = String(profile.total_attempts);
  correctAttempts.textContent = String(profile.correct_attempts);
  accuracy.textContent = profile.accuracy === null ? "—" : `${Math.round(profile.accuracy * 100)}%`;
}

function renderFocusAreas(focusAreas, totalAttemptCount) {
  focusList.replaceChildren();
  if (!totalAttemptCount) {
    const empty = document.createElement("section");
    empty.className = "empty-state";
    empty.append(
      createElement("h3", "从第一道练习开始"),
      createElement("p", "完成一题已审核发布题后，这里会给出基于你自己记录的复盘方向。"),
    );
    focusList.append(empty);
    return;
  }

  if (!focusAreas.length) {
    const steady = document.createElement("section");
    steady.className = "empty-state";
    steady.append(
      createElement("h3", "保持当前节奏"),
      createElement("p", "目前没有需要优先处理的复盘方向，可继续完成新的已发布练习题。"),
    );
    focusList.append(steady);
    return;
  }

  for (const focusArea of focusAreas) {
    const item = document.createElement("article");
    item.className = "focus-item";
    const kind = createElement(
      "p",
      focusArea.kind === "reasoning_pattern" ? "推理模式" : "知识巩固",
    );
    kind.className = "section-kicker";
    const title = createElement("h3", focusArea.title);
    const reason = createElement("p", focusArea.reason);
    const suggestion = createElement("p", focusArea.suggested_practice);
    suggestion.className = "focus-suggestion";
    item.append(kind, title, reason, suggestion);
    focusList.append(item);
  }
}

async function loadProgress() {
  setLoading(true);
  progressStatus.textContent = "正在读取你的学习概览…";
  try {
    const profile = await requestProgress();
    renderMetrics(profile);
    renderFocusAreas(profile.focus_areas, profile.total_attempts);
    progressStatus.textContent = profile.total_attempts
      ? "概览已根据你自己的练习记录更新。"
      : "暂未记录练习；完成一道已发布题后会自动形成概览。";
  } catch (error) {
    totalAttempts.textContent = "—";
    correctAttempts.textContent = "—";
    accuracy.textContent = "—";
    focusList.replaceChildren();
    const failure = document.createElement("section");
    failure.className = "empty-state error-state";
    failure.append(
      createElement("h3", "无法读取学习概览"),
      createElement("p", error.message),
    );
    focusList.append(failure);
    progressStatus.textContent = "概览不可用。请确认当前部署已注入受信代理身份。";
  } finally {
    setLoading(false);
  }
}

refreshProgressButton.addEventListener("click", loadProgress);
loadProgress();
