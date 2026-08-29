const solveForm = document.querySelector("#solve-form");
const confirmationForm = document.querySelector("#confirmation-form");
const conditionsInput = document.querySelector("#conditions");
const queryInput = document.querySelector("#query");
const solveButton = document.querySelector("#solve-button");
const confirmationButton = document.querySelector("#confirmation-submit");
const formError = document.querySelector("#form-error");
const confirmationError = document.querySelector("#confirmation-error");
const confirmationPanel = document.querySelector("#confirmation-panel");
const confirmationFields = document.querySelector("#confirmation-fields");
const resultTitle = document.querySelector("#result-title");
const resultBadge = document.querySelector("#result-badge");
const resultSummary = document.querySelector("#result-summary");
const resultDetails = document.querySelector("#result-details");
const exampleButton = document.querySelector("#example-button");

let pendingConfirmationRequests = [];

function setLoading(button, isLoading) {
  button.disabled = isLoading;
  button.classList.toggle("is-loading", isLoading);
}

function setError(element, message) {
  element.textContent = message;
  element.hidden = !message;
}

function safeText(value) {
  return String(value ?? "");
}

function createElement(tagName, text = "") {
  const element = document.createElement(tagName);
  element.textContent = text;
  return element;
}

function setResultState({ title, badge, status, summary, details = null }) {
  resultTitle.textContent = title;
  resultBadge.textContent = badge;
  resultBadge.className = `status-badge ${status}`;
  resultSummary.textContent = summary;
  resultDetails.replaceChildren();
  resultDetails.hidden = details === null;

  if (details !== null) {
    resultDetails.append(details);
  }
}

function renderListBlock(title, values, className = "") {
  const block = document.createElement("section");
  block.className = "result-block";
  const heading = createElement("h3", title);
  block.append(heading);

  if (!values.length) {
    block.append(createElement("p", "无。"));
    return block;
  }

  const list = document.createElement("ul");
  for (const value of values) {
    const item = document.createElement("li");
    const text = createElement("span", value);
    if (className) {
      text.className = className;
    }
    item.append(text);
    list.append(item);
  }
  block.append(list);
  return block;
}

function renderRulesBlock(rules) {
  const values = rules.map(
    (rule) => `${rule.premise} → ${rule.conclusion}（${rule.source_text}）`,
  );
  return renderListBlock("已解析规则", values, "logic-text");
}

function renderProofBlock(steps) {
  const block = document.createElement("section");
  block.className = "result-block";
  block.append(createElement("h3", "证明轨迹"));

  if (!steps.length) {
    block.append(createElement("p", "当前没有可展示的证明步骤。"));
    return block;
  }

  const list = document.createElement("ol");
  for (const step of steps) {
    const item = document.createElement("li");
    const expression = createElement("span", step.derived);
    expression.className = "logic-text";
    const reason = createElement("span", `：${step.reason}`);
    item.append(expression, reason);
    if (step.source_text) {
      item.append(createElement("span", `（依据：${step.source_text}）`));
    }
    list.append(item);
  }
  block.append(list);
  return block;
}

function renderResultDetails(payload) {
  const fragment = document.createDocumentFragment();
  fragment.append(
    renderListBlock("已解析事实", payload.parsed_facts ?? [], "logic-text"),
    renderRulesBlock(payload.parsed_rules ?? []),
    renderProofBlock(payload.proof_steps ?? []),
  );

  if (payload.known_literals?.length) {
    fragment.append(
      renderListBlock("已知命题", payload.known_literals, "logic-text"),
    );
  }

  if (payload.conflict?.length) {
    fragment.append(renderListBlock("检测到的冲突", payload.conflict, "logic-text"));
  }
  return fragment;
}

function renderConfirmationFields(requests) {
  confirmationFields.replaceChildren();
  for (const [index, request] of requests.entries()) {
    const card = document.createElement("section");
    card.className = "confirmation-card";
    card.dataset.sourceSentence = request.source_sentence;

    const heading = createElement("h3", `复杂条件 ${index + 1}`);
    const source = createElement("p", `原句：${request.source_sentence}`);
    source.className = "logic-text";
    const message = createElement("p", request.message);
    message.className = "risk-message";

    const grid = document.createElement("div");
    grid.className = "confirmation-grid";
    grid.append(
      createConfirmationField(
        `confirmation-facts-${index}`,
        "确认后的事实",
        "每行一个，例如：!甲参加",
      ),
      createConfirmationField(
        `confirmation-rules-${index}`,
        "确认后的规则",
        "每行一条，例如：!甲参加 -> 乙通过",
      ),
    );

    card.append(heading, source, message, grid);
    confirmationFields.append(card);
  }
}

function createConfirmationField(id, labelText, placeholder) {
  const wrapper = document.createElement("div");
  const label = document.createElement("label");
  label.htmlFor = id;
  label.textContent = labelText;
  const textarea = document.createElement("textarea");
  textarea.id = id;
  textarea.name = id;
  textarea.rows = 4;
  textarea.placeholder = placeholder;
  wrapper.append(label, textarea);
  return wrapper;
}

function parseNonEmptyLines(value) {
  return value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function collectConfirmations() {
  return pendingConfirmationRequests.map((request, index) => {
    const factsInput = document.querySelector(`#confirmation-facts-${index}`);
    const rulesInput = document.querySelector(`#confirmation-rules-${index}`);
    const facts = parseNonEmptyLines(factsInput.value);
    const rules = parseNonEmptyLines(rulesInput.value).map((line) => {
      const parts = line.split("->");
      if (parts.length !== 2 || !parts[0].trim() || !parts[1].trim()) {
        throw new Error(`第 ${index + 1} 条复杂条件中的规则格式应为“前提 -> 结论”。`);
      }
      return { premise: parts[0].trim(), conclusion: parts[1].trim() };
    });

    if (!facts.length && !rules.length) {
      throw new Error(`请完成第 ${index + 1} 条复杂条件的结构化确认。`);
    }

    return { source_sentence: request.source_sentence, facts, rules };
  });
}

function messageFromResponse(payload, fallback) {
  if (typeof payload?.detail === "string") {
    return payload.detail;
  }
  return fallback;
}

async function requestSolve(confirmations = []) {
  const response = await fetch("/v1/questions/solve-chinese", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conditions: conditionsInput.value.trim(),
      query: queryInput.value.trim(),
      confirmations,
    }),
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(messageFromResponse(payload, "验证请求未成功，请稍后重试。"));
  }
  return payload;
}

function showConfirmationRequired(payload) {
  pendingConfirmationRequests = payload.confirmation_requests;
  renderConfirmationFields(pendingConfirmationRequests);
  confirmationPanel.hidden = false;
  setResultState({
    title: "需要人工确认",
    badge: "尚未验证",
    status: "pending",
    summary: payload.conclusion,
    details: renderResultDetails(payload),
  });
  confirmationPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  document.querySelector("#confirmation-facts-0")?.focus();
}

function showVerifiedResult(payload) {
  pendingConfirmationRequests = [];
  confirmationPanel.hidden = true;
  const labels = {
    proved: "已证明",
    disproved: "已反证",
    unknown: "无法推出",
    inconsistent: "条件冲突",
  };
  setResultState({
    title: "验证完成",
    badge: labels[payload.status] ?? "已处理",
    status: payload.status,
    summary: `${payload.conclusion} 验证等级：${payload.verification_level}。`,
    details: renderResultDetails(payload),
  });
  document.querySelector("#result-panel").scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}

async function handleSolve(event) {
  event.preventDefault();
  setError(formError, "");
  confirmationPanel.hidden = true;

  if (!conditionsInput.value.trim() || !queryInput.value.trim()) {
    setError(formError, "请先填写条件和待验证命题。");
    return;
  }

  setLoading(solveButton, true);
  try {
    const payload = await requestSolve();
    if (payload.confirmation_required) {
      showConfirmationRequired(payload);
    } else {
      showVerifiedResult(payload);
    }
  } catch (error) {
    setError(formError, safeText(error.message));
    setResultState({
      title: "无法验证",
      badge: "输入待修正",
      status: "error",
      summary: safeText(error.message),
    });
  } finally {
    setLoading(solveButton, false);
  }
}

async function handleConfirmation(event) {
  event.preventDefault();
  setError(confirmationError, "");
  let confirmations;
  try {
    confirmations = collectConfirmations();
  } catch (error) {
    setError(confirmationError, safeText(error.message));
    return;
  }

  setLoading(confirmationButton, true);
  try {
    const payload = await requestSolve(confirmations);
    if (payload.confirmation_required) {
      showConfirmationRequired(payload);
      return;
    }
    showVerifiedResult(payload);
  } catch (error) {
    setError(confirmationError, safeText(error.message));
  } finally {
    setLoading(confirmationButton, false);
  }
}

function fillExample() {
  conditionsInput.value = "甲参加。若甲参加，则乙通过。只有丙入选，乙才通过。";
  queryInput.value = "丙入选";
  setError(formError, "");
  conditionsInput.focus();
}

solveForm.addEventListener("submit", handleSolve);
confirmationForm.addEventListener("submit", handleConfirmation);
exampleButton.addEventListener("click", fillExample);
