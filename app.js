"use strict";

const SAMPLE_TEXT = "namast aap kaise ho";
const MAX_TEXT_LENGTH = 2_000;

const elements = {
  source: document.querySelector("#source-text"),
  count: document.querySelector("#char-count"),
  inputError: document.querySelector("#input-error"),
  run: document.querySelector("#run-preview"),
  loadExample: document.querySelector("#load-example"),
  clear: document.querySelector("#clear-input"),
  connection: document.querySelector("#connection-pill"),
  connectionLabel: document.querySelector("#connection-label"),
  resultContent: document.querySelector("#result-content"),
  empty: document.querySelector("#empty-state"),
  loading: document.querySelector("#loading-state"),
  success: document.querySelector("#success-state"),
  error: document.querySelector("#error-state"),
  resultError: document.querySelector("#result-error-message"),
  resultStatus: document.querySelector("#result-status"),
  resultCount: document.querySelector("#result-count"),
  phase1: document.querySelector("#phase1-output"),
  phase2: document.querySelector("#phase2-output"),
  phase2Decision: document.querySelector("#phase2-decision"),
  phase3: document.querySelector("#phase3-output"),
  traceCount: document.querySelector("#trace-count"),
  traceList: document.querySelector("#trace-list"),
  transformerDecision: document.querySelector("#transformer-decision"),
  copy: document.querySelector("#copy-result"),
};

function setResultState(state) {
  elements.empty.hidden = state !== "empty";
  elements.loading.hidden = state !== "loading";
  elements.success.hidden = state !== "success";
  elements.error.hidden = state !== "error";
  elements.resultContent.setAttribute("aria-busy", String(state === "loading"));
}

function updateCharacterCount() {
  const count = elements.source.value.length;
  elements.count.textContent = `${count.toLocaleString()} / ${MAX_TEXT_LENGTH.toLocaleString()} characters`;
  elements.inputError.hidden = true;
}

function renderTrace(steps) {
  const fragment = document.createDocumentFragment();

  for (const step of steps) {
    const row = document.createElement("li");
    row.className = `trace-item${step.original !== step.corrected ? " is-changed" : ""}`;

    const words = document.createElement("span");
    words.className = "trace-word";
    words.append(document.createTextNode(step.original));
    if (step.original !== step.corrected) {
      words.append(document.createTextNode(" → "));
      const corrected = document.createElement("strong");
      corrected.textContent = step.corrected;
      words.append(corrected);
    }

    const detail = document.createElement("span");
    detail.className = "trace-detail";
    detail.textContent = step.distance === 0
      ? "In lexicon"
      : `${step.distance} edit${step.distance === 1 ? "" : "s"}`;
    detail.title = `${step.decision}${step.candidates.length ? ` · Candidates: ${step.candidates.join(", ")}` : ""}`;
    row.append(words, detail);
    fragment.append(row);
  }

  elements.traceList.replaceChildren(fragment);
}

function updateBackendStatus() {
  fetch("/api/status", { cache: "no-store" })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error("The Python backend did not respond with a valid status.");
      }
      return response.json();
    })
    .then((status) => {
      elements.connection.dataset.state = status.status;
      elements.connectionLabel.textContent = status.stage;
      elements.run.disabled = status.status !== "ready";

      if (status.status === "ready") {
        return;
      }
      if (status.status !== "error") {
        window.setTimeout(updateBackendStatus, 2500);
      }
    })
    .catch(() => {
      elements.connection.dataset.state = "offline";
      elements.connectionLabel.textContent = "Backend offline · run python server.py";
      elements.run.disabled = true;
      window.setTimeout(updateBackendStatus, 4000);
    });
}

async function runCorrection() {
  const text = elements.source.value;
  if (!text.trim()) {
    elements.inputError.textContent = "Enter some text before running correction.";
    elements.inputError.hidden = false;
    elements.source.focus();
    return;
  }
  if (text.length > MAX_TEXT_LENGTH) {
    elements.inputError.textContent = `Text must be ${MAX_TEXT_LENGTH.toLocaleString()} characters or fewer.`;
    elements.inputError.hidden = false;
    elements.source.focus();
    return;
  }

  elements.inputError.hidden = true;
  setResultState("loading");

  try {
    const response = await fetch("/api/correct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const payload = await response.json();
    if (!response.ok) {
      const message = response.status === 404
        ? "The Python backend is not running. Start it with: python server.py"
        : payload.error || "The correction request failed.";
      throw new Error(message);
    }

    elements.phase1.textContent = payload.phase1;
    elements.phase2.textContent = payload.phase2;
    elements.phase2Decision.textContent = payload.phase2_steps.some((step) =>
      step.decision.includes("safeguard"),
    )
      ? "Lexicon safeguard applied where the GRU output was not trustworthy."
      : "Output passed the sample-lexicon checks.";
    elements.phase3.textContent = payload.phase3;
    elements.resultStatus.textContent = "Completed all 3 correction phases";
    elements.resultCount.textContent = `${payload.phase1_steps.length} input word${payload.phase1_steps.length === 1 ? "" : "s"}`;
    elements.traceCount.textContent = `${payload.phase1_steps.length} TOKEN${payload.phase1_steps.length === 1 ? "" : "S"}`;
    elements.transformerDecision.textContent = payload.phase3_trace.decision;
    renderTrace(payload.phase1_steps);
    setResultState("success");
  } catch (error) {
    elements.resultError.textContent = error instanceof Error
      ? error.message
      : "Correction failed. Check that the Python backend is running.";
    setResultState("error");
  }
}

elements.source.addEventListener("input", updateCharacterCount);
elements.run.addEventListener("click", runCorrection);
elements.loadExample.addEventListener("click", () => {
  elements.source.value = SAMPLE_TEXT;
  updateCharacterCount();
  elements.source.focus();
});
elements.clear.addEventListener("click", () => {
  elements.source.value = "";
  updateCharacterCount();
  setResultState("empty");
  elements.traceList.replaceChildren();
  elements.source.focus();
});
elements.copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(elements.phase3.textContent);
    elements.copy.setAttribute("aria-label", "Phase 3 result copied");
    elements.copy.title = "Copied";
    window.setTimeout(() => {
      elements.copy.setAttribute("aria-label", "Copy Phase 3 result");
      elements.copy.title = "Copy Phase 3 result";
    }, 1400);
  } catch {
    elements.copy.setAttribute("aria-label", "Clipboard unavailable; select and copy the Phase 3 result");
    elements.copy.title = "Clipboard unavailable; select and copy the result";
  }
});

updateCharacterCount();
updateBackendStatus();
