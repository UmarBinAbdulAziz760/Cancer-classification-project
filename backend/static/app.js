// Plain vanilla JS - no build step, no framework, no Node.js required.
// Talks to the FastAPI backend served from the same origin (see main.py).

const state = {
  classificationTypes: [],
  selectedKey: null,
  file: null,
};

const el = {
  statusMessage: document.getElementById("status-message"),
  classificationTypeSelect: document.getElementById("classification-type-select"),
  domainInfoSection: document.getElementById("domain-info-section"),
  domainLabel: document.getElementById("domain-label"),
  domainHint: document.getElementById("domain-hint"),
  domainClasses: document.getElementById("domain-classes"),
  uploadZone: document.getElementById("upload-zone"),
  fileInput: document.getElementById("file-input"),
  uploadPlaceholder: document.getElementById("upload-placeholder"),
  uploadPreview: document.getElementById("upload-preview"),
  classifyButton: document.getElementById("classify-button"),
  resultSection: document.getElementById("result-section"),
  lowConfidenceWarning: document.getElementById("low-confidence-warning"),
  predictedClass: document.getElementById("predicted-class"),
  confidence: document.getElementById("confidence"),
  originalImage: document.getElementById("original-image"),
  gradcamImage: document.getElementById("gradcam-image"),
  modelCaption: document.getElementById("model-caption"),
  probabilityBars: document.getElementById("probability-bars"),
  disclaimer: document.getElementById("disclaimer"),
};

function showStatus(message, kind = "info") {
  el.statusMessage.textContent = message;
  el.statusMessage.className = `status-message ${kind}`;
  el.statusMessage.classList.remove("hidden");
}

function hideStatus() {
  el.statusMessage.classList.add("hidden");
}

function currentType() {
  return state.classificationTypes.find((t) => t.key === state.selectedKey) || null;
}

async function loadClassificationTypes() {
  try {
    const response = await fetch("/api/classification-types");
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    const data = await response.json();
    state.classificationTypes = data.classification_types || [];
  } catch (err) {
    showStatus(
      `Could not reach the backend (${err.message}). Make sure it's running - see backend/README.md.`,
      "error"
    );
    return;
  }

  if (state.classificationTypes.length === 0) {
    showStatus("No classification types are configured on the backend.", "error");
    return;
  }

  renderClassificationTypeSelect();

  const firstAvailable = state.classificationTypes.find((t) => t.available);
  state.selectedKey = (firstAvailable || state.classificationTypes[0]).key;
  el.classificationTypeSelect.value = state.selectedKey;

  if (!firstAvailable) {
    showStatus(
      "No trained models are deployed yet. See models/README.md for where to place them.",
      "error"
    );
  } else {
    hideStatus();
  }

  renderDomainInfo();
  updateClassifyButtonState();
}

function renderClassificationTypeSelect() {
  el.classificationTypeSelect.innerHTML = "";
  state.classificationTypes.forEach((type) => {
    const option = document.createElement("option");
    option.value = type.key;
    option.textContent = type.available ? type.label : `${type.label} (unavailable)`;
    option.disabled = !type.available;
    el.classificationTypeSelect.appendChild(option);
  });
}

el.classificationTypeSelect.addEventListener("change", (e) => {
  state.selectedKey = e.target.value;
  renderDomainInfo();
  updateClassifyButtonState();
});

function renderDomainInfo() {
  const type = currentType();
  if (!type) {
    el.domainInfoSection.classList.add("hidden");
    return;
  }
  el.domainLabel.textContent = `${type.label}: ${type.dataset_label}`;
  el.domainHint.textContent = type.dataset_hint;
  el.domainClasses.textContent = type.classes.length
    ? `Possible results: ${type.classes.join(", ")}`
    : "";
  el.domainInfoSection.classList.remove("hidden");
}

function updateClassifyButtonState() {
  const type = currentType();
  el.classifyButton.disabled = !(state.file && type && type.available);
}

function handleFileSelected(file) {
  if (!file) return;
  state.file = file;
  el.uploadPreview.src = URL.createObjectURL(file);
  el.uploadPreview.classList.remove("hidden");
  el.uploadPlaceholder.classList.add("hidden");
  updateClassifyButtonState();
}

el.uploadZone.addEventListener("click", () => el.fileInput.click());
el.fileInput.addEventListener("change", (e) => handleFileSelected(e.target.files?.[0]));

el.uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  el.uploadZone.classList.add("drag-active");
});
el.uploadZone.addEventListener("dragleave", () => el.uploadZone.classList.remove("drag-active"));
el.uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  el.uploadZone.classList.remove("drag-active");
  handleFileSelected(e.dataTransfer.files?.[0]);
});

function renderProbabilityBars(probabilities, predictedClass) {
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  el.probabilityBars.innerHTML = "";

  entries.forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "probability-row";

    const isPredicted = label === predictedClass;
    const pct = (value * 100).toFixed(1);

    row.innerHTML = `
      <span class="probability-label ${isPredicted ? "is-predicted" : ""}">${label}</span>
      <div class="probability-track">
        <div class="probability-fill ${isPredicted ? "is-predicted" : ""}" style="width: 0%"></div>
      </div>
      <span class="probability-value">${pct}%</span>
    `;
    el.probabilityBars.appendChild(row);

    // animate after insertion so the CSS transition actually plays
    requestAnimationFrame(() => {
      row.querySelector(".probability-fill").style.width = `${pct}%`;
    });
  });
}

async function classify() {
  const type = currentType();
  if (!state.file || !type) return;

  el.classifyButton.disabled = true;
  el.classifyButton.classList.add("loading");
  hideStatus();

  const formData = new FormData();
  formData.append("file", state.file);
  formData.append("classification_type", type.key);

  try {
    const response = await fetch("/api/predict", { method: "POST", body: formData });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || `Server returned ${response.status}`);
    }

    if (body.low_confidence) {
      el.lowConfidenceWarning.textContent = `\u26a0\ufe0f ${body.low_confidence_warning}`;
      el.lowConfidenceWarning.classList.remove("hidden");
    } else {
      el.lowConfidenceWarning.classList.add("hidden");
    }

    el.predictedClass.textContent = body.predicted_class;
    el.confidence.textContent = `${(body.confidence * 100).toFixed(1)}%`;
    el.originalImage.src = body.original_image;
    el.gradcamImage.src = body.gradcam_image;
    el.modelCaption.textContent = "Uploaded image";
    renderProbabilityBars(body.probabilities, body.predicted_class);
    el.disclaimer.textContent = body.disclaimer;
    el.resultSection.classList.remove("hidden");
    el.resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    showStatus(err.message, "error");
  } finally {
    el.classifyButton.disabled = false;
    el.classifyButton.classList.remove("loading");
  }
}

el.classifyButton.addEventListener("click", classify);

loadClassificationTypes();
