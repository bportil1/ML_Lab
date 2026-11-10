/* P5-13: browser-local navigation context, never scientific data or authority. */
(() => {
  "use strict";
  const root = document.body;
  if (!root || !root.dataset.workspaceScope) return;
  const key = "ml-lab.workspace.v1." + root.dataset.workspaceScope;
  const validTools = new Set(["data", "models", "representation", "ebm", "optimization", "generative", "xml", "results"]);
  const empty = () => ({ experiment_id: "", run_id: "", active_tool: "" });
  let state = empty();
  let storageReady = false;
  try {
    const saved = JSON.parse(localStorage.getItem(key) || "null");
    if (saved && typeof saved === "object") {
      for (const name of ["experiment_id", "run_id", "active_tool"]) {
        if (typeof saved[name] === "string" && saved[name].length < 130) state[name] = saved[name];
      }
    }
    storageReady = true;
  } catch (_error) { /* Private mode: navigation still works without persistence. */ }
  const save = () => {
    if (!storageReady) return;
    try { localStorage.setItem(key, JSON.stringify(state)); } catch (_error) { storageReady = false; }
  };
  const currentTool = root.dataset.workspaceCurrent;
  const formOrUrl = new URLSearchParams(location.search);
  const currentExperiment = formOrUrl.get("experiment_id");
  if (currentExperiment && currentExperiment.length < 130) {
    if (currentExperiment !== state.experiment_id) state.run_id = "";
    state.experiment_id = currentExperiment;
  }
  // A run detail is authoritative only after /workspace/context verifies it.
  if (formOrUrl.get("run_id") && formOrUrl.get("run_id").length < 130) state.run_id = formOrUrl.get("run_id");
  if (location.hash === "#model-optimization") state.active_tool = "optimization";
  else if (currentTool && validTools.has(currentTool)) state.active_tool = currentTool;
  document.querySelectorAll(".scientific-tool-link").forEach(link => {
    const selected = link.dataset.workspaceTool === state.active_tool;
    link.classList.toggle("active", selected);
    if (selected) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  save();

  const contextURL = root.dataset.workspaceContextUrl;
  let lastRequest = 0;
  const picker = document.querySelector("[data-workspace-picker]");
  const experiments = picker?.querySelector("[data-workspace-experiment-select]");
  const runs = picker?.querySelector("[data-workspace-run-select]");
  const summary = picker?.querySelector("[data-workspace-summary]");
  const clearRuns = () => {
    if (!runs) return;
    runs.replaceChildren(new Option("No run selected", ""));
    runs.disabled = true;
  };
  const loadContext = async () => {
    if (!experiments || !summary || !contextURL) return;
    const id = state.experiment_id;
    const version = ++lastRequest;
    if (!id) {
      clearRuns();
      summary.textContent = "Choose an experiment to see its saved dataset, model configuration, and recorded runs.";
      return;
    }
    summary.textContent = "Reading selected experiment…";
    const url = new URL(contextURL, location.href);
    url.searchParams.set("experiment_id", id);
    if (state.run_id) url.searchParams.set("run_id", state.run_id);
    try {
      let response = await fetch(url.toString(), { headers: { Accept: "application/json" }, cache: "no-store" });
      // The run can become invalid while an experiment remains valid.
      if (response.status === 404 && state.run_id) {
        state.run_id = ""; save();
        url.searchParams.delete("run_id");
        response = await fetch(url.toString(), { headers: { Accept: "application/json" }, cache: "no-store" });
      }
      if (version !== lastRequest) return;
      if (!response.ok) {
        state.experiment_id = ""; state.run_id = ""; save();
        experiments.value = ""; clearRuns();
        summary.textContent = "The saved selection is no longer available in this project. Choose another experiment.";
        return;
      }
      const result = await response.json();
      if (result.experiment_id !== id) return;
      if (runs) {
        clearRuns();
        for (const item of result.runs) {
          const option = new Option(item.run_id.slice(0, 12) + " · rev " + item.revision + " · " + item.status, item.run_id);
          runs.add(option);
        }
        runs.disabled = result.runs.length === 0;
        if (result.runs.some(item => item.run_id === state.run_id)) runs.value = state.run_id;
        else { state.run_id = ""; runs.value = ""; save(); }
      }
      summary.textContent = result.name + " · revision " + result.revision +
        (result.dataset ? " · dataset: " + result.dataset : "") +
        (result.model_family ? " · model: " + result.model_family + "/" + result.model_method : " · no saved model");
    } catch (_error) {
      if (version === lastRequest) summary.textContent = "Context lookup unavailable; use the tools directly. No experiment was changed.";
    }
  };
  if (experiments) {
    if (Array.from(experiments.options).some(item => item.value === state.experiment_id)) experiments.value = state.experiment_id;
    else { state.experiment_id = ""; state.run_id = ""; save(); }
    experiments.addEventListener("change", () => {
      state.experiment_id = experiments.value; state.run_id = ""; save(); loadContext();
    });
    runs?.addEventListener("change", () => { state.run_id = runs.value; save(); });
    loadContext();
  }
  const runMarker = document.querySelector("[data-workspace-current-run-id]");
  if (runMarker && runMarker.dataset.workspaceCurrentRunId) {
    const runID = runMarker.dataset.workspaceCurrentRunId;
    const expID = runMarker.dataset.workspaceCurrentExperimentId;
    if (expID && expID.length < 130) {
      if (state.experiment_id !== expID) state.run_id = "";
      state.experiment_id = expID;
      if (runID.length < 130) state.run_id = runID;
      save();
    }
  }
  document.querySelectorAll("a[data-workspace-run-id]").forEach(link => {
    link.addEventListener("click", () => {
      const runID = link.dataset.workspaceRunId;
      const expID = link.dataset.workspaceExperimentId;
      if (runID && expID && runID.length < 130 && expID.length < 130) {
        state.experiment_id = expID; state.run_id = runID; save();
      }
    });
  });
  const actualFormPicker = document.querySelector("[data-workspace-exp-picker]");
  if (actualFormPicker && !actualFormPicker.value && state.experiment_id &&
      Array.from(actualFormPicker.options).some(item => item.value === state.experiment_id)) {
    actualFormPicker.value = state.experiment_id;
  }

  document.addEventListener("change", event => {
    const select = event.target;
    if (!(select instanceof HTMLSelectElement) || select === experiments || select === actualFormPicker) return;
    if (select.name !== "experiment_id" || !select.value || select.value.length >= 130) return;
    if (state.experiment_id !== select.value) state.run_id = "";
    state.experiment_id = select.value; save();
  });
  document.addEventListener("submit", event => {
    if (!(event.target instanceof HTMLFormElement)) return;
    const control = event.target.elements.namedItem("experiment_id");
    if (control && "value" in control && control.value && control.value.length < 130) {
      if (state.experiment_id !== control.value) state.run_id = "";
      state.experiment_id = control.value; save();
    }
  });
  document.querySelectorAll("a[data-workspace-tool]").forEach(link => {
    const id = link.dataset.workspaceTool;
    if (!validTools.has(id)) return;
    link.addEventListener("click", event => {
      if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      state.active_tool = id;
      save();
      // Scientific tool routes validate these IDs inside the project-owned DB.
      if (!state.experiment_id || !["models", "representation", "ebm", "optimization", "generative", "results"].includes(id)) return;
      const url = new URL(link.href, location.href);
      url.searchParams.set("experiment_id", state.experiment_id);
      link.href = url.toString();
    });
  });
})();
