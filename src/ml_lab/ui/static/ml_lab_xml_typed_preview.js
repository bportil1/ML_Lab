(() => {
  "use strict";
  const panel = document.querySelector("[data-xml-typed-workspace]");
  const payloadNode = document.getElementById("xml-structure-data");
  if (!panel || !payloadNode) return;
  let structure;
  try { structure = JSON.parse(payloadNode.textContent || "{}"); } catch (_) { return; }
  const profiles = Array.isArray(structure.element_profiles) ? structure.element_profiles : [];
  const candidates = Array.isArray(structure.record_candidates) ? structure.record_candidates : [];
  const root = panel.querySelector("[data-xml-typed-root]");
  const tbody = panel.querySelector("[data-xml-typed-variables]");
  const button = panel.querySelector("[data-xml-typed-run]");
  const status = panel.querySelector("[data-xml-typed-status]");
  const saveButton = panel.querySelector("[data-xml-typed-save]");
  const materializeButton = panel.querySelector("[data-xml-typed-materialize]");
  const artifact = panel.querySelector("[data-xml-typed-artifact]");
  const details = panel.querySelector("[data-xml-typed-result]");
  const output = panel.querySelector("[data-xml-typed-json]");
  const limit = panel.querySelector("[data-xml-typed-limit]");
  if (!profiles.length) return;
  // Structure paths are canonical, and selection is independent of legacy C1--C7.
  const roots = candidates.length ? candidates : profiles;
  roots.forEach((candidate) => {
    const opt = document.createElement("option");
    opt.value = candidate.canonical_path;
    opt.textContent = candidate.path || candidate.canonical_path;
    root.appendChild(opt);
  });
  const kinds = ["scalar", "object", "list", "content_tokens", "path_tokens"];
  const clearResult = () => { details.hidden = true; output.textContent = ""; artifact.textContent = ""; status.textContent = "Nothing has been extracted."; };
  const selection = (maxRecords) => {
    const variables = Array.from(tbody.querySelectorAll("tr"))
      .filter(tr => tr.querySelector("input").checked)
      .map(tr => ({canonical_path: tr.dataset.path, representation: tr.querySelector("select").value}));
    if (!variables.length || variables.length > 12) throw new Error("Select 1–12 fields.");
    return {record_root_canonical_path: root.value, variables, max_records: maxRecords};
  };
  const submit = async (url, payload) => {
    const response = await fetch(url, {
      method: "POST", headers: {"Content-Type": "application/json"}, credentials: "same-origin",
      body: JSON.stringify(payload)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Typed XML request failed");
    return result;
  };
  let busy = false;
  const runAction = async (action, label, url, maxRecords) => {
    if (busy) return;
    let payload;
    try { payload = selection(maxRecords); } catch (error) { status.textContent = error.message; return; }
    if (action === "materialize" && !window.confirm("Materialize all matching records (up to 5,000, 32 MiB) as a separate, immutable typed artifact? This does not register a dataset.")) return;
    busy = true; button.disabled = true; saveButton.disabled = true; materializeButton.disabled = true;
    status.textContent = label; artifact.textContent = "";
    try {
      const result = await submit(url, payload);
      if (action === "preview") {
        output.textContent = JSON.stringify(result, null, 2);
        details.hidden = false;
        status.textContent = `Previewed ${result.preview_record_count} of ${result.source_record_count} source records; no dataset created.`;
      } else {
        status.textContent = action === "save" ? `Saved bounded preview (${result.record_count} records).` : `Materialized ${result.record_count} of ${result.record_count} source records (complete).`;
        artifact.textContent = `Immutable artifact: ${result.artifact_id}; project-relative manifest: ${result.relative_manifest_path || result.relative_path}. No dataset registered.`;
      }
    } catch (error) { status.textContent = `${action === "preview" ? "Preview" : "Publication"} failed: ${error.message}`; }
    finally { busy = false; button.disabled = false; saveButton.disabled = false; materializeButton.disabled = false; }
  };
  const renderFields = () => {
    clearResult();
    tbody.replaceChildren();
    const prefix = root.value + "/";
    // Exactly one canonical root, no string prefix confusion such as /item vs /items.
    profiles.filter(p => p.canonical_path === root.value || p.canonical_path.startsWith(prefix)).forEach((profile) => {
      const tr = document.createElement("tr");
      const box = document.createElement("input"); box.type = "checkbox"; box.setAttribute("aria-label", `Include ${profile.path}`);
      const select = document.createElement("select"); select.setAttribute("aria-label", `Representation for ${profile.path}`);
      kinds.forEach((kind) => { const opt = document.createElement("option"); opt.value = kind; opt.textContent = kind.replaceAll("_", " "); select.appendChild(opt); });
      if (profile.repeated || profile.cardinality?.repeated) select.value = "list";
      else if (profile.mixed_content) select.value = "content_tokens";
      else if (profile.child_path_count || profile.children?.length) select.value = "object";
      const td0 = document.createElement("td"); td0.appendChild(box);
      const td1 = document.createElement("td"); td1.textContent = profile.path || profile.canonical_path;
      const td2 = document.createElement("td"); td2.appendChild(select);
      tr.append(td0, td1, td2); tr.dataset.path = profile.canonical_path;
      tbody.appendChild(tr);
    });
  };
  root.addEventListener("change", renderFields);
  renderFields();
  button.addEventListener("click", () => {
    details.hidden = true; output.textContent = "";
    runAction("preview", "Reading a bounded preview…", panel.dataset.typedPreviewUrl, Number(limit.value));
  });
  saveButton.addEventListener("click", () => runAction("save", "Saving verified preview…", panel.dataset.typedSaveUrl, Number(limit.value)));
  materializeButton.addEventListener("click", () => runAction("materialize", "Materializing complete typed records…", panel.dataset.typedMaterializeUrl, 5000));
})();
