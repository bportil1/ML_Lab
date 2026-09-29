(() => {
  "use strict";

  const explorer = document.querySelector("[data-xml-explorer]");
  const payloadNode = document.getElementById("xml-structure-data");
  if (!explorer || !payloadNode) return;

  let payload;
  try {
    payload = JSON.parse(payloadNode.textContent || "{}");
  } catch (_error) {
    return;
  }

  const profiles = Array.isArray(payload.element_profiles) ? payload.element_profiles : [];
  const profileByPath = new Map(profiles.map((profile) => [profile.canonical_path, profile]));
  const candidatePaths = new Set((payload.record_candidates || []).map((candidate) => candidate.canonical_path));
  const tree = explorer.querySelector("[data-xml-tree]");
  const search = explorer.querySelector("[data-xml-tree-search]");
  const candidatesOnly = explorer.querySelector("[data-xml-candidates-only]");
  const status = explorer.querySelector("[data-xml-tree-status]");

  const parentCanonicalPath = (canonicalPath) => {
    const index = canonicalPath.lastIndexOf("/");
    return index > 0 ? canonicalPath.slice(0, index) : null;
  };

  const childrenByParent = new Map();
  profiles.forEach((profile) => {
    const parent = parentCanonicalPath(profile.canonical_path);
    if (!childrenByParent.has(parent)) childrenByParent.set(parent, []);
    childrenByParent.get(parent).push(profile);
  });

  const formatNumber = (value) => {
    if (value === null || value === undefined) return "—";
    if (typeof value === "number" && !Number.isInteger(value)) return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
    return String(value);
  };

  const formatRate = (value) => {
    if (value === null || value === undefined) return "—";
    return `${(Number(value) * 100).toFixed(1)}%`;
  };

  const labelFor = (profile) => profile.prefix ? `${profile.prefix}:${profile.local_name}` : profile.local_name;

  const appendBadge = (row, text, className = "") => {
    const badge = document.createElement("span");
    badge.className = `xml-node-badge ${className}`.trim();
    badge.textContent = text;
    row.appendChild(badge);
  };

  const nodeElements = new Map();

  const buildNode = (profile) => {
    const li = document.createElement("li");
    li.className = "xml-tree-item";
    li.dataset.xmlCanonicalPath = profile.canonical_path;
    li.dataset.xmlSearchText = `${profile.path} ${profile.local_name} ${profile.prefix || ""}`.toLowerCase();
    li.setAttribute("role", "treeitem");

    const row = document.createElement("div");
    row.className = "xml-tree-row";
    const children = childrenByParent.get(profile.canonical_path) || [];

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "xml-tree-toggle";
    if (children.length) {
      toggle.textContent = "▸";
      toggle.setAttribute("aria-label", `Expand ${labelFor(profile)}`);
      toggle.setAttribute("aria-expanded", "false");
    } else {
      toggle.textContent = "·";
      toggle.disabled = true;
      toggle.setAttribute("aria-hidden", "true");
    }
    row.appendChild(toggle);

    const select = document.createElement("button");
    select.type = "button";
    select.className = "xml-tree-select";
    select.dataset.xmlSelectPath = profile.canonical_path;
    const name = document.createElement("span");
    name.className = "xml-node-name";
    name.textContent = labelFor(profile);
    select.appendChild(name);
    const count = document.createElement("span");
    count.className = "xml-node-count";
    count.textContent = `× ${profile.occurrence_count}`;
    select.appendChild(count);
    row.appendChild(select);

    if (profile.cardinality && profile.cardinality.repeated) appendBadge(row, "repeated", "is-repeated");
    if (profile.cardinality && profile.cardinality.optional) appendBadge(row, "optional", "is-optional");
    if (candidatePaths.has(profile.canonical_path)) appendBadge(row, "record candidate", "is-candidate");
    li.appendChild(row);

    let childList = null;
    if (children.length) {
      childList = document.createElement("ul");
      childList.className = "xml-tree-children";
      childList.setAttribute("role", "group");
      childList.hidden = true;
      children.forEach((child) => childList.appendChild(buildNode(child)));
      li.appendChild(childList);
      toggle.addEventListener("click", () => {
        const expand = childList.hidden;
        childList.hidden = !expand;
        toggle.textContent = expand ? "▾" : "▸";
        toggle.setAttribute("aria-expanded", expand ? "true" : "false");
      });
    }

    nodeElements.set(profile.canonical_path, {li, row, toggle, childList});
    return li;
  };

  const rootList = document.createElement("ul");
  rootList.className = "xml-tree-roots";
  rootList.setAttribute("role", "group");
  (childrenByParent.get(null) || []).forEach((profile) => rootList.appendChild(buildNode(profile)));
  tree.replaceChildren(rootList);

  const expandAncestors = (canonicalPath) => {
    let parent = parentCanonicalPath(canonicalPath);
    while (parent) {
      const parts = nodeElements.get(parent);
      if (parts && parts.childList) {
        parts.childList.hidden = false;
        parts.toggle.textContent = "▾";
        parts.toggle.setAttribute("aria-expanded", "true");
      }
      parent = parentCanonicalPath(parent);
    }
  };

  const inspector = explorer.querySelector("[data-xml-inspector]");
  const inspectorEmpty = inspector.querySelector("[data-xml-inspector-empty]");
  const inspectorContent = inspector.querySelector("[data-xml-inspector-content]");
  let selectedPath = null;

  const setText = (selector, value) => {
    const node = inspector.querySelector(selector);
    if (node) node.textContent = value;
  };

  const inspectPath = (canonicalPath, {scroll = false} = {}) => {
    const profile = profileByPath.get(canonicalPath);
    if (!profile) return;
    selectedPath = canonicalPath;
    nodeElements.forEach(({row}, path) => row.classList.toggle("is-selected", path === canonicalPath));
    expandAncestors(canonicalPath);

    inspectorEmpty.hidden = true;
    inspectorContent.hidden = false;
    setText("[data-xml-inspector-name]", labelFor(profile));
    setText("[data-xml-inspector-path]", profile.path);
    setText("[data-xml-inspector-canonical]", profile.canonical_path);
    setText("[data-xml-inspector-namespace]", profile.namespace_uri || "none");
    setText("[data-xml-inspector-depth]", String(profile.depth));
    setText("[data-xml-inspector-occurrences]", String(profile.occurrence_count));

    const cardinality = profile.cardinality || {};
    setText("[data-xml-card-min]", formatNumber(cardinality.min_per_parent));
    setText("[data-xml-card-mean]", formatNumber(cardinality.mean_per_parent));
    setText("[data-xml-card-max]", formatNumber(cardinality.max_per_parent));
    const flags = inspector.querySelector("[data-xml-flags]");
    flags.replaceChildren();
    if (cardinality.repeated) appendBadge(flags, "repeated within parent", "is-repeated");
    if (cardinality.optional) appendBadge(flags, "optional in parent", "is-optional");
    const text = profile.text || {};
    if (text.mixed_content) appendBadge(flags, "mixed content", "is-optional");
    if (!cardinality.repeated && !cardinality.optional && !text.mixed_content) appendBadge(flags, "single / required where observed");

    setText("[data-xml-text-rate]", `${text.occurrence_count || 0} / ${profile.occurrence_count} (${formatRate(text.presence_rate || 0)})`);
    setText("[data-xml-text-distinct]", `${text.distinct_sample_count || 0} / ${text.sampled_value_count || 0} (${formatRate(text.distinct_sample_rate || 0)})`);
    setText("[data-xml-text-id]", text.likely_identifier ? "yes" : "no");

    const attributesBody = inspector.querySelector("[data-xml-attributes]");
    const noAttributes = inspector.querySelector("[data-xml-no-attributes]");
    attributesBody.replaceChildren();
    const attributes = Array.isArray(profile.attributes) ? profile.attributes : [];
    noAttributes.hidden = attributes.length !== 0;
    attributes.forEach((attribute) => {
      const tr = document.createElement("tr");
      const displayName = attribute.prefix ? `${attribute.prefix}:${attribute.local_name}` : attribute.local_name;
      const cells = [
        displayName,
        `${attribute.occurrence_count} (${formatRate(attribute.presence_rate)})`,
        `${attribute.distinct_sample_count}/${attribute.sampled_value_count} (${formatRate(attribute.distinct_sample_rate)})`,
        attribute.likely_identifier ? "likely identifier" : "—",
      ];
      cells.forEach((value, index) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        if (index === 0) cell.className = "xml-monospace-cell";
        tr.appendChild(cell);
      });
      attributesBody.appendChild(tr);
    });

    const candidate = inspector.querySelector("[data-xml-inspector-candidate]");
    candidate.hidden = !candidatePaths.has(canonicalPath);
    setText("[data-xml-candidate-score]", Number(profile.record_candidate_score || 0).toFixed(2));
    const reasons = inspector.querySelector("[data-xml-candidate-reasons]");
    reasons.replaceChildren();
    const reasonValues = Array.isArray(profile.record_candidate_reasons) ? profile.record_candidate_reasons : [];
    if (!reasonValues.length) {
      const item = document.createElement("li");
      item.className = "muted";
      item.textContent = "No record-root evidence assigned to this path.";
      reasons.appendChild(item);
    } else {
      reasonValues.forEach((reason) => {
        const item = document.createElement("li");
        item.textContent = reason.replaceAll("_", " ");
        reasons.appendChild(item);
      });
    }

    if (scroll) {
      const parts = nodeElements.get(canonicalPath);
      if (parts) parts.row.scrollIntoView({behavior: "smooth", block: "center"});
    }
  };

  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-xml-select-path]");
    if (!target) return;
    const canonicalPath = target.dataset.xmlSelectPath;
    if (!profileByPath.has(canonicalPath)) return;
    inspectPath(canonicalPath, {scroll: !explorer.contains(target)});
  });

  const setExpanded = (expanded) => {
    nodeElements.forEach(({toggle, childList}) => {
      if (!childList) return;
      childList.hidden = !expanded;
      toggle.textContent = expanded ? "▾" : "▸";
      toggle.setAttribute("aria-expanded", expanded ? "true" : "false");
    });
  };
  explorer.querySelector("[data-xml-expand-all]")?.addEventListener("click", () => setExpanded(true));
  explorer.querySelector("[data-xml-collapse-all]")?.addEventListener("click", () => {
    setExpanded(false);
    if (selectedPath) expandAncestors(selectedPath);
  });

  const descendantMatches = (profile, query, candidatesMode) => {
    const parts = nodeElements.get(profile.canonical_path);
    if (!parts) return false;
    const selfMatchesQuery = !query || parts.li.dataset.xmlSearchText.includes(query);
    const selfMatchesCandidate = !candidatesMode || candidatePaths.has(profile.canonical_path);
    const children = childrenByParent.get(profile.canonical_path) || [];
    const childMatch = children.some((child) => descendantMatches(child, query, candidatesMode));
    const visible = (selfMatchesQuery && selfMatchesCandidate) || childMatch;
    parts.li.hidden = !visible;
    if (childMatch && parts.childList) {
      parts.childList.hidden = false;
      parts.toggle.textContent = "▾";
      parts.toggle.setAttribute("aria-expanded", "true");
    }
    return visible;
  };

  const applyFilter = () => {
    const query = (search?.value || "").trim().toLowerCase();
    const candidatesMode = Boolean(candidatesOnly?.checked);
    let rootsVisible = 0;
    (childrenByParent.get(null) || []).forEach((root) => {
      if (descendantMatches(root, query, candidatesMode)) rootsVisible += 1;
    });
    const visibleCount = [...nodeElements.values()].filter(({li}) => !li.hidden).length;
    if (status) status.textContent = `${visibleCount} of ${profiles.length} paths visible${candidatesMode ? " · candidate branches" : ""}${query ? ` · filter: ${query}` : ""}`;
    if (!query && !candidatesMode && rootsVisible) setExpanded(false);
  };

  search?.addEventListener("input", applyFilter);
  candidatesOnly?.addEventListener("change", applyFilter);
  applyFilter();

  const fieldSelector = document.querySelector("[data-xml-field-selector]");
  const rulePanel = document.querySelector("[data-xml-rule-panel]");
  const previewPanel = document.querySelector("[data-xml-preview-panel]");
  const recordFieldsUrl = explorer.dataset.recordFieldsUrl || "";
  const collectionPlanUrl = explorer.dataset.collectionPlanUrl || "";
  const previewUrl = explorer.dataset.previewUrl || "";
  const materializeUrl = explorer.dataset.materializeUrl || "";
  const extractionHistoryUrl = explorer.dataset.extractionHistoryUrl || "";
  const extractionReentryUrl = explorer.dataset.extractionReentryUrl || "";
  const extractionCompareUrl = explorer.dataset.extractionCompareUrl || "";
  const historyPanel = document.querySelector("[data-xml-extraction-history]");
  const sourceFingerprint = explorer.dataset.sourceFingerprint || payload.source_fingerprint || "";
  const storageKey = `ml_lab.xml.selection:${sourceFingerprint}`;
  let fieldCatalog = null;
  let selectedFieldIds = new Set();
  let collectionPlan = null;
  let collectionRules = new Map();
  let planRequestSerial = 0;
  let previewResult = null;
  let confirmedPreviewSignature = null;
  let materializationResult = null;

  const loadStoredSelection = () => {
    try {
      const stored = JSON.parse(localStorage.getItem(storageKey) || "null");
      if (!stored || stored.source_fingerprint !== sourceFingerprint) return null;
      return stored;
    } catch (_error) {
      return null;
    }
  };

  const rulePayload = (rule) => ({
    branch_canonical_path: rule.branch_canonical_path,
    strategy: rule.strategy || null,
    options: {
      join_delimiter: rule.options?.join_delimiter ?? null,
      aggregate_operation: rule.options?.aggregate_operation ?? null,
      pivot_key_field_id: rule.options?.pivot_key_field_id ?? null,
      pivot_value_field_id: rule.options?.pivot_value_field_id ?? null,
      separate_table_name: rule.options?.separate_table_name ?? null,
    },
  });

  const currentRulesPayload = () => [...collectionRules.values()].map(rulePayload);

  const saveStoredSelection = () => {
    if (!fieldCatalog) {
      try { localStorage.removeItem(storageKey); } catch (_error) {}
      return;
    }
    const state = {
      schema: "ml-lab.xml-extraction-selection@3",
      source_fingerprint: sourceFingerprint,
      record_root_canonical_path: fieldCatalog.record_root.canonical_path,
      selected_field_ids: [...selectedFieldIds],
      collection_rules: currentRulesPayload(),
      confirmed_preview_signature: confirmedPreviewSignature,
    };
    try { localStorage.setItem(storageKey, JSON.stringify(state)); } catch (_error) {}
  };

  const selectionRecord = () => {
    if (!fieldCatalog) return null;
    const selected = (fieldCatalog.fields || []).filter((field) => selectedFieldIds.has(field.field_id));
    return {
      schema: "ml-lab.xml-extraction-selection@3",
      source_fingerprint: sourceFingerprint,
      record_root: fieldCatalog.record_root,
      selected_fields: selected.map((field) => ({
        field_id: field.field_id,
        kind: field.kind,
        relative_path: field.relative_path,
        source_element_canonical_path: field.source_element_canonical_path,
        repeated: field.repeated,
        optional: field.optional,
        likely_identifier: field.likely_identifier,
      })),
      collection_rules: currentRulesPayload(),
      confirmed_preview_signature: confirmedPreviewSignature,
    };
  };

  const renderSelectionSummary = () => {
    if (!fieldSelector || !fieldCatalog) return;
    fieldSelector.querySelector("[data-xml-record-root-path]").textContent = fieldCatalog.record_root.path;
    fieldSelector.querySelector("[data-xml-record-root-occurrences]").textContent = String(fieldCatalog.record_root.occurrence_count);
    fieldSelector.querySelector("[data-xml-record-root-score]").textContent = Number(fieldCatalog.record_root.candidate_score || 0).toFixed(2);
    fieldSelector.querySelector("[data-xml-selected-field-count]").textContent = String(selectedFieldIds.size);
    fieldSelector.querySelector("[data-xml-selection-json]").textContent = JSON.stringify(selectionRecord(), null, 2);
    nodeElements.forEach(({row}, path) => row.classList.toggle("is-record-root", path === fieldCatalog.record_root.canonical_path));
  };

  const fieldShape = (field) => {
    const parts = [];
    if (field.repeated) parts.push("repeated");
    if (field.optional) parts.push("optional");
    if (!parts.length) parts.push("single");
    return parts.join(" · ");
  };

  const renderFieldRows = () => {
    if (!fieldSelector || !fieldCatalog) return;
    const body = fieldSelector.querySelector("[data-xml-field-rows]");
    const query = (fieldSelector.querySelector("[data-xml-field-search]")?.value || "").trim().toLowerCase();
    body.replaceChildren();
    let visible = 0;
    (fieldCatalog.fields || []).forEach((field) => {
      const haystack = `${field.relative_path} ${field.kind} ${field.source_element_path} ${field.name}`.toLowerCase();
      if (query && !haystack.includes(query)) return;
      visible += 1;
      const tr = document.createElement("tr");
      tr.dataset.xmlFieldId = field.field_id;

      const selectCell = document.createElement("td");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = selectedFieldIds.has(field.field_id);
      checkbox.setAttribute("aria-label", `Select ${field.relative_path}`);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) selectedFieldIds.add(field.field_id);
        else selectedFieldIds.delete(field.field_id);
        renderSelectionSummary();
        saveStoredSelection();
        refreshCollectionPlan();
      });
      selectCell.appendChild(checkbox);
      tr.appendChild(selectCell);

      const values = [
        field.relative_path,
        field.kind === "attribute" ? "attribute" : "element text",
        field.source_element_path,
        fieldShape(field),
        field.likely_identifier ? "likely identifier" : "—",
      ];
      values.forEach((value, index) => {
        const td = document.createElement("td");
        td.textContent = value;
        if (index === 0 || index === 2) td.className = "xml-monospace-cell";
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    fieldSelector.querySelector("[data-xml-field-status]").textContent = `${visible} of ${(fieldCatalog.fields || []).length} scalar fields visible`;
  };

  const normalizedRuleFromPlan = (branch) => {
    const record = branch.rule || {};
    return {
      branch_canonical_path: branch.canonical_path,
      strategy: record.strategy || null,
      options: {...(record.options || {})},
    };
  };

  const renderRuleOptions = (container, branch, rule) => {
    container.replaceChildren();
    const options = rule.options || {};
    const branchFields = branch.selected_fields || [];
    const makeLabel = (text) => {
      const label = document.createElement("label");
      label.className = "xml-rule-option";
      const span = document.createElement("span");
      span.textContent = text;
      label.appendChild(span);
      return label;
    };
    const updateOption = (name, value) => {
      const current = collectionRules.get(branch.canonical_path) || {branch_canonical_path: branch.canonical_path, strategy: rule.strategy, options: {}};
      current.options = {...(current.options || {}), [name]: value};
      collectionRules.set(branch.canonical_path, current);
      refreshCollectionPlan();
    };

    if (rule.strategy === "join") {
      const label = makeLabel("Delimiter");
      const input = document.createElement("input");
      input.type = "text";
      input.maxLength = 32;
      input.value = options.join_delimiter ?? ", ";
      input.addEventListener("change", () => updateOption("join_delimiter", input.value));
      label.appendChild(input);
      container.appendChild(label);
    } else if (rule.strategy === "aggregate") {
      const label = makeLabel("Operation");
      const select = document.createElement("select");
      [["", "Choose operation"], ["sum", "Sum"], ["mean", "Mean"], ["min", "Min"], ["max", "Max"]].forEach(([value, text]) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = text;
        option.selected = (options.aggregate_operation || "") === value;
        select.appendChild(option);
      });
      select.addEventListener("change", () => updateOption("aggregate_operation", select.value || null));
      label.appendChild(select);
      container.appendChild(label);
    } else if (rule.strategy === "pivot") {
      const makeFieldSelect = (optionName, labelText) => {
        const label = makeLabel(labelText);
        const select = document.createElement("select");
        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = `Choose ${labelText.toLowerCase()}`;
        select.appendChild(placeholder);
        branchFields.forEach((field) => {
          const option = document.createElement("option");
          option.value = field.field_id;
          option.textContent = field.relative_path;
          option.selected = options[optionName] === field.field_id;
          select.appendChild(option);
        });
        select.addEventListener("change", () => updateOption(optionName, select.value || null));
        label.appendChild(select);
        return label;
      };
      container.appendChild(makeFieldSelect("pivot_key_field_id", "Key field"));
      container.appendChild(makeFieldSelect("pivot_value_field_id", "Value field"));
    } else if (rule.strategy === "separate_table") {
      const label = makeLabel("Child table name (optional)");
      const input = document.createElement("input");
      input.type = "text";
      input.maxLength = 128;
      input.placeholder = branch.relative_path.split("/").filter(Boolean).pop() || "child_table";
      input.value = options.separate_table_name || "";
      input.addEventListener("change", () => updateOption("separate_table_name", input.value || null));
      label.appendChild(input);
      container.appendChild(label);
    }
  };

  const formatPreviewValue = (value) => {
    if (value === null || value === undefined) return "";
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  };

  const renderPreviewTable = (table, target) => {
    target.replaceChildren();
    if (!table) return;
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    (table.columns || []).forEach((column) => {
      const th = document.createElement("th");
      const sourcePath = (column.source_element_canonical_paths || [])[0];
      if (sourcePath && profileByPath.has(sourcePath)) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "xml-preview-column-link";
        button.textContent = column.name;
        button.title = "Inspect source XML node";
        button.addEventListener("click", () => {
          inspectPath(sourcePath, {scroll: true});
          explorer.scrollIntoView({behavior: "smooth", block: "start"});
        });
        th.appendChild(button);
      } else {
        th.textContent = column.name;
      }
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    target.appendChild(thead);

    const tbody = document.createElement("tbody");
    (table.rows || []).forEach((row) => {
      const tr = document.createElement("tr");
      (table.columns || []).forEach((column) => {
        const td = document.createElement("td");
        const value = formatPreviewValue(row[column.name]);
        td.textContent = value;
        td.title = value;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    target.appendChild(tbody);
  };

  const renderPreviewMapping = (preview) => {
    const body = previewPanel?.querySelector("[data-xml-preview-mapping]");
    if (!body) return;
    body.replaceChildren();
    (preview.main_table?.columns || []).forEach((column) => {
      const tr = document.createElement("tr");
      const output = document.createElement("td");
      output.textContent = column.name;
      output.className = "xml-monospace-cell";
      const strategy = document.createElement("td");
      strategy.textContent = column.strategy || "direct";
      const sources = document.createElement("td");
      sources.textContent = (column.source_relative_paths || []).join(", ") || "—";
      sources.className = "xml-monospace-cell";
      const node = document.createElement("td");
      const sourcePath = (column.source_element_canonical_paths || [])[0];
      if (sourcePath && profileByPath.has(sourcePath)) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "button mini secondary";
        button.textContent = profileByPath.get(sourcePath).path;
        button.addEventListener("click", () => {
          inspectPath(sourcePath, {scroll: true});
          explorer.scrollIntoView({behavior: "smooth", block: "start"});
        });
        node.appendChild(button);
      } else {
        node.textContent = "—";
      }
      tr.append(output, strategy, sources, node);
      body.appendChild(tr);
    });
  };

  const renderChildPreviews = (preview) => {
    const host = previewPanel?.querySelector("[data-xml-child-previews]");
    if (!host) return;
    host.replaceChildren();
    (preview.child_tables || []).forEach((table) => {
      const section = document.createElement("section");
      section.className = "xml-child-preview";
      const heading = document.createElement("div");
      heading.className = "section-head";
      const title = document.createElement("div");
      const h3 = document.createElement("h3");
      h3.textContent = `Child table: ${table.name}`;
      const meta = document.createElement("p");
      meta.className = "tiny muted";
      meta.textContent = `${table.preview_row_count || 0} preview rows${table.truncated ? " · truncated" : ""}`;
      title.append(h3, meta);
      heading.appendChild(title);
      section.appendChild(heading);
      const wrap = document.createElement("div");
      wrap.className = "table-wrap xml-preview-table-wrap";
      const tableNode = document.createElement("table");
      tableNode.className = "xml-preview-table";
      renderPreviewTable(table, tableNode);
      wrap.appendChild(tableNode);
      section.appendChild(wrap);
      host.appendChild(section);
    });
  };

  const renderPreviewPanel = () => {
    if (!previewPanel) return;
    const hasSelection = Boolean(fieldCatalog && selectedFieldIds.size);
    previewPanel.hidden = !hasSelection;
    if (!hasSelection) return;
    const ready = Boolean(collectionPlan?.summary?.ready_for_preview);
    const buildButton = previewPanel.querySelector("[data-xml-build-preview]");
    if (buildButton) buildButton.disabled = !ready || !previewUrl;
    const state = previewPanel.querySelector("[data-xml-preview-state]");
    const confirm = previewPanel.querySelector("[data-xml-confirm-preview]");
    const resultHost = previewPanel.querySelector("[data-xml-preview-result]");
    const statusNode = previewPanel.querySelector("[data-xml-preview-status]");

    if (!previewResult) {
      if (resultHost) resultHost.hidden = true;
      if (confirm) confirm.disabled = true;
      if (state) {
        state.textContent = confirmedPreviewSignature ? "confirmed · rebuild to inspect" : (ready ? "ready to preview" : "rules incomplete");
        state.classList.toggle("ready", ready || Boolean(confirmedPreviewSignature));
      }
      if (statusNode) statusNode.textContent = ready
        ? "Rules are valid. Build a bounded table preview to inspect values and output columns."
        : "Resolve the repeated-branch rules before building a table preview.";
      renderMaterializePanel();
      return;
    }

    if (resultHost) resultHost.hidden = false;
    const confirmed = confirmedPreviewSignature === previewResult.preview_signature;
    if (state) {
      state.textContent = confirmed ? "confirmed" : "preview built";
      state.classList.toggle("ready", true);
    }
    if (confirm) {
      confirm.disabled = confirmed;
      confirm.textContent = confirmed ? "Extraction configuration confirmed" : "Confirm extraction configuration";
    }
    if (statusNode) statusNode.textContent = confirmed
      ? "This exact source fingerprint, record root, field selection, and rule configuration is confirmed for dataset materialization."
      : "Preview built. Inspect the table and field mapping before confirming this extraction configuration.";

    const main = previewResult.main_table || {columns: [], rows: []};
    previewPanel.querySelector("[data-xml-preview-row-count]").textContent = String(main.preview_row_count || 0);
    previewPanel.querySelector("[data-xml-preview-column-count]").textContent = String((main.columns || []).length);
    previewPanel.querySelector("[data-xml-preview-truncation]").textContent = main.truncated
      ? `showing a bounded sample of ${main.estimated_total_rows ?? "more"} record roots`
      : "complete record-root sample";
    renderPreviewTable(main, previewPanel.querySelector("[data-xml-preview-table]"));
    renderPreviewMapping(previewResult);
    renderChildPreviews(previewResult);

    const warnings = previewPanel.querySelector("[data-xml-preview-warnings]");
    warnings.replaceChildren();
    const messages = previewResult.warnings || [];
    warnings.hidden = !messages.length;
    messages.forEach((message) => {
      const row = document.createElement("div");
      row.className = "tiny";
      row.textContent = message;
      warnings.appendChild(row);
    });
    previewPanel.querySelector("[data-xml-preview-json]").textContent = JSON.stringify(previewResult, null, 2);
    renderMaterializePanel();
  };

  const invalidatePreview = ({persist = true} = {}) => {
    previewResult = null;
    confirmedPreviewSignature = null;
    materializationResult = null;
    renderPreviewPanel();
    if (persist) saveStoredSelection();
  };

  const buildPreview = async () => {
    if (!previewUrl || !fieldCatalog || !collectionPlan?.summary?.ready_for_preview || !previewPanel) return;
    const state = previewPanel.querySelector("[data-xml-preview-state]");
    const statusNode = previewPanel.querySelector("[data-xml-preview-status]");
    const maxRows = Number.parseInt(previewPanel.querySelector("[data-xml-preview-max-rows]")?.value || "25", 10);
    if (state) state.textContent = "building…";
    if (statusNode) statusNode.textContent = "Reading a bounded XML sample and applying the current extraction rules…";
    try {
      const response = await fetch(previewUrl, {
        method: "POST",
        headers: {"Accept": "application/json", "Content-Type": "application/json"},
        body: JSON.stringify({
          record_root_canonical_path: fieldCatalog.record_root.canonical_path,
          selected_field_ids: [...selectedFieldIds],
          rules: currentRulesPayload(),
          max_rows: Number.isFinite(maxRows) ? maxRows : 25,
        }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
      previewResult = result;
      confirmedPreviewSignature = null;
      renderPreviewPanel();
      saveStoredSelection();
      previewPanel.scrollIntoView({behavior: "smooth", block: "start"});
    } catch (error) {
      previewResult = null;
      confirmedPreviewSignature = null;
      if (state) state.textContent = "preview failed";
      if (statusNode) statusNode.textContent = `Unable to build table preview: ${error.message}`;
      const resultHost = previewPanel.querySelector("[data-xml-preview-result]");
      if (resultHost) resultHost.hidden = true;
    }
  };

  const renderMaterializePanel = () => {
    if (!previewPanel) return;
    const panel = previewPanel.querySelector("[data-xml-materialize-panel]");
    if (!panel) return;
    const canMaterialize = Boolean(
      materializeUrl &&
      fieldCatalog &&
      selectedFieldIds.size &&
      collectionPlan?.summary?.ready_for_preview &&
      confirmedPreviewSignature
    );
    panel.hidden = !Boolean(confirmedPreviewSignature);
    const button = panel.querySelector("[data-xml-materialize]");
    if (button) button.disabled = !canMaterialize;
    const status = panel.querySelector("[data-xml-materialize-status]");
    const result = panel.querySelector("[data-xml-materialize-result]");
    if (!materializationResult) {
      if (result) result.hidden = true;
      if (status) status.textContent = canMaterialize
        ? "Confirmed extraction is ready to write as a normal ML_Lab dataset."
        : "Confirm the current extraction preview before creating a dataset.";
      return;
    }
    if (result) result.hidden = false;
    if (status) status.textContent = "Dataset created. The main CSV/TSV is ready for ordinary Data Lab workflows.";
    panel.querySelector("[data-xml-materialize-path]").textContent = materializationResult.dataset?.path || "";
    panel.querySelector("[data-xml-materialize-rows]").textContent = String(materializationResult.dataset?.rows ?? "—");
    panel.querySelector("[data-xml-materialize-columns]").textContent = String(materializationResult.dataset?.columns ?? "—");
    panel.querySelector("[data-xml-materialize-structure]").textContent = materializationResult.structure_artifact_path || "";
    const transformLink = panel.querySelector("[data-xml-materialize-transform]");
    if (transformLink && materializationResult.transform_url) transformLink.href = materializationResult.transform_url;
    panel.querySelector("[data-xml-materialize-json]").textContent = JSON.stringify(materializationResult, null, 2);
  };

  const materializeDataset = async () => {
    if (!materializeUrl || !fieldCatalog || !confirmedPreviewSignature || !previewPanel) return;
    const panel = previewPanel.querySelector("[data-xml-materialize-panel]");
    const status = panel?.querySelector("[data-xml-materialize-status]");
    const button = panel?.querySelector("[data-xml-materialize]");
    const output = panel?.querySelector("[data-xml-materialize-output]")?.value?.trim() || "";
    const overwrite = Boolean(panel?.querySelector("[data-xml-materialize-overwrite]")?.checked);
    if (button) button.disabled = true;
    if (status) status.textContent = "Reading the full XML extraction and writing the derived dataset…";
    try {
      const response = await fetch(materializeUrl, {
        method: "POST",
        headers: {"Accept": "application/json", "Content-Type": "application/json"},
        body: JSON.stringify({
          record_root_canonical_path: fieldCatalog.record_root.canonical_path,
          selected_field_ids: [...selectedFieldIds],
          rules: currentRulesPayload(),
          confirmed_preview_signature: confirmedPreviewSignature,
          output,
          overwrite,
        }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
      materializationResult = result;
      renderMaterializePanel();
      loadExtractionHistory();
    } catch (error) {
      materializationResult = null;
      if (status) status.textContent = `Unable to create dataset: ${error.message}`;
      if (button) button.disabled = false;
      const result = panel?.querySelector("[data-xml-materialize-result]");
      if (result) result.hidden = true;
    }
  };

  previewPanel?.querySelector("[data-xml-build-preview]")?.addEventListener("click", buildPreview);
  previewPanel?.querySelector("[data-xml-confirm-preview]")?.addEventListener("click", () => {
    if (!previewResult?.preview_signature) return;
    confirmedPreviewSignature = previewResult.preview_signature;
    materializationResult = null;
    renderPreviewPanel();
    renderMaterializePanel();
    saveStoredSelection();
  });
  previewPanel?.querySelector("[data-xml-materialize]")?.addEventListener("click", materializeDataset);

  const renderRulePanel = () => {
    if (!rulePanel) return;
    if (!fieldCatalog || !selectedFieldIds.size || !collectionPlan) {
      rulePanel.hidden = true;
      renderPreviewPanel();
      return;
    }
    rulePanel.hidden = false;
    const summary = collectionPlan.summary || {};
    rulePanel.querySelector("[data-xml-rule-branch-count]").textContent = String(summary.repeated_branch_count || 0);
    rulePanel.querySelector("[data-xml-rule-unresolved-count]").textContent = String(summary.unresolved_branch_count || 0);
    rulePanel.querySelector("[data-xml-rule-invalid-count]").textContent = String(summary.invalid_rule_count || 0);
    const ready = Boolean(summary.ready_for_preview);
    const readyBadge = rulePanel.querySelector("[data-xml-rule-ready]");
    readyBadge.textContent = ready ? "rules ready" : "rules incomplete";
    readyBadge.classList.toggle("ready", ready);
    rulePanel.querySelector("[data-xml-rule-status]").textContent = summary.repeated_branch_count
      ? "Choose how each selected one-to-many branch will map into the future tabular preview."
      : "No repeated branches are touched by the current field selection; no one-to-many rule is required.";

    const list = rulePanel.querySelector("[data-xml-rule-list]");
    list.replaceChildren();
    const strategyCatalog = collectionPlan.strategy_catalog || [];
    (collectionPlan.repeated_branches || []).forEach((branch) => {
      const card = document.createElement("section");
      card.className = "xml-rule-card";
      const head = document.createElement("div");
      head.className = "xml-rule-card-head";
      const title = document.createElement("div");
      const h3 = document.createElement("h3");
      h3.textContent = branch.relative_path;
      const meta = document.createElement("div");
      meta.className = "tiny muted";
      meta.textContent = `${branch.occurrence_count} occurrences · ${branch.selected_field_ids.length} selected descendant fields · max ${branch.cardinality?.max_per_parent ?? "—"} / parent`;
      title.append(h3, meta);
      head.appendChild(title);

      const rule = normalizedRuleFromPlan(branch);
      const select = document.createElement("select");
      select.className = "xml-rule-strategy";
      const blank = document.createElement("option");
      blank.value = "";
      blank.textContent = "Choose strategy";
      select.appendChild(blank);
      strategyCatalog.forEach((strategy) => {
        const option = document.createElement("option");
        option.value = strategy.id;
        option.textContent = strategy.label;
        option.title = strategy.description;
        option.selected = rule.strategy === strategy.id;
        select.appendChild(option);
      });
      select.addEventListener("change", () => {
        const strategy = select.value || null;
        const next = {branch_canonical_path: branch.canonical_path, strategy, options: {}};
        if (strategy === "join") next.options.join_delimiter = ", ";
        collectionRules.set(branch.canonical_path, next);
        refreshCollectionPlan();
      });
      head.appendChild(select);
      card.appendChild(head);

      const selected = document.createElement("div");
      selected.className = "xml-rule-fields tiny";
      selected.textContent = `Fields: ${(branch.selected_fields || []).map((field) => field.relative_path).join(", ") || "none"}`;
      card.appendChild(selected);

      const optionWrap = document.createElement("div");
      optionWrap.className = "xml-rule-options";
      renderRuleOptions(optionWrap, branch, rule);
      card.appendChild(optionWrap);

      if (branch.rule?.errors?.length) {
        const errors = document.createElement("ul");
        errors.className = "xml-rule-errors";
        branch.rule.errors.forEach((message) => {
          const li = document.createElement("li");
          li.textContent = message;
          errors.appendChild(li);
        });
        card.appendChild(errors);
      }
      list.appendChild(card);
    });

    const warnings = rulePanel.querySelector("[data-xml-rule-warnings]");
    warnings.replaceChildren();
    const planWarnings = collectionPlan.warnings || [];
    warnings.hidden = !planWarnings.length;
    planWarnings.forEach((message) => {
      const item = document.createElement("div");
      item.className = "tiny";
      item.textContent = message;
      warnings.appendChild(item);
    });
    rulePanel.querySelector("[data-xml-rule-json]").textContent = JSON.stringify(collectionPlan, null, 2);
    renderPreviewPanel();
  };

  const postCollectionPlan = async (rules) => {
    const response = await fetch(collectionPlanUrl, {
      method: "POST",
      headers: {"Accept": "application/json", "Content-Type": "application/json"},
      body: JSON.stringify({
        record_root_canonical_path: fieldCatalog.record_root.canonical_path,
        selected_field_ids: [...selectedFieldIds],
        rules,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    return result;
  };

  const refreshCollectionPlan = async () => {
    invalidatePreview({persist: false});
    if (!collectionPlanUrl || !fieldCatalog || !rulePanel) {
      saveStoredSelection();
      return;
    }
    if (!selectedFieldIds.size) {
      collectionPlan = null;
      collectionRules.clear();
      renderRulePanel();
      saveStoredSelection();
      return;
    }
    const serial = ++planRequestSerial;
    rulePanel.hidden = false;
    rulePanel.querySelector("[data-xml-rule-status]").textContent = "Analyzing selected repeated branches…";
    try {
      const discovered = await postCollectionPlan([]);
      if (serial !== planRequestSerial) return;
      const validBranchPaths = new Set((discovered.repeated_branches || []).map((branch) => branch.canonical_path));
      collectionRules = new Map([...collectionRules].filter(([path]) => validBranchPaths.has(path)));
      const configured = currentRulesPayload();
      const result = configured.length ? await postCollectionPlan(configured) : discovered;
      if (serial !== planRequestSerial) return;
      collectionPlan = result;
      (collectionPlan.repeated_branches || []).forEach((branch) => {
        if (branch.rule?.strategy) collectionRules.set(branch.canonical_path, normalizedRuleFromPlan(branch));
      });
      renderRulePanel();
      renderSelectionSummary();
      saveStoredSelection();
    } catch (error) {
      if (serial !== planRequestSerial) return;
      collectionPlan = null;
      rulePanel.hidden = false;
      rulePanel.querySelector("[data-xml-rule-status]").textContent = `Unable to validate repeated-branch rules: ${error.message}`;
      rulePanel.querySelector("[data-xml-rule-list]").replaceChildren();
      rulePanel.querySelector("[data-xml-rule-json]").textContent = "";
      renderPreviewPanel();
    }
  };

  const chooseRecordRoot = async (canonicalPath, {scroll = true, restoreFieldIds = null, restoreRules = null} = {}) => {
    if (!recordFieldsUrl || !profileByPath.has(canonicalPath) || !fieldSelector) return;
    const statusNode = fieldSelector.querySelector("[data-xml-field-status]");
    fieldSelector.hidden = false;
    statusNode.textContent = "Loading selectable fields…";
    try {
      const response = await fetch(`${recordFieldsUrl}?root=${encodeURIComponent(canonicalPath)}`, {headers: {"Accept": "application/json"}});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
      const rootChanged = !fieldCatalog || fieldCatalog.record_root.canonical_path !== canonicalPath;
      fieldCatalog = result;
      const validIds = new Set((fieldCatalog.fields || []).map((field) => field.field_id));
      if (restoreFieldIds !== null) selectedFieldIds = new Set(restoreFieldIds.filter((fieldId) => validIds.has(fieldId)));
      else if (rootChanged) selectedFieldIds = new Set();
      else selectedFieldIds = new Set([...selectedFieldIds].filter((fieldId) => validIds.has(fieldId)));
      if (restoreRules !== null) {
        collectionRules = new Map((restoreRules || []).filter((rule) => rule?.branch_canonical_path).map((rule) => [rule.branch_canonical_path, rule]));
      } else if (rootChanged) {
        collectionRules.clear();
      }
      collectionPlan = null;
      inspectPath(canonicalPath);
      renderFieldRows();
      renderSelectionSummary();
      await refreshCollectionPlan();
      if (scroll) fieldSelector.scrollIntoView({behavior: "smooth", block: "start"});
    } catch (error) {
      fieldCatalog = null;
      selectedFieldIds = new Set();
      collectionPlan = null;
      collectionRules.clear();
      statusNode.textContent = `Unable to load fields: ${error.message}`;
      renderSelectionSummary();
      renderRulePanel();
    }
  };

  const historyLabel = (entry) => {
    const dataset = entry.dataset || {};
    const name = dataset.path ? dataset.path.split(/[\\/]/).pop() : "dataset";
    const created = entry.created_at ? new Date(entry.created_at).toLocaleString() : "unknown time";
    return `${name} · ${created}`;
  };

  const populateHistoryCompareSelectors = (entries) => {
    if (!historyPanel) return;
    const left = historyPanel.querySelector("[data-xml-history-left]");
    const right = historyPanel.querySelector("[data-xml-history-right]");
    [left, right].forEach((select) => {
      if (!select) return;
      select.replaceChildren();
      entries.forEach((entry) => {
        const option = document.createElement("option");
        option.value = entry.event_id || "";
        option.textContent = historyLabel(entry);
        select.appendChild(option);
      });
    });
    if (right && entries.length > 1) right.selectedIndex = 1;
    const compareHost = historyPanel.querySelector("[data-xml-history-compare]");
    if (compareHost) compareHost.hidden = entries.length < 2;
  };

  const loadHistoryRecipe = async (eventId) => {
    if (!extractionReentryUrl || !eventId) return;
    const statusNode = historyPanel?.querySelector("[data-xml-history-status]");
    if (statusNode) statusNode.textContent = "Loading recorded extraction recipe…";
    try {
      const response = await fetch(extractionReentryUrl, {
        method: "POST",
        headers: {"Accept": "application/json", "Content-Type": "application/json"},
        body: JSON.stringify({event_id: eventId}),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
      const selection = result.selection || {};
      const root = selection.record_root_canonical_path;
      if (!root || !profileByPath.has(root)) throw new Error("Recorded root is not present in the current XML structure.");
      await chooseRecordRoot(root, {
        scroll: true,
        restoreFieldIds: selection.selected_field_ids || [],
        restoreRules: selection.collection_rules || [],
      });
      confirmedPreviewSignature = selection.confirmed_preview_signature || null;
      materializationResult = null;
      renderPreviewPanel();
      renderMaterializePanel();
      saveStoredSelection();
      if (statusNode) statusNode.textContent = "Recorded extraction recipe loaded. Edit it or choose a different record root to create another sibling dataset.";
    } catch (error) {
      if (statusNode) statusNode.textContent = `Unable to reopen extraction: ${error.message}`;
    }
  };

  const loadExtractionHistory = async () => {
    if (!historyPanel || !extractionHistoryUrl) return;
    const statusNode = historyPanel.querySelector("[data-xml-history-status]");
    const wrap = historyPanel.querySelector("[data-xml-history-table-wrap]");
    const rows = historyPanel.querySelector("[data-xml-history-rows]");
    statusNode.textContent = "Loading recorded extractions…";
    try {
      const response = await fetch(extractionHistoryUrl, {headers: {"Accept": "application/json"}});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
      const entries = Array.isArray(result.entries) ? result.entries : [];
      rows.replaceChildren();
      entries.forEach((entry) => {
        const tr = document.createElement("tr");
        const dataset = entry.dataset || {};
        const values = [
          entry.created_at ? new Date(entry.created_at).toLocaleString() : "—",
          dataset.path || "—",
          entry.record_root_canonical_path || "—",
          String((entry.selected_field_ids || []).length),
          `${dataset.rows ?? "—"} × ${dataset.columns ?? "—"}`,
          entry.status || "unknown",
        ];
        values.forEach((value, index) => {
          const td = document.createElement("td");
          if (index === 1 || index === 2) {
            const code = document.createElement("code");
            code.textContent = value;
            td.appendChild(code);
          } else td.textContent = value;
          tr.appendChild(td);
        });
        const action = document.createElement("td");
        const button = document.createElement("button");
        button.type = "button";
        button.className = "button mini secondary";
        button.textContent = "Load recipe";
        button.disabled = !entry.event_id;
        button.addEventListener("click", () => loadHistoryRecipe(entry.event_id));
        action.appendChild(button);
        tr.appendChild(action);
        rows.appendChild(tr);
      });
      wrap.hidden = entries.length === 0;
      populateHistoryCompareSelectors(entries);
      statusNode.textContent = entries.length
        ? `${entries.length} recorded extraction${entries.length === 1 ? "" : "s"} for this XML source.`
        : "No recorded XML extractions yet. Materialize a confirmed extraction to create the first one.";
      const warnings = result.warnings || [];
      if (warnings.length) statusNode.textContent += ` ${warnings.length} history warning${warnings.length === 1 ? "" : "s"}.`;
    } catch (error) {
      wrap.hidden = true;
      statusNode.textContent = `Unable to load extraction history: ${error.message}`;
    }
  };

  const compareHistoryRecipes = async () => {
    if (!historyPanel || !extractionCompareUrl) return;
    const left = historyPanel.querySelector("[data-xml-history-left]")?.value || "";
    const right = historyPanel.querySelector("[data-xml-history-right]")?.value || "";
    const statusNode = historyPanel.querySelector("[data-xml-history-compare-status]");
    const resultHost = historyPanel.querySelector("[data-xml-history-compare-result]");
    if (!left || !right) return;
    statusNode.textContent = "Comparing recorded recipes…";
    try {
      const response = await fetch(extractionCompareUrl, {
        method: "POST",
        headers: {"Accept": "application/json", "Content-Type": "application/json"},
        body: JSON.stringify({left_event_id: left, right_event_id: right}),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
      historyPanel.querySelector("[data-xml-history-compare-json]").textContent = JSON.stringify(result, null, 2);
      resultHost.hidden = false;
      statusNode.textContent = result.same_recipe
        ? "The selected extraction recipes are equivalent."
        : `Recipes differ: ${result.fields?.added?.length || 0} fields added, ${result.fields?.removed?.length || 0} removed, ${result.rules?.changed_count || 0} repeated-branch rule changes.`;
    } catch (error) {
      resultHost.hidden = true;
      statusNode.textContent = `Unable to compare extraction recipes: ${error.message}`;
    }
  };

  historyPanel?.querySelector("[data-xml-history-refresh]")?.addEventListener("click", loadExtractionHistory);
  historyPanel?.querySelector("[data-xml-history-compare-button]")?.addEventListener("click", compareHistoryRecipes);

  inspector.querySelector("[data-xml-use-record-root]")?.addEventListener("click", () => {
    if (selectedPath) chooseRecordRoot(selectedPath);
  });

  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-xml-choose-record-root]");
    if (!target) return;
    chooseRecordRoot(target.dataset.xmlChooseRecordRoot);
  });

  fieldSelector?.querySelector("[data-xml-field-search]")?.addEventListener("input", renderFieldRows);
  fieldSelector?.querySelector("[data-xml-clear-fields]")?.addEventListener("click", () => {
    selectedFieldIds.clear();
    renderFieldRows();
    renderSelectionSummary();
    saveStoredSelection();
    refreshCollectionPlan();
  });
  fieldSelector?.querySelector("[data-xml-select-all-fields]")?.addEventListener("click", () => {
    if (!fieldCatalog) return;
    selectedFieldIds = new Set((fieldCatalog.fields || []).map((field) => field.field_id));
    renderFieldRows();
    renderSelectionSummary();
    saveStoredSelection();
    refreshCollectionPlan();
  });
  fieldSelector?.querySelector("[data-xml-select-identifiers]")?.addEventListener("click", () => {
    if (!fieldCatalog) return;
    selectedFieldIds = new Set((fieldCatalog.fields || []).filter((field) => field.likely_identifier).map((field) => field.field_id));
    renderFieldRows();
    renderSelectionSummary();
    saveStoredSelection();
    refreshCollectionPlan();
  });
  fieldSelector?.querySelector("[data-xml-clear-record-root]")?.addEventListener("click", () => {
    fieldCatalog = null;
    selectedFieldIds.clear();
    collectionPlan = null;
    collectionRules.clear();
    fieldSelector.hidden = true;
    if (rulePanel) rulePanel.hidden = true;
    invalidatePreview({persist: false});
    nodeElements.forEach(({row}) => row.classList.remove("is-record-root"));
    saveStoredSelection();
  });

  const firstRoot = (childrenByParent.get(null) || [])[0];
  if (firstRoot) inspectPath(firstRoot.canonical_path);
  loadExtractionHistory();
  const storedSelection = loadStoredSelection();
  if (storedSelection?.record_root_canonical_path && profileByPath.has(storedSelection.record_root_canonical_path)) {
    chooseRecordRoot(storedSelection.record_root_canonical_path, {
      scroll: false,
      restoreFieldIds: storedSelection.selected_field_ids || [],
      restoreRules: storedSelection.collection_rules || [],
    }).then(() => {
      confirmedPreviewSignature = storedSelection.confirmed_preview_signature || null;
      renderPreviewPanel();
      renderMaterializePanel();
      saveStoredSelection();
    });
  }
})();
