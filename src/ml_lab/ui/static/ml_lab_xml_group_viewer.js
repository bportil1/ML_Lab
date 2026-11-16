/* ML-UW-22 — a read-only, lazily rendered view of the C2 observed group union.
 * No XML source is fetched or reparsed here; the C2 union is the only input.
 * Export pure derivation helpers for local/headless regression tests.
 */
(() => {
  "use strict";

  const LIMIT = 80;
  const compactQName = (qname) => {
    const text = String(qname || "");
    const close = text.indexOf("}");
    return text.startsWith("{") && close >= 0 ? text.slice(close + 1) : text;
  };
  const decodedSegment = (path) => String(path || "").split("/").pop()
    .replace(/~1/g, "/").replace(/~0/g, "~");
  const namespaceFor = (qname) => {
    const end = String(qname || "").indexOf("}");
    return String(qname || "").startsWith("{") && end >= 0 ? qname.slice(1, end) : "";
  };
  const intersect = (values, members) => (Array.isArray(values) ? values : []).filter((item) => members.has(item));
  const total = (counts, present) => present.reduce((sum, name) => sum + Number((counts || {})[name] || 0), 0);

  function deriveGroupView(union, groupId) {
    if (!union || !Array.isArray(union.groups) || !Array.isArray(union.paths)) return null;
    const group = union.groups.find((item) => item.group_id === groupId);
    if (!group || !Array.isArray(group.members) || !Array.isArray(group.path_ids)) return null;
    const memberSet = new Set(group.members);
    const ids = new Set(group.path_ids);
    const candidateRoots = new Set(group.shared_record_root_candidates || []);
    const nodes = [];
    const add = (node) => { nodes.push(node); };
    for (const entry of union.paths) {
      if (!ids.has(entry.canonical_path)) continue;
      const present = intersect(entry.present_in, memberSet);
      const optional = intersect(entry.optional_in, memberSet);
      const repeated = intersect(entry.repeated_in, memberSet);
      const mixed = intersect(entry.mixed_content_in, memberSet);
      const qname = decodedSegment(entry.canonical_path);
      add({
        id: entry.canonical_path, parent: entry.parent_canonical_path, kind: "element",
        label: compactQName(qname), namespace: namespaceFor(qname),
        recordRootCandidate: candidateRoots.has(entry.canonical_path),
        present, missing: memberSet.size - present.length, optional, repeated, mixed,
        occurrences: total(entry.occurrences_by_source, present),
        textOccurrences: total(entry.text_occurrences_by_source, present),
      });
      for (const attribute of entry.attributes || []) {
        const attrPresent = intersect(attribute.present_in, memberSet);
        const attrName = attribute.attribute_qname;
        add({
          id: attribute.field_id, parent: entry.canonical_path, kind: "attribute",
          label: "@" + compactQName(attrName), namespace: namespaceFor(attrName),
          recordRootCandidate: false,
          present: attrPresent, missing: memberSet.size - attrPresent.length,
          optional: [], repeated: [], mixed: [],
          occurrences: total(attribute.occurrences_by_source, attrPresent), textOccurrences: null,
        });
      }
    }
    const nodeMap = new Map(nodes.map((node) => [node.id, node]));
    const children = new Map();
    for (const node of nodes) {
      // Broken or truncated union metadata is never silently assigned a made-up parent.
      const parent = node.parent && nodeMap.has(node.parent) ? node.parent : null;
      if (!children.has(parent)) children.set(parent, []);
      children.get(parent).push(node);
    }
    for (const siblings of children.values()) {
      siblings.sort((a, b) => a.kind.localeCompare(b.kind) || a.label.localeCompare(b.label) || a.id.localeCompare(b.id));
    }
    return { group, nodes, nodeMap, children, sourceCount: memberSet.size,
      optionalElementCount: nodes.filter((n) => n.kind === "element" && (n.missing || n.optional.length)).length };
  }

  // ML-UW-23: one union-wide view uses the same canonical path and tree contract
  // as the per-group viewer. No source XML is loaded into the browser.
  function deriveConsolidatedView(union) {
    if (!union || !Array.isArray(union.groups) || !Array.isArray(union.paths) ||
        !Array.isArray(union.source_paths)) return null;
    const sourceSet = new Set(union.source_paths);
    const groupBySource = new Map();
    for (const group of union.groups) {
      if (!Array.isArray(group.members)) return null;
      for (const member of group.members) {
        // Don't display misleading group coverage if union membership is corrupt.
        if (!sourceSet.has(member) || groupBySource.has(member)) return null;
        groupBySource.set(member, group.group_id);
      }
    }
    if (groupBySource.size !== sourceSet.size) return null;
    const candidateRoots = new Set(union.groups.flatMap((group) => group.shared_record_root_candidates || []));
    const nodes = [];
    for (const entry of union.paths) {
      const present = intersect(entry.present_in, sourceSet);
      const optional = intersect(entry.optional_in, sourceSet);
      const repeated = intersect(entry.repeated_in, sourceSet);
      const mixed = intersect(entry.mixed_content_in, sourceSet);
      const qname = decodedSegment(entry.canonical_path);
      nodes.push({
        id: entry.canonical_path, parent: entry.parent_canonical_path, kind: "element",
        label: compactQName(qname), namespace: namespaceFor(qname),
        recordRootCandidate: candidateRoots.has(entry.canonical_path),
        present, missing: sourceSet.size - present.length, optional, repeated, mixed,
        occurrences: total(entry.occurrences_by_source, present),
        textOccurrences: total(entry.text_occurrences_by_source, present),
      });
      for (const attr of entry.attributes || []) {
        const attrPresent = intersect(attr.present_in, sourceSet);
        nodes.push({
          id: attr.field_id, parent: entry.canonical_path, kind: "attribute",
          label: "@" + compactQName(attr.attribute_qname), namespace: namespaceFor(attr.attribute_qname),
          recordRootCandidate: false, present: attrPresent,
          missing: sourceSet.size - attrPresent.length,
          optional: [], repeated: [], mixed: [],
          occurrences: total(attr.occurrences_by_source, attrPresent), textOccurrences: null,
        });
      }
    }
    const nodeMap = new Map(nodes.map((node) => [node.id, node]));
    const children = new Map();
    for (const node of nodes) {
      const parent = node.parent && nodeMap.has(node.parent) ? node.parent : null;
      if (!children.has(parent)) children.set(parent, []);
      children.get(parent).push(node);
    }
    for (const siblings of children.values()) {
      siblings.sort((a, b) => a.kind.localeCompare(b.kind) || a.label.localeCompare(b.label) || a.id.localeCompare(b.id));
    }
    return {
      kind: "union", nodes, nodeMap, children, groups: union.groups, groupBySource,
      sourceCount: sourceSet.size, groupCount: union.groups.length,
      optionalElementCount: nodes.filter((node) => node.kind === "element" &&
        (node.missing > 0 || node.optional.length > 0)).length,
    };
  }

  // Compute group coverage only for the selected field, not for every field in
  // the union (which would create a large paths x groups matrix in the browser).
  function fieldGroupCoverage(view, node) {
    if (!view || view.kind !== "union" || !node) return [];
    const counts = new Map();
    const optional = new Map();
    const repeated = new Map();
    for (const member of node.present) {
      const id = view.groupBySource.get(member);
      if (id !== undefined) counts.set(id, (counts.get(id) || 0) + 1);
    }
    for (const member of node.optional) {
      const id = view.groupBySource.get(member);
      if (id !== undefined) optional.set(id, (optional.get(id) || 0) + 1);
    }
    for (const member of node.repeated) {
      const id = view.groupBySource.get(member);
      if (id !== undefined) repeated.set(id, (repeated.get(id) || 0) + 1);
    }
    return view.groups.map((group) => ({
      groupId: group.group_id, sourceCount: group.members.length,
      present: counts.get(group.group_id) || 0,
      missing: group.members.length - (counts.get(group.group_id) || 0),
      optional: optional.get(group.group_id) || 0,
      repeated: repeated.get(group.group_id) || 0,
    }));
  }

  // A root is resolved through canonical parent IDs, not string prefixes:
  // attributes have different field-ID syntax from element paths.
  function rootOf(node, view) {
    let current = node;
    const seen = new Set();
    while (current && current.parent && view.nodeMap.has(current.parent)) {
      if (seen.has(current.id)) return null;
      seen.add(current.id);
      current = view.nodeMap.get(current.parent);
    }
    return current && current.id;
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { deriveGroupView, deriveConsolidatedView, fieldGroupCoverage,
      decodedSegment, compactQName, namespaceFor, rootOf };
  }
  if (typeof document === "undefined") return;
  const json = document.querySelector("[data-xml-group-union]");
  if (!json) return;
  let union;
  try { union = JSON.parse(json.textContent || "{}"); }
  catch (_error) {
    for (const status of document.querySelectorAll("[data-xml-group-status], [data-xml-union-status]")) {
      status.textContent = "The observed structure could not be displayed; inspect the C2 JSON report.";
    }
    return;
  }
  const element = (tag, className, value) => {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (value !== undefined) result.textContent = String(value);
    return result;
  };
  const put = (target, value) => { if (target) target.textContent = value; };
  const textList = (items, limit = 24) => items.slice(0, limit).join(", ") +
    (items.length > limit ? `, … (+${items.length - limit})` : "");

  // Both viewers share tree rendering, batching, path search and inspection.
  function mountViewer(viewer, mode) {
    if (!viewer) return null;
    const prefix = mode === "union" ? "xml-union" : "xml-group";
    const get = (part) => viewer.querySelector(`[data-${prefix}-${part}]`);
    const selector = mode === "group" ? get("select") : null;
    const rootSelector = mode === "union" ? get("root") : null;
    const tree = get("tree");
    const inspector = get("inspector");
    const search = get("search");
    const matches = get("matches");
    const status = get("status");
    const summary = get("summary");
    const collapse = get("collapse");
    if (!tree || !inspector || !search || !matches || !status || !summary || !collapse) return null;
    let view = null;
    let selected = null;

    function inspect(node) {
      selected = node.id;
      for (const button of tree.querySelectorAll(`[data-${prefix}-node]`)) {
        button.setAttribute("aria-current", button.dataset[mode === "union" ? "xmlUnionNode" : "xmlGroupNode"] === node.id ? "true" : "false");
      }
      inspector.replaceChildren(element("h3", "", node.label));
      const details = element("dl", "xml-group-facts");
      const fact = (name, value) => {
        details.append(element("dt", "", name), element("dd", "", value));
      };
      fact("Kind", node.kind);
      fact("Canonical field path", node.id);
      fact("Namespace", node.namespace || "(none)");
      fact("Source coverage", `${node.present.length} / ${view.sourceCount}`);
      fact(mode === "union" ? "Absent across collection" : "Absent in group", String(node.missing));
      fact("Observed occurrences", String(node.occurrences));
      if (node.kind === "element") {
        fact("Record-root candidate", node.recordRootCandidate ? "Yes (review required)" : "No");
        fact("Text observations", String(node.textOccurrences));
        fact("Optional within source", `${node.optional.length} source(s)`);
        fact("Repeated", `${node.repeated.length} source(s)`);
        fact("Mixed content", `${node.mixed.length} source(s)`);
      }
      fact("Present in", textList(node.present) || "—");
      inspector.appendChild(details);
      if (mode === "union") {
        const coverage = fieldGroupCoverage(view, node);
        const observedGroups = coverage.filter((item) => item.present > 0).length;
        const description = element("p", "tiny muted",
          `Observed in ${observedGroups} of ${coverage.length} candidate group(s). ` +
          `Missing-source counts are structural observations, not null counts per record.`);
        inspector.appendChild(description);
        inspector.appendChild(element("h4", "", "Coverage by candidate group"));
        const list = element("ul", "xml-union-coverage");
        inspector.appendChild(list);
        const batch = 25;
        const appendPage = (offset) => {
          for (const item of coverage.slice(offset, offset + batch)) {
            const row = element("li", "");
            row.appendChild(element("strong", "", `${item.groupId}: `));
            row.appendChild(element("span", "",
              `${item.present}/${item.sourceCount} sources · ${item.missing} absent` +
              (node.kind === "element" ? ` · ${item.optional} optional · ${item.repeated} repeated` : "")));
            list.appendChild(row);
          }
          if (offset + batch < coverage.length) {
            const more = element("li", "");
            const button = element("button", "button mini secondary", `Show more groups (${coverage.length - offset - batch} remaining)`);
            button.type = "button";
            button.addEventListener("click", () => { more.remove(); appendPage(offset + batch); });
            more.appendChild(button);
            list.appendChild(more);
          }
        };
        appendPage(0);
      }
      inspector.appendChild(element("p", "tiny muted",
        "Read-only observed evidence. A shared recipe still requires explicit validation and approval."));
    }

    function renderSiblings(parent, siblings, offset = 0) {
      const end = Math.min(offset + LIMIT, siblings.length);
      for (const node of siblings.slice(offset, end)) {
        const li = element("li", "xml-group-item");
        li.setAttribute("role", "treeitem");
        const row = element("div", "xml-group-node-row");
        const descendants = view.children.get(node.id) || [];
        const toggle = element("button", "xml-group-toggle", descendants.length ? "▸" : "·");
        toggle.type = "button";
        toggle.disabled = !descendants.length;
        toggle.setAttribute("aria-label", `Toggle children of ${node.label}`);
        if (descendants.length) toggle.setAttribute("aria-expanded", "false");
        const button = element("button", "xml-group-select-node", node.label);
        button.type = "button";
        button.dataset[mode === "union" ? "xmlUnionNode" : "xmlGroupNode"] = node.id;
        button.setAttribute("aria-current", node.id === selected ? "true" : "false");
        button.addEventListener("click", () => inspect(node));
        row.append(toggle, button);
        if (node.recordRootCandidate) row.appendChild(element("span", "xml-group-tag", "record root?"));
        if (node.missing) row.appendChild(element("span", "xml-group-tag xml-group-optional", `${node.missing} absent`));
        if (node.optional.length) row.appendChild(element("span", "xml-group-tag", "optional"));
        if (node.repeated.length) row.appendChild(element("span", "xml-group-tag", "repeated"));
        if (mode === "union") {
          const seen = new Set(node.present.map((name) => view.groupBySource.get(name)));
          row.appendChild(element("span", "tiny muted", `${seen.size}/${view.groupCount} groups`));
        }
        row.appendChild(element("span", "tiny muted", `${node.present.length}/${view.sourceCount} sources`));
        li.appendChild(row);
        toggle.addEventListener("click", () => {
          const existing = li.querySelector(":scope > ul");
          if (existing) {
            existing.remove();
            toggle.textContent = "▸";
            toggle.setAttribute("aria-expanded", "false");
            return;
          }
          const nested = element("ul", "xml-group-branch");
          nested.setAttribute("role", "group");
          li.appendChild(nested);
          renderSiblings(nested, descendants);
          toggle.textContent = "▾";
          toggle.setAttribute("aria-expanded", "true");
        });
        parent.appendChild(li);
      }
      if (end < siblings.length) {
        const more = element("li", "xml-group-more");
        const button = element("button", "button mini secondary", `Show next ${Math.min(LIMIT, siblings.length - end)} of ${siblings.length}`);
        button.type = "button";
        button.addEventListener("click", () => { more.remove(); renderSiblings(parent, siblings, end); });
        more.appendChild(button);
        parent.appendChild(more);
      }
    }

    function scoped(node) {
      return !rootSelector || !rootSelector.value || rootOf(node, view) === rootSelector.value;
    }
    function findNodes() {
      matches.replaceChildren();
      if (!view) return;
      const query = search.value.trim().toLocaleLowerCase();
      if (!query) return;
      const found = view.nodes.filter((node) => scoped(node) &&
        `${node.label} ${node.id}`.toLocaleLowerCase().includes(query));
      const limit = 40;
      put(status, `${found.length} matching paths; showing first ${Math.min(found.length, limit)}.`);
      for (const node of found.slice(0, limit)) {
        const button = element("button", "button mini secondary", node.id);
        button.type = "button";
        button.addEventListener("click", () => {
          inspect(node);
          inspector.scrollIntoView({ behavior: "smooth", block: "nearest" });
        });
        matches.appendChild(button);
      }
    }
    function paint() {
      tree.replaceChildren();
      inspector.replaceChildren(element("p", "muted", "Select a path to inspect observed structure and coverage."));
      selected = null;
      matches.replaceChildren();
      if (!view) { put(status, "Structure is unavailable in the observed C2 report."); return; }
      const elementCount = view.nodes.filter((node) => node.kind === "element").length;
      put(summary, `${view.sourceCount} verified source(s) · ${elementCount} element paths · ` +
        `${view.nodes.length - elementCount} attribute paths · ${view.optionalElementCount} optional element paths` +
        (mode === "union" ? ` · ${view.groupCount} candidate groups` : ""));
      const allRoots = view.children.get(null) || [];
      const roots = rootSelector && rootSelector.value ? allRoots.filter((node) => node.id === rootSelector.value) : allRoots;
      const list = element("ul", "xml-group-branch xml-group-root");
      list.setAttribute("role", "tree");
      list.setAttribute("aria-label", mode === "union" ? "Consolidated XML collection hierarchy" : `Structure of ${selector.value}`);
      tree.appendChild(list);
      renderSiblings(list, roots);
      put(status, mode === "union" ?
        "Displaying all candidate groups in one observed union. Expand paths and inspect group coverage." :
        `Displaying ${selector.value}. Expand branches to inspect deeper paths. This is read-only structural evidence.`);
      if (search.value.trim()) findNodes();
    }
    function load(groupId) {
      view = mode === "union" ? deriveConsolidatedView(union) : deriveGroupView(union, groupId);
      if (rootSelector) {
        rootSelector.replaceChildren(element("option", "", "All observed roots"));
        rootSelector.firstChild.value = "";
        for (const root of (view?.children.get(null) || [])) {
          const option = element("option", "", `${root.label}${root.namespace ? ` (${root.namespace})` : ""} — ${root.present.length} source(s)`);
          option.value = root.id;
          rootSelector.appendChild(option);
        }
      }
      paint();
    }
    if (selector) selector.addEventListener("change", () => load(selector.value));
    if (rootSelector) rootSelector.addEventListener("change", paint);
    search.addEventListener("input", findNodes);
    collapse.addEventListener("click", paint);
    load(selector?.value);
    return { load };
  }

  const groupViewer = mountViewer(document.querySelector("[data-xml-group-viewer]"), "group");
  mountViewer(document.querySelector("[data-xml-union-viewer]"), "union");
  if (groupViewer) {
    for (const button of document.querySelectorAll("[data-xml-open-group]")) {
      button.addEventListener("click", () => {
        const viewer = document.querySelector("[data-xml-group-viewer]");
        const selector = viewer.querySelector("[data-xml-group-select]");
        selector.value = button.dataset.xmlOpenGroup;
        groupViewer.load(selector.value);
        viewer.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }
  }
})();
