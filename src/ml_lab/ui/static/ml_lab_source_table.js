(() => {
  const root = document.getElementById("source-viewer");
  if (!root) return;
  const rowsUrl = root.dataset.rowsUrl;
  const search = document.getElementById("source-search");
  const pageSize = document.getElementById("source-page-size");
  const clear = document.getElementById("source-clear");
  const range = document.getElementById("source-range");
  const status = document.getElementById("source-status");
  const head = document.getElementById("source-head");
  const body = document.getElementById("source-body");
  const prev = document.getElementById("source-prev");
  const next = document.getElementById("source-next");
  const pageLabel = document.getElementById("source-page-label");
  const state = {page: 1, size: "50", search: "", filters: {}, sort: "", direction: "asc", columns: []};
  let timer = null;

  const escapeText = (value) => value == null ? "" : String(value);
  const schedule = () => { window.clearTimeout(timer); timer = window.setTimeout(load, 180); };

  function renderHead(columns) {
    if (state.columns.join("\0") === columns.join("\0") && head.children.length) return;
    state.columns = columns;
    head.innerHTML = "";
    const labels = document.createElement("tr");
    const filters = document.createElement("tr");
    filters.className = "column-filter-row";
    columns.forEach(column => {
      const th = document.createElement("th"); th.className = "sortable-head"; th.textContent = column;
      th.addEventListener("click", () => { if (state.sort === column) state.direction = state.direction === "asc" ? "desc" : "asc"; else { state.sort = column; state.direction = "asc"; } state.page = 1; load(); });
      labels.append(th);
      const filterTh = document.createElement("th"); const input = document.createElement("input"); input.className = "column-filter"; input.placeholder = "Filter"; input.value = state.filters[column] || "";
      input.addEventListener("input", () => { state.filters[column] = input.value; state.page = 1; schedule(); });
      filterTh.append(input); filters.append(filterTh);
    });
    head.append(labels, filters);
  }

  async function load() {
    status.textContent = "Loading rows…";
    const params = new URLSearchParams({page: String(state.page), page_size: state.size, search: state.search, filters: JSON.stringify(state.filters), direction: state.direction});
    if (state.sort) params.set("sort", state.sort);
    try {
      const response = await fetch(`${rowsUrl}?${params}`, {headers: {"Accept": "application/json"}});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not load source rows");
      state.page = payload.page; renderHead(payload.columns || []); body.innerHTML = "";
      (payload.rows || []).forEach(row => {
        const tr = document.createElement("tr");
        row.forEach(value => { const td = document.createElement("td"); td.textContent = escapeText(value); if (escapeText(value).trim() === "") td.classList.add("cell-missing"); tr.append(td); });
        body.append(tr);
      });
      range.textContent = payload.matched_row_count ? `${payload.first_row}–${payload.last_row} of ${payload.matched_row_count} matching · ${payload.valid_source_row_count} valid source rows` : `0 matching · ${payload.valid_source_row_count} valid source rows`;
      pageLabel.textContent = state.size === "all" ? "All matching rows" : `Page ${payload.page} of ${payload.page_count}`;
      prev.disabled = state.size === "all" || payload.page <= 1; next.disabled = state.size === "all" || payload.page >= payload.page_count;
      status.textContent = payload.skipped_malformed_row_count ? `${payload.skipped_malformed_row_count} malformed-width row(s) excluded from tabular rendering.` : "Ready";
    } catch (exc) { status.textContent = exc.message; }
  }

  search.addEventListener("input", () => { state.search = search.value; state.page = 1; schedule(); });
  pageSize.addEventListener("change", () => { state.size = pageSize.value; state.page = 1; load(); });
  clear.addEventListener("click", () => { state.search = ""; state.filters = {}; state.sort = ""; state.direction = "asc"; state.page = 1; search.value = ""; head.querySelectorAll("input").forEach(input => input.value = ""); load(); });
  prev.addEventListener("click", () => { if (state.page > 1) { state.page--; load(); } });
  next.addEventListener("click", () => { state.page++; load(); });
  load();
})();
