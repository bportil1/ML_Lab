(() => {
  const pageSizes = [25, 50, 100, 250, 500, "all"];
  const text = (cell) => (cell?.innerText || "").trim();
  const sortKey = (value) => {
    if (value === "" || value === "—") return {rank: 2, value: ""};
    const numeric = Number(value.replace(/[%,$]/g, ""));
    if (Number.isFinite(numeric)) return {rank: 0, value: numeric};
    return {rank: 1, value: value.toLocaleLowerCase()};
  };

  document.querySelectorAll("table.interactive-table").forEach((table) => {
    const tbody = table.tBodies[0];
    if (!tbody || !table.tHead?.rows.length) return;
    const rows = Array.from(tbody.rows);
    const header = table.tHead.rows[0];
    const state = {page: 1, pageSize: table.dataset.pageSize || "50", search: "", filters: Array.from(header.cells, () => ""), sort: null, direction: "asc"};

    const toolbar = document.createElement("div");
    toolbar.className = "interactive-table-toolbar";
    toolbar.innerHTML = `<label>Search <input type="search" class="table-search" placeholder="Search table"></label><label>Rows/page <select class="table-select">${pageSizes.map(v => `<option value="${v}" ${String(v) === String(state.pageSize) ? "selected" : ""}>${v === "all" ? "All" : v}</option>`).join("")}</select></label><button type="button" class="button mini secondary table-clear">Clear filters</button><span class="muted table-count"></span>`;
    table.closest(".table-wrap")?.before(toolbar);

    const filterRow = document.createElement("tr");
    filterRow.className = "column-filter-row";
    Array.from(header.cells).forEach((_, index) => {
      const th = document.createElement("th");
      if (index < header.cells.length - 1 || header.cells[index].innerText.trim().toLowerCase() !== "actions") {
        const input = document.createElement("input");
        input.className = "column-filter";
        input.placeholder = "Filter";
        input.addEventListener("input", () => { state.filters[index] = input.value.toLocaleLowerCase(); state.page = 1; render(); });
        th.append(input);
      }
      filterRow.append(th);
    });
    table.tHead.append(filterRow);

    Array.from(header.cells).forEach((cell, index) => {
      cell.classList.add("sortable-head");
      cell.addEventListener("click", () => {
        if (state.sort === index) state.direction = state.direction === "asc" ? "desc" : "asc";
        else { state.sort = index; state.direction = "asc"; }
        render();
      });
    });

    const pager = document.createElement("div");
    pager.className = "interactive-pagination";
    pager.innerHTML = `<button type="button" class="button mini secondary prev">Previous</button><span class="muted page-label"></span><button type="button" class="button mini secondary next">Next</button>`;
    table.closest(".table-wrap")?.after(pager);

    const search = toolbar.querySelector(".table-search");
    const size = toolbar.querySelector(".table-select");
    const count = toolbar.querySelector(".table-count");
    search.addEventListener("input", () => { state.search = search.value.toLocaleLowerCase(); state.page = 1; render(); });
    size.addEventListener("change", () => { state.pageSize = size.value; state.page = 1; render(); });
    toolbar.querySelector(".table-clear").addEventListener("click", () => {
      state.search = ""; state.filters.fill(""); state.sort = null; state.direction = "asc"; state.page = 1; search.value = "";
      filterRow.querySelectorAll("input").forEach(input => input.value = ""); render();
    });
    pager.querySelector(".prev").addEventListener("click", () => { if (state.page > 1) { state.page--; render(); } });
    pager.querySelector(".next").addEventListener("click", () => { state.page++; render(); });

    function render() {
      let filtered = rows.filter(row => {
        const cells = Array.from(row.cells).map(text);
        if (state.search && !cells.some(v => v.toLocaleLowerCase().includes(state.search))) return false;
        return state.filters.every((filter, i) => !filter || (cells[i] || "").toLocaleLowerCase().includes(filter));
      });
      if (state.sort !== null) {
        filtered = [...filtered].sort((a, b) => {
          const av = sortKey(text(a.cells[state.sort])); const bv = sortKey(text(b.cells[state.sort]));
          let result = av.rank - bv.rank;
          if (!result) result = av.value < bv.value ? -1 : av.value > bv.value ? 1 : 0;
          return state.direction === "desc" ? -result : result;
        });
      }
      const sizeValue = state.pageSize === "all" ? Math.max(filtered.length, 1) : Number(state.pageSize);
      const pageCount = Math.max(1, Math.ceil(filtered.length / sizeValue));
      state.page = Math.min(state.page, pageCount);
      const start = (state.page - 1) * sizeValue;
      const visible = new Set(filtered.slice(start, start + sizeValue));
      const ordered = state.sort === null ? rows : [...filtered, ...rows.filter(row => !filtered.includes(row))];
      ordered.forEach(row => tbody.append(row));
      rows.forEach(row => { row.hidden = !visible.has(row); });
      count.textContent = filtered.length ? `${start + 1}–${Math.min(start + sizeValue, filtered.length)} of ${filtered.length}` : "0 rows";
      pager.querySelector(".page-label").textContent = `Page ${state.page} of ${pageCount}`;
      pager.querySelector(".prev").disabled = state.page <= 1;
      pager.querySelector(".next").disabled = state.page >= pageCount;
    }
    render();
  });
})();
