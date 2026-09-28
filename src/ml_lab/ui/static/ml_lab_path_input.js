(() => {
  const STORAGE_KEY = "ml_lab.path_input.recent.v1";

  const readRecent = () => {
    try {
      const parsed = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(parsed) ? parsed.filter((item) => typeof item === "string") : [];
    } catch (_) {
      return [];
    }
  };

  const writeRecent = (paths) => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(paths.slice(0, 20)));
    } catch (_) {
      // Browser storage is optional; path completion still works without it.
    }
  };

  const remember = (value) => {
    const path = String(value || "").trim();
    if (!path) return;
    const recent = readRecent().filter((item) => item !== path);
    recent.unshift(path);
    writeRecent(recent);
  };

  const initPathInput = (input) => {
    const shell = input.closest("[data-path-input]");
    const menu = shell && shell.querySelector("[data-path-suggestion-menu]");
    if (!shell || !menu) return;

    const endpoint = input.dataset.pathSuggestionsUrl;
    const kind = input.dataset.pathKind || "either";
    const extensions = input.dataset.pathExtensions || "";
    const multiline = input.dataset.pathMultiline === "true";
    let activeIndex = -1;
    let suggestions = [];
    let requestSerial = 0;

    const close = () => {
      menu.hidden = true;
      menu.innerHTML = "";
      activeIndex = -1;
      suggestions = [];
      input.setAttribute("aria-expanded", "false");
    };

    const currentSegment = () => {
      if (!multiline) return { query: input.value.trim(), start: 0, end: input.value.length };
      const value = input.value;
      const cursor = typeof input.selectionStart === "number" ? input.selectionStart : value.length;
      const start = value.lastIndexOf("\n", Math.max(0, cursor - 1)) + 1;
      const nextBreak = value.indexOf("\n", cursor);
      const end = nextBreak === -1 ? value.length : nextBreak;
      return { query: value.slice(start, end).trim(), start, end };
    };

    const choose = (item) => {
      if (multiline) {
        const segment = currentSegment();
        input.setRangeText(item.path, segment.start, segment.end, "end");
      } else {
        input.value = item.path;
      }
      remember(item.path);
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
      close();
      input.focus();
    };

    const render = (items) => {
      suggestions = items;
      menu.innerHTML = "";
      activeIndex = -1;
      if (!items.length) {
        close();
        return;
      }
      items.forEach((item, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "path-suggestion-item";
        button.dataset.pathSuggestionIndex = String(index);
        button.innerHTML = `<span class="path-suggestion-kind">${item.kind === "directory" ? "DIR" : item.kind === "recent" ? "RECENT" : "FILE"}</span><span class="path-suggestion-value"></span>`;
        button.querySelector(".path-suggestion-value").textContent = item.path;
        button.addEventListener("mousedown", (event) => {
          event.preventDefault();
          choose(item);
        });
        menu.appendChild(button);
      });
      menu.hidden = false;
      input.setAttribute("aria-expanded", "true");
    };

    const setActive = (index) => {
      const buttons = [...menu.querySelectorAll("[data-path-suggestion-index]")];
      buttons.forEach((button) => button.classList.remove("is-active"));
      if (!buttons.length) {
        activeIndex = -1;
        return;
      }
      activeIndex = Math.max(0, Math.min(index, buttons.length - 1));
      buttons[activeIndex].classList.add("is-active");
      buttons[activeIndex].scrollIntoView({ block: "nearest" });
    };

    const recentMatches = (query) => {
      if (!query) return readRecent().slice(0, 8).map((path) => ({ path, kind: "recent" }));
      const folded = query.toLowerCase();
      return readRecent()
        .filter((path) => path.toLowerCase().includes(folded))
        .slice(0, 6)
        .map((path) => ({ path, kind: "recent" }));
    };

    const refresh = async () => {
      const query = currentSegment().query;
      const serial = ++requestSerial;
      const recent = recentMatches(query);
      if (!endpoint) {
        render(recent);
        return;
      }
      const url = new URL(endpoint, window.location.href);
      url.searchParams.set("q", query);
      url.searchParams.set("kind", kind);
      if (extensions) url.searchParams.set("extensions", extensions);
      url.searchParams.set("limit", "30");
      try {
        const response = await fetch(url.toString(), { headers: { Accept: "application/json" } });
        if (!response.ok || serial !== requestSerial) return;
        const payload = await response.json();
        if (serial !== requestSerial) return;
        const seen = new Set();
        const combined = [];
        [...recent, ...(payload.suggestions || [])].forEach((item) => {
          if (!item || !item.path || seen.has(item.path)) return;
          seen.add(item.path);
          combined.push(item);
        });
        render(combined);
      } catch (_) {
        if (serial === requestSerial) render(recent);
      }
    };

    let timer = null;
    input.setAttribute("role", "combobox");
    input.setAttribute("aria-autocomplete", "list");
    input.setAttribute("aria-expanded", "false");
    input.addEventListener("input", () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(refresh, 90);
    });
    input.addEventListener("focus", refresh);
    input.addEventListener("blur", () => window.setTimeout(close, 120));
    input.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown") {
        if (menu.hidden) refresh();
        else setActive(activeIndex + 1);
        event.preventDefault();
      } else if (event.key === "ArrowUp" && !menu.hidden) {
        setActive(activeIndex <= 0 ? suggestions.length - 1 : activeIndex - 1);
        event.preventDefault();
      } else if ((event.key === "Enter" || event.key === "Tab") && !menu.hidden && activeIndex >= 0) {
        choose(suggestions[activeIndex]);
        if (event.key === "Enter") event.preventDefault();
      } else if (event.key === "Escape") {
        close();
      }
    });
  };

  document.querySelectorAll("[data-path-autocomplete]").forEach(initPathInput);
  document.querySelectorAll("form").forEach((form) => {
    form.addEventListener("submit", () => {
      form.querySelectorAll("[data-path-autocomplete]").forEach((input) => {
        if (input.dataset.pathMultiline === "true") {
          input.value.split(/\r?\n/).map((value) => value.trim()).filter(Boolean).forEach(remember);
        } else {
          remember(input.value);
        }
      });
    });
  });
})();
