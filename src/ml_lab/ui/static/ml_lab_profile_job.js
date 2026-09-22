(() => {
  const root = document.getElementById("profile-job");
  if (!root?.dataset.statusUrl) return;
  const statusUrl = root.dataset.statusUrl;
  const badge = document.getElementById("job-badge");
  const dot = document.getElementById("run-dot");
  const stage = document.getElementById("job-stage");
  const path = document.getElementById("job-path");
  const progress = document.getElementById("job-progress");
  const elapsed = document.getElementById("job-elapsed");
  const error = document.getElementById("job-error");
  const warnings = document.getElementById("job-warnings");
  const results = document.getElementById("job-results");

  const update = async () => {
    try {
      const response = await fetch(statusUrl, {headers: {"Accept": "application/json"}});
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || "Could not read profile status");
      badge.textContent = job.status;
      badge.className = `badge ${job.status === "completed" ? "ready" : job.status === "failed" ? "dependency_missing" : "planned"}`;
      dot.className = `run-dot ${job.status}`;
      stage.textContent = job.stage || job.status;
      path.textContent = job.current_path || "";
      progress.textContent = `${job.current || 0} / ${job.total || 0}`;
      elapsed.textContent = job.elapsed_seconds == null ? "—" : `${Number(job.elapsed_seconds).toFixed(1)}s`;
      if (job.error) { error.textContent = job.error; error.classList.remove("hidden"); }
      warnings.innerHTML = (job.warnings || []).map(value => `<div class="tiny muted">• ${String(value).replaceAll("<", "&lt;")}</div>`).join("");
      if (job.profile_links?.length) results.innerHTML = job.profile_links.map(link => `<a class="button" href="${link.url}">Open profile · ${String(link.label).replaceAll("<", "&lt;")}</a>`).join("");
      if (!["completed", "failed"].includes(job.status)) window.setTimeout(update, 700);
    } catch (exc) {
      error.textContent = exc.message; error.classList.remove("hidden");
      window.setTimeout(update, 1500);
    }
  };
  update();
})();
