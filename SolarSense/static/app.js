// Solar Sense — shared front-end behavior

document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("navToggle");
  const nav = document.getElementById("mainNav");
  if (toggle && nav) {
    toggle.addEventListener("click", () => {
      const open = nav.classList.toggle("open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  // generic tab wiring: any .tabs with .tab-btn[data-tab] controlling .tab-panel[data-panel]
  document.querySelectorAll(".tabs").forEach((tabGroup) => {
    const container = tabGroup.closest("[data-tab-scope]") || document;
    tabGroup.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        tabGroup.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const target = btn.dataset.tab;
        container.querySelectorAll(".tab-panel").forEach((p) => {
          p.classList.toggle("active", p.dataset.panel === target);
        });
      });
    });
  });
});

const SolarSense = {
  fmt(value, digits = 1, unit = "") {
    if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
    return `${Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })}${unit}`;
  },

  statusBadge(status) {
    if (!status) return "";
    const cls = status === "NORMAL" ? "badge-normal" : "badge-abnormal";
    return `<span class="badge ${cls}">${status}</span>`;
  },

  async getJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    return res.json();
  },

  chartColors: {
    gold: "#f2b705",
    green: "#34c77b",
    red: "#ef5b4e",
    blue: "#4fa3e3",
    ink: "#b9c2d6",
    grid: "rgba(255,255,255,0.06)",
  },

  baseChartOptions(extra = {}) {
    return Object.assign({
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: this.chartColors.ink, font: { family: "Inter", size: 11 } } },
        tooltip: { backgroundColor: "#121a2b", borderColor: "#26314a", borderWidth: 1, titleColor: "#f3f5fa", bodyColor: "#b9c2d6" },
      },
      scales: {
        x: { ticks: { color: this.chartColors.ink, maxRotation: 0, autoSkip: true }, grid: { color: this.chartColors.grid } },
        y: { ticks: { color: this.chartColors.ink }, grid: { color: this.chartColors.grid } },
      },
    }, extra);
  },
};
