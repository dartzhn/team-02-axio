const state = {
  flights: [],
  filtered: [],
  hasResults: false,
  filters: {
    search: "",
    risk: "All",
    origin: "All",
    carrier: "All",
    dest: "All",
    hour: 23,
  },
  selectedId: null,
  uploadedFile: null,
};

const els = {
  statusLine: document.querySelector("#statusLine"),
  totalFlights: document.querySelector("#totalFlights"),
  highRiskFlights: document.querySelector("#highRiskFlights"),
  mediumRiskFlights: document.querySelector("#mediumRiskFlights"),
  avgDelay: document.querySelector("#avgDelay"),
  topRoute: document.querySelector("#topRoute"),
  resultCount: document.querySelector("#resultCount"),
  flightRows: document.querySelector("#flightRows"),
  searchInput: document.querySelector("#searchInput"),
  originSelect: document.querySelector("#originSelect"),
  carrierSelect: document.querySelector("#carrierSelect"),
  destSelect: document.querySelector("#destSelect"),
  hourRange: document.querySelector("#hourRange"),
  hourValue: document.querySelector("#hourValue"),
  csvInput: document.querySelector("#csvInput"),
  runButton: document.querySelector("#runButton"),
  refreshButton: document.querySelector("#refreshButton"),
  downloadButton: document.querySelector("#downloadButton"),
  clearFilters: document.querySelector("#clearFilters"),
  busyOverlay: document.querySelector("#busyOverlay"),
  toast: document.querySelector("#toast"),
  emptyDetail: document.querySelector("#emptyDetail"),
  flightDetail: document.querySelector("#flightDetail"),
  detailRiskBadge: document.querySelector("#detailRiskBadge"),
  detailFlight: document.querySelector("#detailFlight"),
  detailRoute: document.querySelector("#detailRoute"),
  detailRisk: document.querySelector("#detailRisk"),
  detailTime: document.querySelector("#detailTime"),
  detailDelay: document.querySelector("#detailDelay"),
  detailBucket: document.querySelector("#detailBucket"),
  reasonList: document.querySelector("#reasonList"),
  actionList: document.querySelector("#actionList"),
  riskFactorList: document.querySelector("#riskFactorList"),
  probBars: document.querySelector("#probBars"),
};

function riskClass(level) {
  return `risk-${String(level || "Low").toLowerCase()}`;
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.remove("hidden");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => els.toast.classList.add("hidden"), 3400);
}

function setBusy(isBusy) {
  els.busyOverlay.classList.toggle("hidden", !isBusy);
  els.runButton.disabled = isBusy;
  els.refreshButton.disabled = isBusy;
}

function formatTime(hour) {
  const h = Number(hour);
  if (h >= 23) return "23:59";
  return `${String(h).padStart(2, "0")}:59`;
}

function fillSelect(select, values, label) {
  select.innerHTML = "";
  const all = document.createElement("option");
  all.value = "All";
  all.textContent = `All ${label}`;
  select.appendChild(all);
  values.forEach((value) => {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    select.appendChild(opt);
  });
}

function updateSummary(summary) {
  els.totalFlights.textContent = summary.total ?? 0;
  els.highRiskFlights.textContent = summary.highRisk ?? 0;
  els.mediumRiskFlights.textContent = summary.mediumRisk ?? 0;
  els.avgDelay.textContent = `${summary.averageDelay ?? 0} min`;
  els.topRoute.textContent = summary.topRiskRoute ?? "-";
}

function resetDashboard() {
  state.flights = [];
  state.filtered = [];
  state.selectedId = null;
  state.hasResults = false;
  updateSummary({
    total: "-",
    highRisk: "-",
    mediumRisk: "-",
    averageDelay: "-",
    topRiskRoute: "-",
  });
  els.avgDelay.textContent = "-";
  els.statusLine.textContent = "Upload a CSV to start";
  els.resultCount.textContent = "No CSV uploaded yet";
  els.flightRows.innerHTML = `
    <tr>
      <td colspan="6" class="empty-row">
        Upload a flight CSV, then press Run to see the risk queue.
      </td>
    </tr>
  `;
  els.emptyDetail.textContent = "Upload and run a CSV to review flight risk briefs.";
  els.emptyDetail.classList.remove("hidden");
  els.flightDetail.classList.add("hidden");
  fillSelect(els.originSelect, [], "origins");
  fillSelect(els.carrierSelect, [], "carriers");
  fillSelect(els.destSelect, [], "destinations");
  els.downloadButton.disabled = true;
}

function applyFilters() {
  const query = state.filters.search.trim().toLowerCase();
  state.filtered = state.flights.filter((flight) => {
    const searchable = [
      flight.flight_id,
      flight.carrier,
      flight.origin,
      flight.dest,
      flight.route,
      flight.main_reason,
    ].join(" ").toLowerCase();

    if (query && !searchable.includes(query)) return false;
    if (state.filters.risk !== "All" && flight.risk_level !== state.filters.risk) return false;
    if (state.filters.origin !== "All" && flight.origin !== state.filters.origin) return false;
    if (state.filters.carrier !== "All" && flight.carrier !== state.filters.carrier) return false;
    if (state.filters.dest !== "All" && flight.dest !== state.filters.dest) return false;
    if (Number(flight.dep_hour) > Number(state.filters.hour)) return false;
    return true;
  });

  state.filtered.sort((a, b) => b.delay_risk - a.delay_risk);
  if (!state.selectedId && state.filtered.length) {
    state.selectedId = state.filtered[0].flight_id;
  }
  if (state.selectedId && !state.filtered.some((f) => f.flight_id === state.selectedId)) {
    state.selectedId = state.filtered[0]?.flight_id ?? null;
  }
  renderTable();
  renderDetail();
}

function renderTable() {
  els.resultCount.textContent = `${state.filtered.length} flights shown`;
  const rows = state.filtered.map((flight) => {
    const selected = flight.flight_id === state.selectedId ? "selected" : "";
    return `
      <tr class="${selected}" data-flight-id="${flight.flight_id}">
        <td class="flight-cell">
          <strong>${flight.flight_id}</strong>
          <span class="subtext">${flight.carrier}</span>
        </td>
        <td class="route-cell">
          <strong>${flight.origin} → ${flight.dest}</strong>
          <span class="subtext">${flight.date}</span>
        </td>
        <td>${flight.sched_dep_local ?? "-"}</td>
        <td>
          <span class="risk-number">${flight.delay_risk_pct}%</span>
          <span class="risk-badge ${riskClass(flight.risk_level)}">${flight.risk_level}</span>
        </td>
        <td>
          <strong>${flight.predicted_delay_min} min</strong>
          <span class="subtext">${flight.predicted_delay_bucket_label}</span>
        </td>
        <td class="reason-cell">${flight.main_reason}</td>
      </tr>
    `;
  });
  els.flightRows.innerHTML = rows.join("");
}

function renderList(container, items) {
  container.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    container.appendChild(li);
  });
}

function renderRiskFactors(items) {
  const factors = items && items.length ? items : [{
    label: "Risk basis",
    value: "No dominant driver",
    level: "Low",
    detail: "The score comes from several smaller schedule and operating signals.",
  }];

  els.riskFactorList.innerHTML = factors.map((factor) => `
    <div class="risk-factor ${riskClass(factor.level)}">
      <div>
        <strong>${factor.label}</strong>
        <span>${factor.detail}</span>
      </div>
      <em>${factor.value}</em>
    </div>
  `).join("");
}

function renderProbBars(probabilities) {
  const rows = [
    ["On time", probabilities.on_time],
    ["15-30", probabilities.delay_15_30],
    ["30-60", probabilities.delay_30_60],
    ["60-90", probabilities.delay_60_90],
    ["90+", probabilities.delay_90_plus],
  ];
  els.probBars.innerHTML = rows.map(([label, value]) => {
    const width = Math.max(0, Math.min(100, Math.round((Number(value) || 0) * 100)));
    return `
      <div class="prob-row">
        <span class="prob-label">${label}</span>
        <span class="prob-track"><span class="prob-fill" style="width:${width}%"></span></span>
        <span class="prob-value">${width}%</span>
      </div>
    `;
  }).join("");
}

function renderDetail() {
  const flight = state.flights.find((item) => item.flight_id === state.selectedId);
  els.emptyDetail.classList.toggle("hidden", Boolean(flight));
  els.flightDetail.classList.toggle("hidden", !flight);
  if (!flight) return;

  els.detailRiskBadge.textContent = flight.risk_level;
  els.detailRiskBadge.className = `risk-badge ${riskClass(flight.risk_level)}`;
  els.detailFlight.textContent = flight.flight_id;
  els.detailRoute.textContent = `${flight.origin} → ${flight.dest} · ${flight.carrier}`;
  els.detailRisk.textContent = `${flight.delay_risk_pct}%`;
  els.detailTime.textContent = flight.sched_dep_local ?? "-";
  els.detailDelay.textContent = `${flight.predicted_delay_min} min`;
  els.detailBucket.textContent = flight.predicted_delay_bucket_label;
  renderList(els.reasonList, flight.reasons || []);
  renderList(els.actionList, flight.recommended_actions || []);
  renderRiskFactors(flight.risk_factors || []);
  renderProbBars(flight.probabilities || {});
}

async function loadResults() {
  const res = await fetch("/api/results");
  const data = await res.json();
  if (!res.ok || data.error) {
    throw new Error(data.error || "Could not load results.");
  }

  state.flights = data.flights || [];
  state.hasResults = true;
  updateSummary(data.summary || {});
  fillSelect(els.originSelect, data.filters?.origins || [], "origins");
  fillSelect(els.carrierSelect, data.filters?.carriers || [], "carriers");
  fillSelect(els.destSelect, data.filters?.destinations || [], "destinations");
  els.statusLine.textContent = "Ready for controller review";
  els.downloadButton.disabled = false;
  applyFilters();
}

async function runPrediction() {
  if (!state.uploadedFile) {
    showToast("Upload a CSV first.");
    els.statusLine.textContent = "Upload a CSV to start";
    return;
  }

  setBusy(true);
  try {
    const payload = {
      csvText: await state.uploadedFile.text(),
    };

    const res = await fetch("/api/run-prediction", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      throw new Error(data.error || "Prediction failed.");
    }

    state.flights = data.flights || [];
    state.hasResults = true;
    updateSummary(data.summary || {});
    fillSelect(els.originSelect, data.filters?.origins || [], "origins");
    fillSelect(els.carrierSelect, data.filters?.carriers || [], "carriers");
    fillSelect(els.destSelect, data.filters?.destinations || [], "destinations");
    state.selectedId = null;
    els.statusLine.textContent = "Prediction complete";
    els.downloadButton.disabled = false;
    showToast("Prediction complete. Risk queue updated.");
    applyFilters();
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
}

function bindEvents() {
  els.flightRows.addEventListener("click", (event) => {
    const row = event.target.closest("tr[data-flight-id]");
    if (!row) return;
    state.selectedId = row.dataset.flightId;
    renderTable();
    renderDetail();
  });

  els.searchInput.addEventListener("input", (event) => {
    state.filters.search = event.target.value;
    applyFilters();
  });

  document.querySelectorAll(".segment").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".segment").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      state.filters.risk = button.dataset.risk;
      applyFilters();
    });
  });

  els.originSelect.addEventListener("change", (event) => {
    state.filters.origin = event.target.value;
    applyFilters();
  });
  els.carrierSelect.addEventListener("change", (event) => {
    state.filters.carrier = event.target.value;
    applyFilters();
  });
  els.destSelect.addEventListener("change", (event) => {
    state.filters.dest = event.target.value;
    applyFilters();
  });
  els.hourRange.addEventListener("input", (event) => {
    state.filters.hour = Number(event.target.value);
    els.hourValue.textContent = formatTime(state.filters.hour);
    applyFilters();
  });

  els.clearFilters.addEventListener("click", () => {
    state.filters = { search: "", risk: "All", origin: "All", carrier: "All", dest: "All", hour: 23 };
    els.searchInput.value = "";
    els.originSelect.value = "All";
    els.carrierSelect.value = "All";
    els.destSelect.value = "All";
    els.hourRange.value = "23";
    els.hourValue.textContent = "23:59";
    document.querySelectorAll(".segment").forEach((item) => {
      item.classList.toggle("active", item.dataset.risk === "All");
    });
    applyFilters();
  });

  els.csvInput.addEventListener("change", (event) => {
    state.uploadedFile = event.target.files?.[0] || null;
    if (state.uploadedFile) {
      els.statusLine.textContent = `Selected ${state.uploadedFile.name}`;
      showToast("CSV selected. Press Run to score it.");
      state.hasResults = false;
      state.flights = [];
      state.filtered = [];
      state.selectedId = null;
      els.resultCount.textContent = "CSV selected, not scored yet";
      els.flightRows.innerHTML = `
        <tr>
          <td colspan="6" class="empty-row">
            Press Run to score ${state.uploadedFile.name}.
          </td>
        </tr>
      `;
      els.emptyDetail.textContent = "Run prediction to see flight details.";
      els.emptyDetail.classList.remove("hidden");
      els.flightDetail.classList.add("hidden");
      updateSummary({
        total: "-",
        highRisk: "-",
        mediumRisk: "-",
        averageDelay: "-",
        topRiskRoute: "-",
      });
      els.avgDelay.textContent = "-";
      els.downloadButton.disabled = true;
    }
  });

  els.runButton.addEventListener("click", runPrediction);
  els.refreshButton.addEventListener("click", () => {
    if (!state.hasResults) {
      showToast("Upload and run a CSV first.");
      return;
    }
    loadResults().then(() => showToast("Results refreshed.")).catch((error) => showToast(error.message));
  });
  els.downloadButton.addEventListener("click", () => {
    if (!state.hasResults) {
      showToast("Upload and run a CSV first.");
      return;
    }
    window.location.href = "/api/download";
  });
}

bindEvents();
resetDashboard();
