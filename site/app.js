/* ==========================================================================
   Habitat Sentinel — client-side logic
   1) Live wildfire events from NASA EONET (fetched in-browser, no backend)
   2) Interactive risk gauge that replicates the trained logistic regression
      (coefficients loaded from model_results.json, produced by
      analysis/train_risk_model.py against real MODIS-derived data)
   ========================================================================== */

const EONET_URL = "https://eonet.gsfc.nasa.gov/api/v3/events?category=wildfires&status=open&limit=150";

// --------------------------------------------------------------------------
// Rough continent bucketing by bounding box, used only to surface a
// human-readable "most active region" stat from raw lat/lon points.
// --------------------------------------------------------------------------
function bucketRegion(lat, lon) {
  if (lat > 15 && lon > -170 && lon < -50) return "North America";
  if (lat <= 15 && lat > -60 && lon > -90 && lon < -30) return "South America";
  if (lat > 35 && lon > -12 && lon < 60) return "Europe";
  if (lat <= 35 && lat > -37 && lon > -20 && lon < 55) return "Africa";
  if (lon >= 60 && lon < 150 && lat > 5) return "Asia";
  if (lat <= -10 && lon >= 110 && lon < 180) return "Oceania";
  if (lat > 5 && lon >= 150) return "Pacific / Asia";
  return "Other";
}

function fmtTime(d) {
  return d.toISOString().slice(11, 16) + " UTC";
}

async function loadMap() {
  const map = L.map("map", { scrollWheelZoom: false }).setView([15, 10], 2);

  L.tileLayer(
    "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: "abcd",
      maxZoom: 18,
    }
  ).addTo(map);

  const statusChip = document.getElementById("status-chip");

  try {
    const res = await fetch(EONET_URL);
    if (!res.ok) throw new Error("EONET responded with " + res.status);
    const data = await res.json();
    const events = data.events || [];

    const regionCounts = {};
    let plotted = 0;

    events.forEach((ev) => {
      const geoms = ev.geometry || [];
      const last = geoms[geoms.length - 1];
      if (!last || !last.coordinates) return;

      const [lon, lat] = last.coordinates;
      if (typeof lat !== "number" || typeof lon !== "number") return;

      const region = bucketRegion(lat, lon);
      regionCounts[region] = (regionCounts[region] || 0) + 1;

      const isOpen = !ev.closed;
      const color = isOpen ? "#ff6a3d" : "#4fd1ae";

      const marker = L.circleMarker([lat, lon], {
        radius: 6,
        color: color,
        weight: 1.5,
        fillColor: color,
        fillOpacity: 0.55,
      }).addTo(map);

      const sourceLink =
        ev.sources && ev.sources[0]
          ? `<a href="${ev.sources[0].url}" target="_blank" rel="noopener" style="color:#4fd1ae">source</a>`
          : "";

      marker.bindPopup(
        `<strong>${ev.title}</strong><br/>${new Date(
          last.date
        ).toUTCString()}<br/>${sourceLink}`
      );
      plotted++;
    });

    document.getElementById("tele-count").textContent = String(plotted);

    const topRegion = Object.entries(regionCounts).sort(
      (a, b) => b[1] - a[1]
    )[0];
    document.getElementById("tele-region").textContent = topRegion
      ? `${topRegion[0]}`
      : "No open events";

    const now = new Date();
    document.getElementById("tele-updated").textContent = fmtTime(now);
    document.getElementById("tele-updated-sub").textContent =
      now.toISOString().slice(0, 10);

    statusChip.textContent = `Live · ${plotted} events plotted`;
  } catch (err) {
    console.error(err);
    statusChip.textContent = "Feed unreachable from this network";
    document.getElementById("tele-count").textContent = "—";
    document.getElementById("tele-region").textContent = "Unavailable";
    document.getElementById("tele-updated").textContent = "—";
    document.getElementById("tele-updated-sub").textContent =
      "Try opening this page directly (some sandboxed / offline environments block third-party fetches).";
  }
}

// --------------------------------------------------------------------------
// Risk gauge — replicates the trained logistic regression client-side.
// --------------------------------------------------------------------------
async function loadRiskModel() {
  const res = await fetch("model_results.json");
  const results = await res.json();

  const { coefficients, intercept, scaler_mean, scaler_scale } = results.model;
  const m = results.metrics;

  document.getElementById("metrics-strip").innerHTML = `
    <div class="m"><div class="k">Accuracy</div><div class="v">${(
      m.accuracy * 100
    ).toFixed(1)}%</div></div>
    <div class="m"><div class="k">Precision</div><div class="v">${(
      m.precision * 100
    ).toFixed(1)}%</div></div>
    <div class="m"><div class="k">Recall</div><div class="v">${(
      m.recall * 100
    ).toFixed(1)}%</div></div>
    <div class="m"><div class="k">Test n</div><div class="v">${
      results.n_test
    }</div></div>
  `;

  const ndviSlider = document.getElementById("ndvi-slider");
  const lstSlider = document.getElementById("lst-slider");
  const ndviVal = document.getElementById("ndvi-val");
  const lstVal = document.getElementById("lst-val");
  const gaugeValue = document.getElementById("gauge-value");
  const gaugeFill = document.getElementById("gauge-fill");
  const gaugeVerdict = document.getElementById("gauge-verdict");

  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  function update() {
    const ndvi = parseFloat(ndviSlider.value);
    const lst = parseFloat(lstSlider.value);
    ndviVal.textContent = ndvi.toFixed(2);
    lstVal.textContent = lst.toFixed(0);

    const ndviScaled = (ndvi - scaler_mean[0]) / scaler_scale[0];
    const lstScaled = (lst - scaler_mean[1]) / scaler_scale[1];

    const z =
      coefficients[0] * ndviScaled + coefficients[1] * lstScaled + intercept;
    const prob = sigmoid(z);
    const pct = Math.round(prob * 100);

    gaugeValue.textContent = pct + "%";
    gaugeFill.style.width = pct + "%";

    let color = "#4fd1ae";
    let verdict = "Low estimated risk under these conditions.";
    if (pct >= 66) {
      color = "#ff6a3d";
      verdict =
        "High estimated risk — hot, sparse vegetation matches the fire-labeled cases in the training data.";
    } else if (pct >= 33) {
      color = "#d9a441";
      verdict = "Moderate estimated risk — worth monitoring.";
    }
    gaugeValue.style.color = color;
    gaugeVerdict.textContent = verdict;
  }

  ndviSlider.addEventListener("input", update);
  lstSlider.addEventListener("input", update);
  update();
}

loadMap();
loadRiskModel();
