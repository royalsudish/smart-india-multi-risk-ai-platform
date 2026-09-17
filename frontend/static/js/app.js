const statusGrid = document.getElementById("statusGrid");
const result = document.getElementById("result");

async function getStatus() {
  const r = await fetch("/api/models/status");
  const j = await r.json();

  statusGrid.innerHTML = "";
  const data = j.data || {};

  Object.entries(data).forEach(([name, info]) => {
    const div = document.createElement("div");
    div.className = "card " + (info.loaded ? "ok" : "bad");
    div.innerHTML = `
      <h3>${name.toUpperCase()}</h3>
      <p><strong>Loaded:</strong> ${info.loaded}</p>
      <p><strong>File:</strong> ${info.model_file || "Not found"}</p>
      <p><strong>Features:</strong> ${(info.features || []).join(", ") || "Not defined"}</p>
      ${info.load_error ? `<p><strong>Error:</strong> ${info.load_error}</p>` : ""}
    `;
    statusGrid.appendChild(div);
  });
}

function parseBox(id) {
  const text = document.getElementById(id).value.trim();
  return text ? JSON.parse(text) : {};
}

document.getElementById("predictBtn").addEventListener("click", async () => {
  try {
    const payload = {
      lat: Number(document.getElementById("lat").value),
      lon: Number(document.getElementById("lon").value),
      auto_weather: document.getElementById("autoWeather").checked,
      weather: parseBox("weather"),
      agriculture: parseBox("agriculture"),
      waterlogging: parseBox("waterlogging"),
      disaster: parseBox("disaster")
    };

    result.textContent = "Running...";

    const r = await fetch("/api/predict-all", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });

    const j = await r.json();
    result.textContent = JSON.stringify(j, null, 2);
  } catch (err) {
    result.textContent = "Error: " + err.message;
  }
});

document.getElementById("reloadBtn").addEventListener("click", async () => {
  await fetch("/api/models/reload", {method: "POST"});
  await getStatus();
});

getStatus();
