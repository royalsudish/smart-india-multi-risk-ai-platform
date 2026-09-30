import subprocess
import time
import json
import urllib.request
import asyncio
import websockets
import sys

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9224
BASE_URL = "http://127.0.0.1:5000"

async def test_dashboard():
    # 1. Start Chrome headless
    proc = subprocess.Popen([
        CHROME_PATH,
        f"--remote-debugging-port={PORT}",
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox"
    ])
    try:
        # Wait for debug port
        version_url = f"http://127.0.0.1:{PORT}/json/version"
        for _ in range(25):
            try:
                with urllib.request.urlopen(version_url) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                time.sleep(0.2)
        else:
            print("Failed to connect to Chrome debug port")
            return False

        # Open new tab
        new_tab_url = f"http://127.0.0.1:{PORT}/json/new"
        req = urllib.request.Request(new_tab_url, data=b"", method="PUT")
        with urllib.request.urlopen(req) as resp:
            tab_info = json.loads(resp.read().decode())
        
        ws_url = tab_info.get("webSocketDebuggerUrl")
        print(f"Connected to Chrome WebSocket: {ws_url}")

        console_messages = []
        js_errors = []

        async with websockets.connect(ws_url, max_size=20_000_000) as ws:
            # Enable domains
            await ws.send(json.dumps({"id": 1, "method": "Runtime.enable"}))
            await ws.send(json.dumps({"id": 2, "method": "Page.enable"}))
            await ws.send(json.dumps({"id": 3, "method": "DOM.enable"}))

            # Navigate to http://127.0.0.1:5000
            print(f"Navigating to {BASE_URL}...")
            await ws.send(json.dumps({
                "id": 4,
                "method": "Page.navigate",
                "params": {"url": BASE_URL}
            }))

            # Helper to send eval and get result
            cmd_id = 100
            async def evaluate(expr):
                nonlocal cmd_id
                cmd_id += 1
                my_id = cmd_id
                await ws.send(json.dumps({
                    "id": my_id,
                    "method": "Runtime.evaluate",
                    "params": {
                        "expression": expr,
                        "awaitPromise": True,
                        "returnByValue": True
                    }
                }))
                while True:
                    m = await ws.recv()
                    data = json.loads(m)
                    if data.get("method") == "Runtime.consoleAPICalled":
                        text = " ".join([str(a.get("value", a.get("description", ""))) for a in data.get("params", {}).get("args", [])])
                        console_messages.append(f"[{data['params']['type']}] {text}")
                    elif data.get("method") == "Runtime.exceptionThrown":
                        details = data.get("params", {}).get("exceptionDetails", {})
                        js_errors.append(f"EXCEPTION: {details.get('text')} - {details.get('exception', {}).get('description')}")
                    elif data.get("id") == my_id:
                        res = data.get("result", {})
                        if "exceptionDetails" in res:
                            js_errors.append(str(res["exceptionDetails"]))
                        return res.get("result", {}).get("value")

            # Wait 3 seconds for initial scripts to run and compile
            await asyncio.sleep(3)

            # Check if any syntax errors or exceptions occurred
            print("\n--- Initial Compilation & Console Check ---")
            print(f"Total Console logs captured: {len(console_messages)}")
            for log in console_messages[:10]:
                print(" ", log)
            print(f"JavaScript exceptions: {len(js_errors)}")
            for err in js_errors:
                print(" ", err)

            # Test 1: Verify title and DOM readiness
            title = await evaluate("document.title")
            print(f"\nDocument Title: '{title}'")

            # Test 2: Check map instance and layers
            map_ready = await evaluate("Boolean(window.map || (typeof mapState !== 'undefined' && mapState.map))")
            print(f"Leaflet Map initialized: {map_ready}")

            # Test 3: Test Location Search
            print("\n--- Testing Location Search Flow ---")
            search_cities = [
                "Patna, Bihar, India",
                "Delhi, India",
                "Mumbai, Maharashtra, India",
                "Guntur, Andhra Pradesh, India"
            ]
            for city in search_cities:
                res = await evaluate(f"""
                    (async () => {{
                        const inp = document.getElementById("placeSearchInput");
                        const btn = document.getElementById("searchLocationBtn");
                        if (!inp) return "placeSearchInput missing";
                        inp.value = "{city}";
                        inp.dispatchEvent(new Event("input", {{ bubbles: true }}));
                        if (btn) btn.click();
                        await new Promise(r => setTimeout(r, 1200));
                        const resultsBox = document.getElementById("placeSearchResults");
                        const firstItem = resultsBox ? resultsBox.querySelector(".search-result-item") : null;
                        const found = firstItem ? firstItem.textContent.trim() : "none";
                        if (firstItem) firstItem.click();
                        await new Promise(r => setTimeout(r, 600));
                        const selCard = document.getElementById("selectedLocationCard");
                        const selName = document.getElementById("selectedPlaceName");
                        return {{
                            searched: "{city}",
                            dropdownItem: found,
                            selectedName: selName ? selName.textContent.trim() : null,
                            cardVisible: selCard ? window.getComputedStyle(selCard).display !== "none" : false,
                            lat: document.getElementById("latInput") ? document.getElementById("latInput").value : null,
                            lon: document.getElementById("lonInput") ? document.getElementById("lonInput").value : null
                        }};
                    }})()
                """)
                print(f"Search '{city}':", json.dumps(res, indent=2))

            # Test 4: Test Analyze Live Conditions
            print("\n--- Testing Analyze Live Conditions CTA ---")
            analyze_res = await evaluate("""
                (async () => {
                    const btn = document.getElementById("analyzeBtn");
                    if (!btn) return "analyzeBtn missing";
                    btn.click();
                    // Wait for analysis to complete (up to 30s)
                    for (let i = 0; i < 30; i++) {
                        await new Promise(r => setTimeout(r, 1000));
                        const loading = document.getElementById("loadingIndicator");
                        const isHidden = !loading || loading.style.display === "none" || window.getComputedStyle(loading).display === "none";
                        const riskBadge = document.getElementById("overallRiskBadge");
                        if (isHidden && riskBadge && riskBadge.textContent.trim() !== "--") {
                            break;
                        }
                    }
                    const getTxt = (id) => {
                        const el = document.getElementById(id);
                        return el ? el.textContent.trim() : "--";
                    };
                    return {
                        overallRisk: getTxt("overallRiskBadge"),
                        weatherTemp: getTxt("wTemp"),
                        weatherHumidity: getTxt("wHumidity"),
                        waterloggingRisk: getTxt("wlRiskLevel"),
                        waterloggingProb: getTxt("wlProb"),
                        disasterClass: getTxt("dClass"),
                        disasterConfidence: getTxt("dConfidence"),
                        agricultureStatus: getTxt("agStatusBadge")
                    };
                })()
            """)
            print("Analysis Result on Dashboard:", json.dumps(analyze_res, indent=2))

            # Test 5: Test Layer Toggles
            print("\n--- Testing Layer Toggles ---")
            toggles_res = await evaluate("""
                (() => {
                    const checkboxes = document.querySelectorAll(".layer-toggle-item input[type='checkbox']");
                    const results = [];
                    checkboxes.forEach(cb => {
                        const id = cb.id;
                        const before = cb.checked;
                        cb.click();
                        const after = cb.checked;
                        cb.click(); // toggle back
                        results.push({ id, before, after, toggledBack: cb.checked });
                    });
                    return { count: checkboxes.length, toggles: results };
                })()
            """)
            print("Layer Toggles Result:", json.dumps(toggles_res, indent=2))

            # Test 6: Final check on console errors and unhandled rejections
            print("\n--- Final Console Audit ---")
            print(f"Total Console logs: {len(console_messages)}")
            print(f"Total JS Errors: {len(js_errors)}")
            if js_errors:
                for err in js_errors:
                    print("  ERROR:", err)
            else:
                print("  SUCCESS: 0 JavaScript SyntaxErrors, 0 uncaught exceptions!")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(test_dashboard())
