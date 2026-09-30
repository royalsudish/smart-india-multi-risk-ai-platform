import subprocess
import time
import json
import urllib.request
import asyncio
import websockets
import sys

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9225
BASE_URL = "http://127.0.0.1:5000"

async def test_search():
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
            return

        new_tab_url = f"http://127.0.0.1:{PORT}/json/new"
        req = urllib.request.Request(new_tab_url, data=b"", method="PUT")
        with urllib.request.urlopen(req) as resp:
            tab_info = json.loads(resp.read().decode())
        
        ws_url = tab_info.get("webSocketDebuggerUrl")

        async with websockets.connect(ws_url, max_size=20_000_000) as ws:
            await ws.send(json.dumps({"id": 1, "method": "Runtime.enable"}))
            await ws.send(json.dumps({"id": 2, "method": "Page.enable"}))

            await ws.send(json.dumps({
                "id": 4,
                "method": "Page.navigate",
                "params": {"url": BASE_URL}
            }))

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
                    if data.get("id") == my_id:
                        res = data.get("result", {})
                        return res.get("result", {}).get("value")

            # Wait for initial load
            await asyncio.sleep(2)

            test_cities = [
                "Patna, Bihar, India",
                "Delhi, India",
                "Mumbai, Maharashtra, India",
                "Guntur, Andhra Pradesh, India"
            ]

            print("=" * 80)
            print("TESTING 4-CITY SEARCH FLOW IN REAL CHROME")
            print("=" * 80)

            for city in test_cities:
                print(f"\n--- Searching: '{city}' ---")
                res = await evaluate(f"""
                    (async () => {{
                        const inp = document.getElementById("placeSearchInput");
                        const btn = document.getElementById("searchLocationBtn");
                        inp.value = "{city}";
                        btn.click();

                        // Wait for search results container to appear
                        const resultsBox = document.getElementById("placeSearchResults");
                        let foundItem = null;
                        for (let i = 0; i < 20; i++) {{
                            await new Promise(r => setTimeout(r, 200));
                            if (resultsBox && resultsBox.style.display !== "none") {{
                                const items = resultsBox.querySelectorAll(".search-result-item");
                                if (items.length > 0) {{
                                    foundItem = items[0];
                                    break;
                                }}
                            }}
                        }}

                        if (!foundItem) {{
                            return {{ error: "No search results dropdown appeared", city: "{city}" }};
                        }}

                        const itemName = foundItem.textContent.trim();
                        // Click the item
                        foundItem.click();
                        await new Promise(r => setTimeout(r, 500));

                        const selCard = document.getElementById("selectedLocationCard");
                        const selName = document.getElementById("selectedPlaceName");
                        const latVal = document.getElementById("latInput").value;
                        const lonVal = document.getElementById("lonInput").value;
                        const mapCenter = (typeof mapState !== "undefined" && mapState.map) ? mapState.map.getCenter() : null;

                        return {{
                            success: true,
                            city: "{city}",
                            dropdownSelected: itemName,
                            selectedPlaceName: selName ? selName.textContent.trim() : null,
                            cardVisible: selCard && window.getComputedStyle(selCard).display !== "none",
                            lat: latVal,
                            lon: lonVal,
                            mapCenter: mapCenter ? {{ lat: mapCenter.lat.toFixed(4), lng: mapCenter.lng.toFixed(4) }} : null
                        }};
                    }})()
                """)
                print(json.dumps(res, indent=2))
                assert res.get("success") is True, f"Search failed for {city}: {res}"
                assert float(res["lat"]) != 0, f"Lat must be valid for {city}"
                assert float(res["lon"]) != 0, f"Lon must be valid for {city}"
                assert res["cardVisible"] is True, f"Card must be visible for {city}"
                print(f" [PASS] Location '{city}' selected: ({res['lat']}, {res['lon']}), card visible: {res['cardVisible']}")

            print("\n>>> ALL 4 CITIES SEARCH FLOW VERIFIED 100% IN BROWSER! <<<")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(test_search())
