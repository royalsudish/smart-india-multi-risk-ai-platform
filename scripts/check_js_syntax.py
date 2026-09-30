import subprocess
import time
import json
import urllib.request
import urllib.error
import sys

def check_syntax():
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    port = 9223
    proc = subprocess.Popen([
        chrome_path,
        f"--remote-debugging-port={port}",
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox"
    ])
    try:
        # Wait for debugging port
        version_url = f"http://127.0.0.1:{port}/json/version"
        for _ in range(20):
            try:
                with urllib.request.urlopen(version_url) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                time.sleep(0.2)
        else:
            print("Failed to connect to Chrome debug port")
            return

        # Create a new page
        new_tab_url = f"http://127.0.0.1:{port}/json/new"
        req = urllib.request.Request(new_tab_url, data=b"", method="PUT")
        with urllib.request.urlopen(req) as resp:
            tab_info = json.loads(resp.read().decode())
        
        ws_url = tab_info.get("webSocketDebuggerUrl")
        print(f"Connected to WS: {ws_url}")

        # Connect via websockets
        import asyncio
        import websockets

        async def run():
            async with websockets.connect(ws_url) as ws:
                # Enable Page, Runtime, Console
                await ws.send(json.dumps({"id": 1, "method": "Runtime.enable"}))
                await ws.send(json.dumps({"id": 2, "method": "Page.enable"}))
                
                # Read the app.js file content
                with open("frontend/static/js/app.js", "r", encoding="utf-8") as f:
                    js_code = f.read()

                # Call Runtime.compileScript
                await ws.send(json.dumps({
                    "id": 3,
                    "method": "Runtime.compileScript",
                    "params": {
                        "expression": js_code,
                        "sourceURL": "app.js",
                        "persistScript": False
                    }
                }))

                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    if data.get("id") == 3:
                        res = data.get("result", {})
                        if "exceptionDetails" in res:
                            ex = res["exceptionDetails"]
                            print("=== CHROME SYNTAX ERROR ===")
                            print("Line:", ex.get("lineNumber", 0) + 1)
                            print("Column:", ex.get("columnNumber", 0) + 1)
                            print("Text:", ex.get("text"))
                            print("Exception:", ex.get("exception", {}).get("description"))
                        else:
                            print("SUCCESS: Chrome compiled app.js without any syntax errors!")
                        break

        asyncio.run(run())

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    check_syntax()
