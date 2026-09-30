import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from app import create_app

app = create_app()
client = app.test_client()

print("=" * 70)
print("AUDITING ALERT APIS")
print("=" * 70)

# 1. GET /api/alerts/active
r_active = client.get("/api/alerts/active")
print(f"GET /api/alerts/active: {r_active.status_code}")
assert r_active.status_code == 200
data_active = r_active.get_json()
assert data_active["success"] is True
active_list = data_active["data"].get("alerts", [])
print(f"  Active alerts count: {len(active_list)}")

# 2. GET /api/alerts/history
r_hist = client.get("/api/alerts/history?limit=10")
print(f"GET /api/alerts/history: {r_hist.status_code}")
assert r_hist.status_code == 200
data_hist = r_hist.get_json()
assert data_hist["success"] is True
events_list = data_hist["data"].get("events", [])
print(f"  History events count: {len(events_list)}")

# 3. GET /api/alerts/<alert_id>
if active_list:
    test_id = active_list[0]["alert_id"]
elif events_list:
    test_id = events_list[0]["alert_id"]
else:
    test_id = None

if test_id:
    r_single = client.get(f"/api/alerts/{test_id}")
    print(f"GET /api/alerts/{test_id}: {r_single.status_code}")
    assert r_single.status_code == 200
    data_single = r_single.get_json()
    assert data_single["success"] is True
    alert_obj = data_single["data"]
    assert alert_obj["alert_id"] == test_id
    print(f"  Alert hazard: {alert_obj.get('hazard')}, events count: {len(alert_obj.get('events', []))}")

# 4. GET /api/alerts/subscriptions
r_sub = client.get("/api/alerts/subscriptions")
print(f"GET /api/alerts/subscriptions: {r_sub.status_code}")
assert r_sub.status_code == 200
data_sub = r_sub.get_json()
assert data_sub["success"] is True
print(f"  Subscriptions count: {len(data_sub['data'].get('subscriptions', []))}")

# 5. POST & DELETE /api/alerts/subscriptions
sub_payload = {
    "user_id": "test_user_audit",
    "channel": "in_app",
    "destination": "test_user_audit",
    "state": "Bihar",
    "district": "Patna",
    "hazard": "flood",
    "min_severity": "HIGH"
}
r_post_sub = client.post("/api/alerts/subscriptions", json=sub_payload)
print(f"POST /api/alerts/subscriptions: {r_post_sub.status_code}")
assert r_post_sub.status_code in [200, 201]
post_sub_data = r_post_sub.get_json()
assert post_sub_data["success"] is True
created_sub_id = post_sub_data["data"]["subscription_id"]
print(f"  Created subscription ID: {created_sub_id}")

r_del_sub = client.delete(f"/api/alerts/subscriptions/{created_sub_id}")
print(f"DELETE /api/alerts/subscriptions/{created_sub_id}: {r_del_sub.status_code}")
assert r_del_sub.status_code == 200
del_sub_data = r_del_sub.get_json()
assert del_sub_data["success"] is True

# 6. GET /api/alerts/notifications
r_notif = client.get("/api/alerts/notifications?limit=10")
print(f"GET /api/alerts/notifications: {r_notif.status_code}")
assert r_notif.status_code == 200
data_notif = r_notif.get_json()
assert data_notif["success"] is True
print(f"  Notifications count: {len(data_notif['data'].get('notifications', []))}")

print("\n>>> ALL ALERT APIS VALIDATED SUCCESSFULLY WITH ZERO DEFECTS! <<<")
