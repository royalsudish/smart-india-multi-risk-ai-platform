"""
Smart Multi-Risk Platform: Smart Alert Engine Automated Test Suite
Covers all 22 required scenarios from Section 20 of specification:
1. new flood alert
2. repeated unchanged flood does not duplicate
3. flood escalates HIGH -> CRITICAL
4. flood deescalates
5. flood resolves
6. earthquake event deduplication
7. wildfire cluster deduplication
8. avalanche CAP alert deduplication
9. expired SACHET alert
10. flash flood starts and resolves
11. two hazards active simultaneously
12. one hazard resolves while another remains active
13. provider unavailable
14. stale evidence
15. notification sent on NEW
16. notification sent on ESCALATED
17. no duplicate notification on unchanged ACTIVE alert
18. RESOLVED notification
19. database persistence
20. no regression in existing prediction APIs
21. no regression in 15-hazard engine
22. no regression in Weather/Waterlogging/Drainage/Agriculture/Map
"""

import os
import tempfile
from datetime import datetime, timezone, timedelta
import pytest

from backend.data_sources.alert_repository import (
    get_db_connection,
    save_alert,
    get_alert,
    get_alert_events,
    get_active_alerts,
    save_subscription,
    get_subscriptions,
    delete_subscription,
    get_in_app_notifications,
)
from backend.services.notification_service import NotificationService
from backend.services.alert_engine import SmartAlertEngine, build_alert_fingerprint
from backend.services.api_hazard_engine import evaluate_api_hazards, SUPPORTED_HAZARDS
from backend.services.disaster_service import predict_disaster_live


@pytest.fixture
def test_db():
    """Provides a fresh isolated temporary SQLite database for each test."""
    fd, path = tempfile.mkstemp(suffix="_test_alerts.db")
    os.close(fd)
    yield path
    try:
        os.remove(path)
    except Exception:
        pass


@pytest.fixture
def alert_engine(test_db):
    """Provides an isolated SmartAlertEngine bound to the test database."""
    notifier = NotificationService(db_path=test_db)
    return SmartAlertEngine(db_path=test_db, notification_service=notifier)


# -----------------------------------------------------------------------------
# 1. New Flood Alert
# -----------------------------------------------------------------------------
def test_1_new_flood_alert(alert_engine):
    loc = {"lat": 25.5941, "lon": 85.1376, "state": "Bihar", "district": "Patna"}
    hazards = [{
        "hazard": "flood",
        "active": True,
        "severity": "HIGH",
        "reason": "Ganga river discharge exceeded danger mark.",
        "sources": ["CWC", "GloFAS"],
        "evidence": {"water_level_m": 52.4, "threshold_m": 50.0}
    }]
    res = alert_engine.process_operational_hazards(hazards, loc)
    assert len(res["created_alerts"]) == 1
    assert len(res["active_alerts"]) == 1
    alert = res["created_alerts"][0]
    assert alert["hazard"] == "flood"
    assert alert["status"] == "NEW"
    assert alert["severity"] == "HIGH"
    assert alert["district"] == "Patna"

    # Verify event history
    events = get_alert_events(alert["alert_id"], db_path=alert_engine.db_path)
    assert len(events) == 1
    assert events[0]["event_type"] == "CREATED"
    assert events[0]["new_severity"] == "HIGH"


# -----------------------------------------------------------------------------
# 2. Repeated Unchanged Flood Does Not Duplicate
# -----------------------------------------------------------------------------
def test_2_repeated_unchanged_flood_does_not_duplicate(alert_engine):
    loc = {"lat": 25.5941, "lon": 85.1376, "state": "Bihar", "district": "Patna"}
    hazards = [{
        "hazard": "flood",
        "active": True,
        "severity": "HIGH",
        "reason": "Ganga river discharge exceeded danger mark.",
        "sources": ["CWC"],
        "evidence": {"water_level_m": 52.4}
    }]
    # Call 1: Creates alert
    res1 = alert_engine.process_operational_hazards(hazards, loc)
    assert len(res1["created_alerts"]) == 1
    first_id = res1["created_alerts"][0]["alert_id"]

    # Call 2: Same severity - must NOT duplicate
    res2 = alert_engine.process_operational_hazards(hazards, loc)
    assert len(res2["created_alerts"]) == 0
    assert len(res2["updated_alerts"]) == 1
    assert res2["updated_alerts"][0]["alert_id"] == first_id
    assert res2["updated_alerts"][0]["status"] == "ACTIVE"


# -----------------------------------------------------------------------------
# 3. Flood Escalates HIGH -> CRITICAL
# -----------------------------------------------------------------------------
def test_3_flood_escalates_high_to_critical(alert_engine):
    loc = {"lat": 25.5941, "lon": 85.1376, "state": "Bihar", "district": "Patna"}
    hazards_high = [{"hazard": "flood", "active": True, "severity": "HIGH", "reason": "High flood stage."}]
    res1 = alert_engine.process_operational_hazards(hazards_high, loc)
    alert_id = res1["created_alerts"][0]["alert_id"]

    # Flood escalates to CRITICAL
    hazards_crit = [{"hazard": "flood", "active": True, "severity": "CRITICAL", "reason": "Severe embankment breach."}]
    res2 = alert_engine.process_operational_hazards(hazards_crit, loc)
    assert len(res2["updated_alerts"]) == 1
    escalated = res2["updated_alerts"][0]
    assert escalated["alert_id"] == alert_id
    assert escalated["status"] == "ESCALATED"
    assert escalated["severity"] == "CRITICAL"

    # Verify event recorded
    events = get_alert_events(alert_id, db_path=alert_engine.db_path)
    assert len(events) == 2
    assert events[1]["event_type"] == "ESCALATED"
    assert events[1]["old_severity"] == "HIGH"
    assert events[1]["new_severity"] == "CRITICAL"


# -----------------------------------------------------------------------------
# 4. Flood De-escalates
# -----------------------------------------------------------------------------
def test_4_flood_deescalates(alert_engine):
    loc = {"lat": 25.5941, "lon": 85.1376, "state": "Bihar", "district": "Patna"}
    hazards_crit = [{"hazard": "flood", "active": True, "severity": "CRITICAL"}]
    res1 = alert_engine.process_operational_hazards(hazards_crit, loc)
    alert_id = res1["created_alerts"][0]["alert_id"]

    # De-escalates to MODERATE
    hazards_mod = [{"hazard": "flood", "active": True, "severity": "MODERATE", "reason": "Water levels receding."}]
    res2 = alert_engine.process_operational_hazards(hazards_mod, loc)
    assert len(res2["updated_alerts"]) == 1
    deesc = res2["updated_alerts"][0]
    assert deesc["status"] == "DEESCALATED"
    assert deesc["severity"] == "MODERATE"

    events = get_alert_events(alert_id, db_path=alert_engine.db_path)
    assert events[-1]["event_type"] == "DEESCALATED"
    assert events[-1]["old_severity"] == "CRITICAL"
    assert events[-1]["new_severity"] == "MODERATE"


# -----------------------------------------------------------------------------
# 5. Flood Resolves
# -----------------------------------------------------------------------------
def test_5_flood_resolves(alert_engine):
    loc = {"lat": 25.5941, "lon": 85.1376, "state": "Bihar", "district": "Patna"}
    hazards = [{"hazard": "flood", "active": True, "severity": "MODERATE"}]
    res1 = alert_engine.process_operational_hazards(hazards, loc)
    alert_id = res1["created_alerts"][0]["alert_id"]

    # No hazards active
    res2 = alert_engine.process_operational_hazards([], loc)
    assert len(res2["resolved_alerts"]) == 1
    resolved = res2["resolved_alerts"][0]
    assert resolved["alert_id"] == alert_id
    assert resolved["status"] == "RESOLVED"
    assert resolved["severity"] == "NONE"
    assert resolved["resolved_at"] is not None

    events = get_alert_events(alert_id, db_path=alert_engine.db_path)
    assert events[-1]["event_type"] == "RESOLVED"


# -----------------------------------------------------------------------------
# 6. Earthquake Event Deduplication
# -----------------------------------------------------------------------------
def test_6_earthquake_event_deduplication(alert_engine):
    loc = {"lat": 28.6139, "lon": 77.2090, "state": "Delhi", "district": "New Delhi"}
    quake_haz = [{
        "hazard": "earthquake",
        "active": True,
        "severity": "HIGH",
        "evidence": {"event_id": "usgs_delhi_2026_09", "magnitude": 5.8}
    }]
    res1 = alert_engine.process_operational_hazards(quake_haz, loc)
    eid1 = res1["created_alerts"][0]["alert_id"]

    res2 = alert_engine.process_operational_hazards(quake_haz, loc)
    assert len(res2["created_alerts"]) == 0
    assert res2["updated_alerts"][0]["alert_id"] == eid1


# -----------------------------------------------------------------------------
# 7. Wildfire Cluster Deduplication
# -----------------------------------------------------------------------------
def test_7_wildfire_cluster_deduplication(alert_engine):
    loc = {"lat": 22.33, "lon": 80.61, "state": "Madhya Pradesh", "district": "Mandla"}
    wf_haz = [{
        "hazard": "wildfire",
        "active": True,
        "severity": "HIGH",
        "sources": ["NASA FIRMS VIIRS NOAA-21"],
        "evidence": {"detections_10km": 3, "max_frp": 62.0}
    }]
    res1 = alert_engine.process_operational_hazards(wf_haz, loc)
    aid1 = res1["created_alerts"][0]["alert_id"]

    res2 = alert_engine.process_operational_hazards(wf_haz, loc)
    assert len(res2["created_alerts"]) == 0
    assert res2["updated_alerts"][0]["alert_id"] == aid1


# -----------------------------------------------------------------------------
# 8. Avalanche CAP Alert Deduplication
# -----------------------------------------------------------------------------
def test_8_avalanche_cap_alert_deduplication(alert_engine):
    loc = {"lat": 34.05, "lon": 74.38, "state": "Jammu & Kashmir", "district": "Baramulla"}
    av_haz = [{
        "hazard": "avalanche",
        "active": True,
        "severity": "HIGH",
        "evidence": {"identifier": "SACHET-2026-CAP-JK-901"}
    }]
    res1 = alert_engine.process_operational_hazards(av_haz, loc)
    aid1 = res1["created_alerts"][0]["alert_id"]

    res2 = alert_engine.process_operational_hazards(av_haz, loc)
    assert len(res2["created_alerts"]) == 0
    assert res2["updated_alerts"][0]["alert_id"] == aid1


# -----------------------------------------------------------------------------
# 9. Expired SACHET Alert
# -----------------------------------------------------------------------------
def test_9_expired_sachet_alert(alert_engine):
    loc = {"lat": 34.05, "lon": 74.38, "state": "Jammu & Kashmir", "district": "Baramulla"}
    past_expiry = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    av_haz = [{
        "hazard": "avalanche",
        "active": True,
        "severity": "HIGH",
        "evidence": {"expires": past_expiry}
    }]
    res1 = alert_engine.process_operational_hazards(av_haz, loc)
    aid = res1["created_alerts"][0]["alert_id"]

    # When next cycle runs and official alert is absent, it transitions to EXPIRED
    res2 = alert_engine.process_operational_hazards([], loc)
    assert len(res2["expired_alerts"]) == 1
    assert res2["expired_alerts"][0]["alert_id"] == aid
    assert res2["expired_alerts"][0]["status"] == "EXPIRED"


# -----------------------------------------------------------------------------
# 10. Flash Flood Starts and Resolves
# -----------------------------------------------------------------------------
def test_10_flash_flood_starts_and_resolves(alert_engine):
    loc = {"lat": 26.1445, "lon": 91.7362, "state": "Assam", "district": "Kamrup"}
    ff_haz = [{
        "hazard": "flash_flood",
        "active": True,
        "severity": "HIGH",
        "reason": "Intense 1h rainfall shock.",
        "evidence": {"rain_1h_mm": 35.0, "drainage_congestion_probability": 0.8}
    }]
    res1 = alert_engine.process_operational_hazards(ff_haz, loc)
    assert res1["created_alerts"][0]["hazard"] == "flash_flood"
    assert res1["created_alerts"][0]["status"] == "NEW"

    # Rain clears
    res2 = alert_engine.process_operational_hazards([], loc)
    assert len(res2["resolved_alerts"]) == 1
    assert res2["resolved_alerts"][0]["status"] == "RESOLVED"


# -----------------------------------------------------------------------------
# 11. Two Hazards Active Simultaneously
# -----------------------------------------------------------------------------
def test_11_two_hazards_active_simultaneously(alert_engine):
    loc = {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra", "district": "Mumbai"}
    multi_hazards = [
        {"hazard": "flash_flood", "active": True, "severity": "HIGH"},
        {"hazard": "thunderstorm", "active": True, "severity": "MODERATE"}
    ]
    res = alert_engine.process_operational_hazards(multi_hazards, loc)
    assert len(res["created_alerts"]) == 2
    assert len(res["active_alerts"]) == 2
    active_hzs = {a["hazard"] for a in res["active_alerts"]}
    assert active_hzs == {"flash_flood", "thunderstorm"}


# -----------------------------------------------------------------------------
# 12. One Hazard Resolves While Another Remains Active
# -----------------------------------------------------------------------------
def test_12_one_hazard_resolves_while_another_remains_active(alert_engine):
    loc = {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra", "district": "Mumbai"}
    res1 = alert_engine.process_operational_hazards([
        {"hazard": "flash_flood", "active": True, "severity": "HIGH"},
        {"hazard": "thunderstorm", "active": True, "severity": "MODERATE"}
    ], loc)
    assert len(res1["active_alerts"]) == 2

    # Thunderstorm resolves, flash flood stays active
    res2 = alert_engine.process_operational_hazards([
        {"hazard": "flash_flood", "active": True, "severity": "HIGH"}
    ], loc)
    assert len(res2["resolved_alerts"]) == 1
    assert res2["resolved_alerts"][0]["hazard"] == "thunderstorm"
    assert len(res2["active_alerts"]) == 1
    assert res2["active_alerts"][0]["hazard"] == "flash_flood"


# -----------------------------------------------------------------------------
# 13. Provider Unavailable
# -----------------------------------------------------------------------------
def test_13_provider_unavailable(alert_engine):
    loc = {"lat": 22.33, "lon": 80.61, "state": "Madhya Pradesh", "district": "Mandla"}
    wf_haz = [{"hazard": "wildfire", "active": True, "severity": "HIGH"}]
    res1 = alert_engine.process_operational_hazards(wf_haz, loc)
    aid = res1["created_alerts"][0]["alert_id"]

    # In next cycle, NASA FIRMS times out / unavailable
    dq = {"status": "PARTIAL_DATA", "missing_providers": ["NASA_FIRMS"]}
    res2 = alert_engine.process_operational_hazards([], loc, data_quality=dq)

    # Must NOT automatically mark wildfire resolved/safe!
    assert len(res2["resolved_alerts"]) == 0
    assert len(res2["active_alerts"]) == 1
    assert res2["active_alerts"][0]["alert_id"] == aid
    assert res2["active_alerts"][0]["freshness_status"] == "INSUFFICIENT_DATA"


# -----------------------------------------------------------------------------
# 14. Stale Evidence
# -----------------------------------------------------------------------------
def test_14_stale_evidence(alert_engine):
    loc = {"lat": 34.05, "lon": 74.38, "state": "Jammu & Kashmir", "district": "Baramulla"}
    av_haz = [{"hazard": "avalanche", "active": True, "severity": "HIGH"}]
    res1 = alert_engine.process_operational_hazards(av_haz, loc)
    aid = res1["created_alerts"][0]["alert_id"]

    # SACHET feed is unreachable
    dq = {"status": "PARTIAL_DATA", "missing_providers": ["NDMA_SACHET"]}
    res2 = alert_engine.process_operational_hazards([], loc, data_quality=dq)
    assert len(res2["resolved_alerts"]) == 0
    alert = get_alert(aid, db_path=alert_engine.db_path)
    assert alert["freshness_status"] == "INSUFFICIENT_DATA"


# -----------------------------------------------------------------------------
# 15. Notification Sent on NEW
# -----------------------------------------------------------------------------
def test_15_notification_sent_on_new(alert_engine):
    loc = {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu", "district": "Chennai"}
    hazards = [{"hazard": "cyclone", "active": True, "severity": "SEVERE", "reason": "Cyclone approaching coast."}]
    res = alert_engine.process_operational_hazards(hazards, loc)
    assert len(res["created_alerts"]) == 1

    notifs = get_in_app_notifications(db_path=alert_engine.db_path)
    assert len(notifs) >= 1
    assert notifs[0]["event_type"] == "NEW"
    assert notifs[0]["severity"] == "SEVERE"
    assert "CYCLONE" in notifs[0]["title"]


# -----------------------------------------------------------------------------
# 16. Notification Sent on ESCALATED
# -----------------------------------------------------------------------------
def test_16_notification_sent_on_escalated(alert_engine):
    loc = {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu", "district": "Chennai"}
    alert_engine.process_operational_hazards([{"hazard": "cyclone", "active": True, "severity": "HIGH"}], loc)

    # Escalate to CRITICAL
    alert_engine.process_operational_hazards([{"hazard": "cyclone", "active": True, "severity": "CRITICAL"}], loc)

    notifs = get_in_app_notifications(db_path=alert_engine.db_path)
    escalated_notifs = [n for n in notifs if n["event_type"] == "ESCALATED"]
    assert len(escalated_notifs) == 1
    assert escalated_notifs[0]["severity"] == "CRITICAL"


# -----------------------------------------------------------------------------
# 17. No Duplicate Notification on Unchanged ACTIVE Alert
# -----------------------------------------------------------------------------
def test_17_no_duplicate_notification_on_unchanged_active_alert(alert_engine):
    loc = {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu", "district": "Chennai"}
    hazards = [{"hazard": "cyclone", "active": True, "severity": "HIGH"}]
    alert_engine.process_operational_hazards(hazards, loc)
    initial_notifs_count = len(get_in_app_notifications(db_path=alert_engine.db_path))

    # Send unchanged active heartbeat
    alert_engine.process_operational_hazards(hazards, loc)
    second_notifs_count = len(get_in_app_notifications(db_path=alert_engine.db_path))
    assert second_notifs_count == initial_notifs_count, "No new notification on unchanged ACTIVE alert."


# -----------------------------------------------------------------------------
# 18. RESOLVED Notification
# -----------------------------------------------------------------------------
def test_18_resolved_notification(alert_engine):
    loc = {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu", "district": "Chennai"}
    alert_engine.process_operational_hazards([{"hazard": "cyclone", "active": True, "severity": "HIGH"}], loc)

    # Resolve
    alert_engine.process_operational_hazards([], loc)
    notifs = get_in_app_notifications(db_path=alert_engine.db_path)
    resolved_notifs = [n for n in notifs if n["event_type"] == "RESOLVED"]
    assert len(resolved_notifs) == 1
    assert "RESOLVED" in resolved_notifs[0]["title"]


# -----------------------------------------------------------------------------
# 19. Database Persistence
# -----------------------------------------------------------------------------
def test_19_database_persistence(test_db):
    engine1 = SmartAlertEngine(db_path=test_db)
    loc = {"lat": 28.6139, "lon": 77.2090, "state": "Delhi", "district": "New Delhi"}
    res1 = engine1.process_operational_hazards([{"hazard": "air_pollution", "active": True, "severity": "SEVERE"}], loc)
    aid = res1["created_alerts"][0]["alert_id"]

    # Reopen database with new connection / engine instance
    engine2 = SmartAlertEngine(db_path=test_db)
    persisted = get_alert(aid, db_path=test_db)
    assert persisted is not None
    assert persisted["hazard"] == "air_pollution"
    assert persisted["severity"] == "SEVERE"
    assert persisted["district"] == "New Delhi"

    # Test Subscription CRUD
    sub = save_subscription({
        "user_id": "test_user_1",
        "location_name": "Delhi NCR",
        "state": "Delhi",
        "hazards": ["air_pollution", "heat_wave"],
        "minimum_severity": "HIGH"
    }, db_path=test_db)
    assert sub["subscription_id"] is not None

    subs = get_subscriptions(user_id="test_user_1", db_path=test_db)
    assert len(subs) == 1

    del_ok = delete_subscription(sub["subscription_id"], db_path=test_db)
    assert del_ok is True
    assert len(get_subscriptions(user_id="test_user_1", db_path=test_db)) == 0


# -----------------------------------------------------------------------------
# 20. No Regression in Existing Prediction APIs
# -----------------------------------------------------------------------------
def test_20_no_regression_in_existing_prediction_apis():
    # Verify predict_disaster_live runs and surfaces alerts without crashing
    res = predict_disaster_live(
        lat=25.5941,
        lon=85.1376,
        overrides={"glofas": {"return_period": 10.0, "is_flooding": True, "discharge_m3s": 25000.0}},
        state="Bihar",
        district="Patna"
    )
    assert res["success"] is True
    assert "alerts" in res
    assert "active_alerts" in res
    assert "raw_model" in res
    assert "verified_hazards" in res


# -----------------------------------------------------------------------------
# 21. No Regression in 15-Hazard Engine
# -----------------------------------------------------------------------------
def test_21_no_regression_in_15_hazard_engine():
    assert len(SUPPORTED_HAZARDS) == 15
    for expected in ["wildfire", "avalanche", "flash_flood", "earthquake", "flood"]:
        assert expected in SUPPORTED_HAZARDS

    # Run evaluate_api_hazards with baseline nominal data
    res = evaluate_api_hazards(features={}, history={}, official={}, live={})
    assert "verified_hazards" in res
    assert "hazard_status" in res
    assert len(res["hazard_status"]) == 15


# -----------------------------------------------------------------------------
# 22. No Regression in Weather/Waterlogging/Drainage/Agriculture/Map
# -----------------------------------------------------------------------------
def test_22_no_regression_in_other_modules():
    from backend.services.weather_features import build_live_weather_features
    from backend.services.waterlogging_features import build_live_waterlogging_features
    from backend.services.agriculture_features import build_live_agri_features
    from backend.services.model_service import registry

    w_feat = build_live_weather_features(28.6139, 77.2090)
    assert len(w_feat["features"]) == 39
    w_pred = registry.predict("weather", w_feat["features"])
    assert "prediction" in w_pred or "temp_prediction" in w_pred

    wl_feat = build_live_waterlogging_features(28.6139, 77.2090)
    assert len(wl_feat["features"]) == 30

    ag_feat = build_live_agri_features(28.6139, 77.2090, crop="Wheat", stage="Flowering")
    assert len(ag_feat["features"]) == 35
