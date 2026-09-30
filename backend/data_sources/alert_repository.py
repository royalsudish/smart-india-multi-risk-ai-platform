"""
Smart Multi-Risk Platform: Alert Repository & Persistence Engine
SQLite-backed persistent store for operational alerts, lifecycle event history,
user/location subscriptions, and in-app notifications.

Features:
- Thread-safe connection context manager
- WAL journal mode for high concurrency
- Indexed queries for alert fingerprint, hazard, status, state, district, and timestamps
- Normalized AlertEvent history tracking lifecycle progression
- JSON serialization/deserialization for complex telemetry snapshots
- Fully compatible with in-memory SQLite (:memory:) for isolated unit testing
"""

import os
import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from backend.config.settings import ALERT_DB_PATH

_thread_local = threading.local()
_init_lock = threading.Lock()
_initialized_dbs = set()


def get_now_utc_iso() -> str:
    """Returns standard ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Retrieves or establishes a thread-local SQLite connection with foreign keys enabled
    and row factory set to sqlite3.Row for dictionary-like access.
    """
    path = db_path or ALERT_DB_PATH
    
    # If using :memory:, thread-local connection must be persistent per thread
    conn_key = f"conn_{path}"
    conn = getattr(_thread_local, conn_key, None)
    
    if conn is None:
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        if path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL;")
        setattr(_thread_local, conn_key, conn)
    
    init_db_schema(conn, path)
    return conn


def init_db_schema(conn: sqlite3.Connection, path: str):
    """Initializes tables and high-performance indexes if not already initialized."""
    with _init_lock:
        if path in _initialized_dbs:
            return
        
        cursor = conn.cursor()
        
        # 1. Operational Alerts Master Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                hazard TEXT NOT NULL,
                status TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                state TEXT,
                district TEXT,
                sources TEXT,
                evidence TEXT,
                confidence_type TEXT,
                started_at TEXT NOT NULL,
                last_updated_at TEXT NOT NULL,
                last_verified_at TEXT NOT NULL,
                resolved_at TEXT,
                expires_at TEXT,
                provider_event_id TEXT,
                data_quality TEXT,
                notification_state TEXT,
                last_evidence_at TEXT,
                provider_timestamp TEXT,
                freshness_status TEXT DEFAULT 'FRESH'
            );
        """)
        
        # Indexes for Alerts
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_fingerprint ON alerts(fingerprint);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_hazard ON alerts(hazard);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_state_dist ON alerts(state, district);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_started_at ON alerts(started_at);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_last_updated ON alerts(last_updated_at);")

        # 2. Alert Lifecycle Event History Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                old_severity TEXT,
                new_severity TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                evidence_snapshot TEXT,
                reason TEXT,
                FOREIGN KEY (alert_id) REFERENCES alerts(alert_id) ON DELETE CASCADE
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_alert_id ON alert_events(alert_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON alert_events(timestamp);")

        # 3. User & Location Alert Subscriptions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_subscriptions (
                subscription_id TEXT PRIMARY KEY,
                user_id TEXT,
                location_name TEXT,
                lat REAL,
                lon REAL,
                state TEXT,
                district TEXT,
                hazards TEXT,
                minimum_severity TEXT DEFAULT 'MODERATE',
                notification_channels TEXT,
                enabled INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subs_user_id ON alert_subscriptions(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subs_state_dist ON alert_subscriptions(state, district);")

        # 4. In-App Notifications Queue Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS in_app_notifications (
                notification_id TEXT PRIMARY KEY,
                alert_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                channel TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (alert_id) REFERENCES alerts(alert_id) ON DELETE CASCADE
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_notif_alert_id ON in_app_notifications(alert_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_notif_created_at ON in_app_notifications(created_at);")

        conn.commit()
        _initialized_dbs.add(path)


def _row_to_alert_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Deserializes JSON columns from an alerts row."""
    d = dict(row)
    for col in ("sources", "evidence", "data_quality", "notification_state"):
        if d.get(col) and isinstance(d[col], str):
            try:
                d[col] = json.loads(d[col])
            except Exception:
                pass
        elif d.get(col) is None and col in ("sources",):
            d[col] = []
        elif d.get(col) is None and col in ("evidence", "data_quality", "notification_state"):
            d[col] = {}
    return d


def _row_to_event_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Deserializes JSON columns from an alert_events row."""
    d = dict(row)
    if d.get("evidence_snapshot") and isinstance(d["evidence_snapshot"], str):
        try:
            d["evidence_snapshot"] = json.loads(d["evidence_snapshot"])
        except Exception:
            pass
    return d


def _row_to_sub_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Deserializes JSON columns from an alert_subscriptions row."""
    d = dict(row)
    for col in ("hazards", "notification_channels"):
        if d.get(col) and isinstance(d[col], str):
            try:
                d[col] = json.loads(d[col])
            except Exception:
                d[col] = []
    d["enabled"] = bool(d.get("enabled", 1))
    return d


# =========================================================================
# CRUD OPERATIONS: ALERTS
# =========================================================================

def save_alert(alert: Dict[str, Any], db_path: Optional[str] = None) -> Dict[str, Any]:
    """Inserts a new alert or updates an existing alert."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    alert_id = alert["alert_id"]
    fingerprint = alert.get("fingerprint") or ""
    hazard = alert.get("hazard") or ""
    status = alert.get("status") or "NEW"
    severity = alert.get("severity") or "LOW"
    title = alert.get("title") or f"{hazard.upper()} Alert"
    description = alert.get("description") or ""
    lat = float(alert.get("lat") or 0.0)
    lon = float(alert.get("lon") or 0.0)
    state = alert.get("state") or ""
    district = alert.get("district") or ""
    sources = json.dumps(alert.get("sources") or [])
    evidence = json.dumps(alert.get("evidence") or {})
    confidence_type = alert.get("confidence_type") or "multi_source_evidence"
    started_at = alert.get("started_at") or get_now_utc_iso()
    last_updated_at = alert.get("last_updated_at") or get_now_utc_iso()
    last_verified_at = alert.get("last_verified_at") or get_now_utc_iso()
    resolved_at = alert.get("resolved_at")
    expires_at = alert.get("expires_at")
    provider_event_id = alert.get("provider_event_id")
    data_quality = json.dumps(alert.get("data_quality") or {})
    notification_state = json.dumps(alert.get("notification_state") or {})
    last_evidence_at = alert.get("last_evidence_at")
    provider_timestamp = alert.get("provider_timestamp")
    freshness_status = alert.get("freshness_status") or "FRESH"

    cursor.execute("""
        INSERT INTO alerts (
            alert_id, fingerprint, hazard, status, severity, title, description,
            lat, lon, state, district, sources, evidence, confidence_type,
            started_at, last_updated_at, last_verified_at, resolved_at, expires_at,
            provider_event_id, data_quality, notification_state,
            last_evidence_at, provider_timestamp, freshness_status
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ON CONFLICT(alert_id) DO UPDATE SET
            fingerprint = excluded.fingerprint,
            hazard = excluded.hazard,
            status = excluded.status,
            severity = excluded.severity,
            title = excluded.title,
            description = excluded.description,
            lat = excluded.lat,
            lon = excluded.lon,
            state = excluded.state,
            district = excluded.district,
            sources = excluded.sources,
            evidence = excluded.evidence,
            confidence_type = excluded.confidence_type,
            last_updated_at = excluded.last_updated_at,
            last_verified_at = excluded.last_verified_at,
            resolved_at = excluded.resolved_at,
            expires_at = excluded.expires_at,
            provider_event_id = excluded.provider_event_id,
            data_quality = excluded.data_quality,
            notification_state = excluded.notification_state,
            last_evidence_at = excluded.last_evidence_at,
            provider_timestamp = excluded.provider_timestamp,
            freshness_status = excluded.freshness_status;
    """, (
        alert_id, fingerprint, hazard, status, severity, title, description,
        lat, lon, state, district, sources, evidence, confidence_type,
        started_at, last_updated_at, last_verified_at, resolved_at, expires_at,
        provider_event_id, data_quality, notification_state,
        last_evidence_at, provider_timestamp, freshness_status
    ))
    conn.commit()
    return get_alert(alert_id, db_path=db_path)


def get_alert(alert_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetches an alert by primary key alert_id."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM alerts WHERE alert_id = ?;", (alert_id,))
    row = cursor.fetchone()
    return _row_to_alert_dict(row) if row else None


def get_alert_by_fingerprint(
    fingerprint: str,
    open_only: bool = True,
    db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Finds existing alert with the given fingerprint.
    If open_only is True, filters to open states (not RESOLVED/EXPIRED).
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    if open_only:
        cursor.execute(
            """
            SELECT * FROM alerts
            WHERE fingerprint = ?
              AND status IN ('NEW', 'ACTIVE', 'UPDATED', 'ESCALATED', 'DEESCALATED')
            ORDER BY started_at DESC
            LIMIT 1;
            """,
            (fingerprint,)
        )
    else:
        cursor.execute(
            "SELECT * FROM alerts WHERE fingerprint = ? ORDER BY started_at DESC LIMIT 1;",
            (fingerprint,)
        )
    row = cursor.fetchone()
    return _row_to_alert_dict(row) if row else None


def get_open_alerts_for_location(
    state: Optional[str] = None,
    district: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: float = 35.0,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves all open alerts matching state/district or within radius_km.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Query all currently non-resolved, non-expired alerts
    cursor.execute("""
        SELECT * FROM alerts
        WHERE status IN ('NEW', 'ACTIVE', 'UPDATED', 'ESCALATED', 'DEESCALATED')
        ORDER BY started_at DESC;
    """)
    rows = cursor.fetchall()
    results = []
    
    st_norm = state.strip().lower() if state else None
    dt_norm = district.strip().lower() if district else None

    for row in rows:
        a = _row_to_alert_dict(row)
        a_st = (a.get("state") or "").strip().lower()
        a_dt = (a.get("district") or "").strip().lower()

        # Match 1: Exact district and state match
        if dt_norm and a_dt and dt_norm == a_dt:
            results.append(a)
            continue
        
        # Match 2: State match if district not specified
        if st_norm and not dt_norm and a_st and st_norm == a_st:
            results.append(a)
            continue

        # Match 3: Spatial distance match if coordinates supplied
        if lat is not None and lon is not None:
            a_lat = a.get("lat")
            a_lon = a.get("lon")
            if a_lat is not None and a_lon is not None:
                # Approximate euclidean distance for quick spatial filter (1 deg ~ 111 km)
                d_lat = (a_lat - lat) * 111.0
                d_lon = (a_lon - lon) * 111.0 * 0.9  # cos(avg_lat)
                dist = (d_lat**2 + d_lon**2)**0.5
                if dist <= radius_km:
                    results.append(a)
                    continue

    return results


def get_active_alerts(
    hazard: Optional[str] = None,
    state: Optional[str] = None,
    district: Optional[str] = None,
    min_severity: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves currently active operational alerts with flexible filtering.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    sql = """
        SELECT * FROM alerts
        WHERE status IN ('NEW', 'ACTIVE', 'UPDATED', 'ESCALATED', 'DEESCALATED')
    """
    params = []

    if hazard:
        sql += " AND hazard = ?"
        params.append(hazard.lower())
    if state:
        sql += " AND LOWER(state) = ?"
        params.append(state.strip().lower())
    if district:
        sql += " AND LOWER(district) = ?"
        params.append(district.strip().lower())

    sql += " ORDER BY last_updated_at DESC LIMIT ?;"
    params.append(limit)

    cursor.execute(sql, tuple(params))
    rows = cursor.fetchall()
    
    sev_order = {"NONE": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3, "SEVERE": 4, "CRITICAL": 5}
    min_rank = sev_order.get((min_severity or "").upper(), 0)

    alerts = [_row_to_alert_dict(r) for r in rows]
    if min_rank > 0:
        alerts = [a for a in alerts if sev_order.get((a.get("severity") or "").upper(), 0) >= min_rank]
        
    return alerts


def get_all_alerts(
    hazard: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Lists alerts with status and hazard filters."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    sql = "SELECT * FROM alerts WHERE 1=1"
    params = []
    if hazard:
        sql += " AND hazard = ?"
        params.append(hazard.lower())
    if status:
        sql += " AND status = ?"
        params.append(status.upper())
    sql += " ORDER BY last_updated_at DESC LIMIT ?;"
    params.append(limit)
    cursor.execute(sql, tuple(params))
    return [_row_to_alert_dict(r) for r in cursor.fetchall()]


# =========================================================================
# CRUD OPERATIONS: ALERT EVENTS HISTORY
# =========================================================================

def record_alert_event(
    alert_id: str,
    event_type: str,
    new_severity: str,
    old_severity: Optional[str] = None,
    evidence_snapshot: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
    timestamp: Optional[str] = None,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Records an immutable lifecycle event transition in alert_events."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    ts = timestamp or get_now_utc_iso()
    ev_json = json.dumps(evidence_snapshot or {})
    
    cursor.execute("""
        INSERT INTO alert_events (
            alert_id, event_type, old_severity, new_severity,
            timestamp, evidence_snapshot, reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
    """, (alert_id, event_type.upper(), old_severity, new_severity.upper(), ts, ev_json, reason))
    conn.commit()
    
    return {
        "event_id": cursor.lastrowid,
        "alert_id": alert_id,
        "event_type": event_type.upper(),
        "old_severity": old_severity,
        "new_severity": new_severity.upper(),
        "timestamp": ts,
        "evidence_snapshot": evidence_snapshot or {},
        "reason": reason or "",
    }


def get_alert_events(alert_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves full chronological history of events for a specific alert."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM alert_events
        WHERE alert_id = ?
        ORDER BY timestamp ASC, event_id ASC;
    """, (alert_id,))
    return [_row_to_event_dict(r) for r in cursor.fetchall()]


def get_recent_alert_events(
    limit: int = 50,
    hazard: Optional[str] = None,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves recent alert lifecycle events across the platform with alert metadata."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    sql = """
        SELECT e.*, a.hazard, a.title, a.state, a.district, a.lat, a.lon
        FROM alert_events e
        JOIN alerts a ON e.alert_id = a.alert_id
    """
    params = []
    if hazard:
        sql += " WHERE a.hazard = ?"
        params.append(hazard.lower())
    sql += " ORDER BY e.timestamp DESC LIMIT ?;"
    params.append(limit)
    cursor.execute(sql, tuple(params))
    
    out = []
    for r in cursor.fetchall():
        d = _row_to_event_dict(r)
        out.append(d)
    return out


# =========================================================================
# CRUD OPERATIONS: SUBSCRIPTIONS
# =========================================================================

def save_subscription(sub: Dict[str, Any], db_path: Optional[str] = None) -> Dict[str, Any]:
    """Creates or updates a user/location alert subscription."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    sub_id = sub.get("subscription_id") or f"sub_{int(datetime.now().timestamp()*1000)}"
    user_id = sub.get("user_id") or "anonymous"
    loc_name = sub.get("location_name") or ""
    lat = sub.get("lat")
    lon = sub.get("lon")
    state = sub.get("state") or ""
    district = sub.get("district") or ""
    hazards = json.dumps(sub.get("hazards") or ["*"])
    min_sev = (sub.get("minimum_severity") or "MODERATE").upper()
    channels = json.dumps(sub.get("notification_channels") or ["IN_APP"])
    enabled = 1 if sub.get("enabled", True) else 0
    created_at = sub.get("created_at") or get_now_utc_iso()

    cursor.execute("""
        INSERT INTO alert_subscriptions (
            subscription_id, user_id, location_name, lat, lon, state, district,
            hazards, minimum_severity, notification_channels, enabled, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(subscription_id) DO UPDATE SET
            user_id = excluded.user_id,
            location_name = excluded.location_name,
            lat = excluded.lat,
            lon = excluded.lon,
            state = excluded.state,
            district = excluded.district,
            hazards = excluded.hazards,
            minimum_severity = excluded.minimum_severity,
            notification_channels = excluded.notification_channels,
            enabled = excluded.enabled;
    """, (sub_id, user_id, loc_name, lat, lon, state, district, hazards, min_sev, channels, enabled, created_at))
    conn.commit()
    
    return get_subscription(sub_id, db_path=db_path)


def get_subscription(sub_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single subscription by ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM alert_subscriptions WHERE subscription_id = ?;", (sub_id,))
    row = cursor.fetchone()
    return _row_to_sub_dict(row) if row else None


def get_subscriptions(
    user_id: Optional[str] = None,
    enabled_only: bool = True,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves subscriptions, optionally filtered by user_id and enabled flag."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    sql = "SELECT * FROM alert_subscriptions WHERE 1=1"
    params = []
    if user_id:
        sql += " AND user_id = ?"
        params.append(user_id)
    if enabled_only:
        sql += " AND enabled = 1"
    sql += " ORDER BY created_at DESC;"
    cursor.execute(sql, tuple(params))
    return [_row_to_sub_dict(r) for r in cursor.fetchall()]


def delete_subscription(sub_id: str, db_path: Optional[str] = None) -> bool:
    """Deletes a subscription by ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM alert_subscriptions WHERE subscription_id = ?;", (sub_id,))
    conn.commit()
    return cursor.rowcount > 0


# =========================================================================
# CRUD OPERATIONS: IN-APP NOTIFICATIONS
# =========================================================================

def save_in_app_notification(
    alert_id: str,
    event_type: str,
    title: str,
    body: str,
    severity: str,
    channel: str = "IN_APP",
    status: str = "DELIVERED",
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Stores an in-app notification entry."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    notif_id = f"notif_{int(datetime.now().timestamp()*1000)}_{abs(hash(title))%10000}"
    created_at = get_now_utc_iso()

    cursor.execute("""
        INSERT INTO in_app_notifications (
            notification_id, alert_id, event_type, channel,
            title, body, severity, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (notif_id, alert_id, event_type.upper(), channel.upper(), title, body, severity.upper(), status, created_at))
    conn.commit()

    return {
        "notification_id": notif_id,
        "alert_id": alert_id,
        "event_type": event_type.upper(),
        "channel": channel.upper(),
        "title": title,
        "body": body,
        "severity": severity.upper(),
        "status": status,
        "created_at": created_at,
    }


def get_in_app_notifications(limit: int = 50, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves recent in-app notifications."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM in_app_notifications
        ORDER BY created_at DESC
        LIMIT ?;
    """, (limit,))
    return [dict(r) for r in cursor.fetchall()]
