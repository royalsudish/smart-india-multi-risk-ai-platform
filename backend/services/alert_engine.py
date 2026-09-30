"""
Smart Multi-Risk Platform: Smart Alert Engine
Authoritative operational alert lifecycle manager and event orchestrator.

Responsibilities:
1. Compare incoming verified operational hazards with persistent active alerts.
2. Maintain strict alert identity and deduplication via deterministic fingerprinting.
3. Manage lifecycle state machine: NEW -> ACTIVE -> ESCALATED / DEESCALATED -> RESOLVED / EXPIRED.
4. Record immutable timeline transitions in AlertEvent history.
5. Track data freshness and provider availability (never mark safe on provider failure).
6. Trigger event-driven notifications without generating spam on unchanged heartbeats.
7. Support multi-hazard concurrency across India.
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from backend.data_sources.alert_repository import (
    save_alert,
    get_alert_by_fingerprint,
    get_open_alerts_for_location,
    record_alert_event,
    get_now_utc_iso,
    get_active_alerts,
)
from backend.services.notification_service import NotificationService, default_notification_service

# Severity Ranking per Specification:
# NONE < LOW < MODERATE < HIGH < SEVERE < CRITICAL
SEVERITY_RANKS = {
    "NONE": 0,
    "LOW": 1,
    "MODERATE": 2,
    "HIGH": 3,
    "SEVERE": 4,
    "CRITICAL": 5,
}

# Operational Default TTLs (hours)
DEFAULT_TTL_HOURS = {
    "flash_flood": 4,
    "flood": 24,
    "thunderstorm": 3,
    "high_wind": 4,
    "earthquake": 6,
    "wildfire": 12,
    "heat_wave": 18,
    "cold_wave": 18,
    "cyclone": 24,
    "avalanche": 12,
    "tsunami": 12,
    "volcano": 48,
    "air_pollution": 12,
    "landslide": 12,
    "drought": 72,
}

# Mapping of hazards to their primary data providers for data-quality tracking
HAZARD_PROVIDER_DEPENDENCIES = {
    "wildfire": ["NASA_FIRMS", "FIRMS"],
    "avalanche": ["NDMA_SACHET", "SACHET", "DGRE"],
    "earthquake": ["USGS"],
    "cyclone": ["GDACS"],
    "volcano": ["GDACS"],
    "tsunami": ["USGS", "GDACS"],
}


def build_alert_fingerprint(
    hazard: str,
    state: Optional[str] = None,
    district: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    provider_event_id: Optional[str] = None,
) -> str:
    """
    Constructs a deterministic, stable fingerprint for alert deduplication.
    Prefers provider_event_id when available (USGS ID, SACHET CAP identifier, etc.).
    Otherwise uses normalized state and district, falling back to spatial grid (~16 km).
    """
    hz = hazard.strip().lower()
    
    # Priority 1: Official authoritative provider event ID
    if provider_event_id and str(provider_event_id).strip():
        clean_eid = str(provider_event_id).strip().replace(" ", "_")
        return f"{hz}:provider:{clean_eid}"
    
    # Priority 2: Geographic administrative area (State + District)
    st = (state or "").strip().lower().replace(" ", "_")
    dt = (district or "").strip().lower().replace(" ", "_")
    if dt:
        return f"{hz}:{st}:{dt}"
    if st:
        return f"{hz}:{st}:regional"
    
    # Priority 3: Coarse spatial grid (0.15 deg ~ 16 km)
    if lat is not None and lon is not None:
        grid_lat = round(lat / 0.15) * 0.15
        grid_lon = round(lon / 0.15) * 0.15
        return f"{hz}:grid:{grid_lat:.2f}:{grid_lon:.2f}"
    
    return f"{hz}:national_default"


def extract_provider_event_id(hazard: str, hazard_item: Dict[str, Any], live_evidence: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Extracts official provider event identifier from hazard metadata or live telemetry."""
    evidence = hazard_item.get("evidence") or {}
    live_ev = live_evidence or {}

    # 1. SACHET CAP alert identifier
    if hazard == "avalanche":
        if evidence.get("identifier"):
            return str(evidence["identifier"])
        sachet_alerts = live_ev.get("sachet_alerts") or []
        if sachet_alerts and isinstance(sachet_alerts, list) and sachet_alerts[0].get("identifier"):
            return str(sachet_alerts[0]["identifier"])

    # 2. USGS Earthquake event ID
    elif hazard == "earthquake":
        if evidence.get("event_id"):
            return str(evidence["event_id"])
        usgs_q = live_ev.get("usgs_quake") or {}
        if usgs_q.get("event_id"):
            return str(usgs_q["event_id"])

    # 3. GDACS Cyclone / Volcano event ID
    elif hazard in ("cyclone", "volcano"):
        if evidence.get("event_id") or evidence.get("eventid"):
            return str(evidence.get("event_id") or evidence.get("eventid"))
        gdacs_list = live_ev.get("gdacs") or []
        if isinstance(gdacs_list, list) and gdacs_list and gdacs_list[0].get("eventid"):
            return str(gdacs_list[0]["eventid"])

    # 4. CWC River Basin bulletin identifier
    elif hazard == "flood":
        if evidence.get("bulletin_id") or evidence.get("cwc_station"):
            return str(evidence.get("bulletin_id") or evidence.get("cwc_station"))

    return None


def calculate_expiry_iso(hazard: str, hazard_item: Dict[str, Any], now_dt: datetime) -> str:
    """Calculates alert expiry timestamp based on provider metadata or operational TTL."""
    evidence = hazard_item.get("evidence") or {}
    
    # Official provider expiry (e.g. SACHET CAP expires)
    if evidence.get("expires"):
        try:
            exp_str = evidence["expires"]
            dt_exp = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
            return dt_exp.astimezone(timezone.utc).isoformat()
        except Exception:
            pass

    # Operational TTL fallback
    ttl_hours = DEFAULT_TTL_HOURS.get(hazard.lower(), 12)
    return (now_dt + timedelta(hours=ttl_hours)).isoformat()


class SmartAlertEngine:
    """
    Operational Alert Engine responsible for alert lifecycle transitions,
    deduplication, event history tracking, and notification triggering.
    """

    def __init__(self, db_path: Optional[str] = None, notification_service: Optional[NotificationService] = None):
        self.db_path = db_path
        self.notifier = notification_service if notification_service is not None else NotificationService(db_path=db_path)

    def process_operational_hazards(
        self,
        verified_hazards: List[Dict[str, Any]],
        location: Dict[str, Any],
        live_evidence: Optional[Dict[str, Any]] = None,
        data_quality: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Core operational ingestion method.
        Evaluates current verified hazards against persistent active alerts.
        """
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()

        lat = float(location.get("lat") or 0.0)
        lon = float(location.get("lon") or 0.0)
        state = location.get("state") or ""
        district = location.get("district") or ""
        live_evidence = live_evidence or {}
        data_quality = data_quality or {}
        missing_providers = [p.upper() for p in (data_quality.get("missing_providers") or [])]

        created_alerts = []
        updated_alerts = []
        resolved_alerts = []
        expired_alerts = []
        active_alerts = []

        # 1. Fetch all currently open alerts for this location
        existing_open_alerts = get_open_alerts_for_location(
            state=state,
            district=district,
            lat=lat,
            lon=lon,
            radius_km=35.0,
            db_path=self.db_path
        )
        
        # Index existing alerts by fingerprint
        open_by_fingerprint = {a["fingerprint"]: a for a in existing_open_alerts}
        open_by_hazard = {a["hazard"]: a for a in existing_open_alerts}

        # Track which existing alerts are corroborated in current cycle
        seen_alert_ids = set()

        # 2. Process each currently verified active hazard
        for vh in verified_hazards:
            hazard_name = (vh.get("hazard") or "").strip().lower()
            if not hazard_name or not vh.get("active", True):
                continue
            
            new_severity = (vh.get("severity") or "LOW").upper()
            provider_event_id = extract_provider_event_id(hazard_name, vh, live_evidence)
            fingerprint = build_alert_fingerprint(
                hazard=hazard_name,
                state=state,
                district=district,
                lat=lat,
                lon=lon,
                provider_event_id=provider_event_id
            )

            # Check if this hazard maps to an existing open alert
            existing = open_by_fingerprint.get(fingerprint) or open_by_hazard.get(hazard_name)

            if not existing:
                # -----------------------------------------------------------------
                # STATE TRANSITION: NEW ALERT CREATED
                # -----------------------------------------------------------------
                alert_id = f"alt_{uuid.uuid4().hex[:12]}"
                title = f"{hazard_name.upper().replace('_', ' ')} Warning - {district or state or 'Regional'}"
                description = vh.get("reason") or f"Active {hazard_name} operational hazard detected."
                expires_iso = calculate_expiry_iso(hazard_name, vh, now_utc)

                alert_record = {
                    "alert_id": alert_id,
                    "fingerprint": fingerprint,
                    "hazard": hazard_name,
                    "status": "NEW",
                    "severity": new_severity,
                    "title": title,
                    "description": description,
                    "lat": lat,
                    "lon": lon,
                    "state": state,
                    "district": district,
                    "sources": vh.get("sources") or [],
                    "evidence": vh.get("evidence") or {},
                    "confidence_type": vh.get("confidence_type") or "multi_source_evidence",
                    "started_at": now_iso,
                    "last_updated_at": now_iso,
                    "last_verified_at": now_iso,
                    "resolved_at": None,
                    "expires_at": expires_iso,
                    "provider_event_id": provider_event_id,
                    "data_quality": data_quality,
                    "notification_state": {"last_notified_severity": new_severity, "last_notified_at": now_iso},
                    "last_evidence_at": now_iso,
                    "provider_timestamp": now_iso,
                    "freshness_status": "FRESH",
                }

                saved = save_alert(alert_record, db_path=self.db_path)
                
                # Record CREATED event in history
                record_alert_event(
                    alert_id=alert_id,
                    event_type="CREATED",
                    new_severity=new_severity,
                    old_severity=None,
                    evidence_snapshot=vh.get("evidence"),
                    reason=description,
                    timestamp=now_iso,
                    db_path=self.db_path
                )

                # Trigger Notification for NEW alert
                self.notifier.dispatch(
                    alert=saved,
                    event_type="NEW",
                    old_severity=None
                )

                created_alerts.append(saved)
                active_alerts.append(saved)
                seen_alert_ids.add(alert_id)

            else:
                # -----------------------------------------------------------------
                # EXISTING ALERT UPDATED / ESCALATED / DEESCALATED / ACTIVE HEARTBEAT
                # -----------------------------------------------------------------
                alert_id = existing["alert_id"]
                seen_alert_ids.add(alert_id)
                old_severity = (existing.get("severity") or "LOW").upper()
                old_rank = SEVERITY_RANKS.get(old_severity, 0)
                new_rank = SEVERITY_RANKS.get(new_severity, 0)

                event_type = None
                if new_rank > old_rank:
                    event_type = "ESCALATED"
                    status = "ESCALATED"
                elif new_rank < old_rank:
                    event_type = "DEESCALATED"
                    status = "DEESCALATED"
                else:
                    # Same severity: Heartbeat or evidence update
                    status = "ACTIVE"
                    # Check if evidence snapshot changed significantly
                    old_ev = existing.get("evidence") or {}
                    new_ev = vh.get("evidence") or {}
                    if str(old_ev) != str(new_ev):
                        event_type = "UPDATED"

                # Update existing record
                existing["status"] = status
                existing["severity"] = new_severity
                existing["last_updated_at"] = now_iso
                existing["last_verified_at"] = now_iso
                existing["last_evidence_at"] = now_iso
                existing["evidence"] = vh.get("evidence") or {}
                existing["sources"] = vh.get("sources") or existing.get("sources") or []
                existing["description"] = vh.get("reason") or existing.get("description")
                existing["data_quality"] = data_quality
                existing["freshness_status"] = "FRESH"
                if provider_event_id:
                    existing["provider_event_id"] = provider_event_id

                saved = save_alert(existing, db_path=self.db_path)

                # Record event in history if lifecycle changed
                if event_type:
                    record_alert_event(
                        alert_id=alert_id,
                        event_type=event_type,
                        new_severity=new_severity,
                        old_severity=old_severity,
                        evidence_snapshot=vh.get("evidence"),
                        reason=vh.get("reason") or f"Severity updated to {new_severity}",
                        timestamp=now_iso,
                        db_path=self.db_path
                    )

                    # Dispatch event-driven notification (NEW / ESCALATED / DEESCALATED)
                    self.notifier.dispatch(
                        alert=saved,
                        event_type=event_type,
                        old_severity=old_severity
                    )

                updated_alerts.append(saved)
                active_alerts.append(saved)

        # 3. Check existing open alerts that were NOT corroborated in verified_hazards
        for existing in existing_open_alerts:
            alert_id = existing["alert_id"]
            if alert_id in seen_alert_ids:
                continue

            hz = existing["hazard"].lower()
            
            # Provider Health Check:
            # If the authoritative provider for this hazard failed/timed out,
            # DO NOT automatically mark the hazard safe or resolved!
            required_providers = HAZARD_PROVIDER_DEPENDENCIES.get(hz, [])
            provider_down = any(rp in missing_providers for rp in required_providers)

            if provider_down:
                # Mark as DATA_STALE / INSUFFICIENT_DATA and leave open
                existing["freshness_status"] = "INSUFFICIENT_DATA"
                existing["last_updated_at"] = now_iso
                saved = save_alert(existing, db_path=self.db_path)
                active_alerts.append(saved)
                continue

            # Check if alert has expired according to official validity time
            expires_at = existing.get("expires_at")
            is_expired = False
            if expires_at:
                try:
                    dt_exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                    if dt_exp <= now_utc:
                        is_expired = True
                except Exception:
                    pass

            old_sev = existing.get("severity") or "LOW"

            if is_expired:
                # -----------------------------------------------------------------
                # STATE TRANSITION: EXPIRED
                # -----------------------------------------------------------------
                existing["status"] = "EXPIRED"
                existing["last_updated_at"] = now_iso
                existing["resolved_at"] = now_iso
                saved = save_alert(existing, db_path=self.db_path)

                record_alert_event(
                    alert_id=alert_id,
                    event_type="EXPIRED",
                    new_severity="NONE",
                    old_severity=old_sev,
                    evidence_snapshot={},
                    reason="Alert validity window has expired.",
                    timestamp=now_iso,
                    db_path=self.db_path
                )

                self.notifier.dispatch(alert=saved, event_type="EXPIRED", old_severity=old_sev)
                expired_alerts.append(saved)

            else:
                # -----------------------------------------------------------------
                # STATE TRANSITION: RESOLVED
                # -----------------------------------------------------------------
                existing["status"] = "RESOLVED"
                existing["severity"] = "NONE"
                existing["last_updated_at"] = now_iso
                existing["resolved_at"] = now_iso
                saved = save_alert(existing, db_path=self.db_path)

                record_alert_event(
                    alert_id=alert_id,
                    event_type="RESOLVED",
                    new_severity="NONE",
                    old_severity=old_sev,
                    evidence_snapshot={},
                    reason="Operational telemetry confirms hazard is no longer active.",
                    timestamp=now_iso,
                    db_path=self.db_path
                )

                self.notifier.dispatch(alert=saved, event_type="RESOLVED", old_severity=old_sev)
                resolved_alerts.append(saved)

        return {
            "created_alerts": created_alerts,
            "updated_alerts": updated_alerts,
            "resolved_alerts": resolved_alerts,
            "expired_alerts": expired_alerts,
            "active_alerts": active_alerts,
        }


# Module default instance
default_alert_engine = SmartAlertEngine()
