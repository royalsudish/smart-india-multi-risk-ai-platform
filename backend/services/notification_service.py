"""
Smart Multi-Risk Platform: Notification Service
Multi-channel, event-driven alert notification dispatcher.

Supported Channels:
1. IN-APP: Always active, stores persistent notification records in the alert repository.
2. EMAIL: SMTP notification adapter; gracefully reports unavailable if SMTP credentials absent.
3. PUSH: FCM (Firebase Cloud Messaging) adapter; gracefully reports unavailable if FCM credentials absent.

Notification Policy:
- Dispatches on lifecycle events: NEW, ESCALATED, RESOLVED, and DEESCALATED.
- Suppresses repeated notifications on unchanged ACTIVE heartbeats.
- Timezone formatting: Converts UTC to Indian Standard Time (IST, UTC+05:30) for human display.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import requests

from backend.config.settings import (
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM, FCM_SERVER_KEY
)
from backend.data_sources.alert_repository import save_in_app_notification


IST_OFFSET = timedelta(hours=5, minutes=30)


def format_utc_to_ist_str(utc_iso: Optional[str]) -> str:
    """Converts an ISO-8601 UTC timestamp string to formatted IST string."""
    if not utc_iso:
        dt = datetime.now(timezone.utc)
    else:
        try:
            # Handle trailing 'Z' or standard offsets
            cleaned = utc_iso.replace("Z", "+00:00")
            dt = datetime.fromisoformat(cleaned)
        except Exception:
            dt = datetime.now(timezone.utc)
    
    ist_dt = dt.astimezone(timezone(IST_OFFSET))
    return ist_dt.strftime("%Y-%m-%d %H:%M IST")


def build_notification_content(
    alert: Dict[str, Any],
    event_type: str,
    old_severity: Optional[str] = None
) -> Dict[str, str]:
    """
    Constructs authoritative notification title and body according to project standard.
    """
    hazard = (alert.get("hazard") or "DISASTER").upper().replace("_", " ")
    severity = (alert.get("severity") or "LOW").upper()
    loc_parts = []
    if alert.get("district"):
        loc_parts.append(alert["district"])
    if alert.get("state"):
        loc_parts.append(alert["state"])
    location_str = ", ".join(loc_parts) or f"Coord ({alert.get('lat', 0.0):.2f}, {alert.get('lon', 0.0):.2f})"
    
    verified_time = format_utc_to_ist_str(alert.get("last_verified_at") or alert.get("last_updated_at"))
    sources = alert.get("sources") or ["Multi-Sensor Telemetry"]
    sources_str = "\n".join(f"- {s}" for s in sources)
    reason = alert.get("description") or alert.get("reason") or "Operational hazard criteria met."

    ev = event_type.upper()

    if ev == "CREATED" or ev == "NEW":
        title = f"{hazard} — {severity} [{location_str}]"
        body = (
            f"{hazard} — {severity}\n"
            f"{location_str}\n\n"
            f"{reason}\n\n"
            f"Sources:\n{sources_str}\n\n"
            f"Last verified:\n{verified_time}"
        )
    elif ev == "ESCALATED":
        title = f"ESCALATED: {hazard} — {severity} [{location_str}]"
        old_sev_str = f" from {old_severity.upper()}" if old_severity else ""
        body = (
            f"ALERT ESCALATED: {hazard} escalated{old_sev_str} to {severity}\n"
            f"{location_str}\n\n"
            f"{reason}\n\n"
            f"Sources:\n{sources_str}\n\n"
            f"Last verified:\n{verified_time}"
        )
    elif ev == "DEESCALATED":
        title = f"DE-ESCALATED: {hazard} — {severity} [{location_str}]"
        old_sev_str = f" from {old_severity.upper()}" if old_severity else ""
        body = (
            f"ALERT DE-ESCALATED: {hazard} de-escalated{old_sev_str} to {severity}\n"
            f"{location_str}\n\n"
            f"{reason}\n\n"
            f"Sources:\n{sources_str}\n\n"
            f"Last verified:\n{verified_time}"
        )
    elif ev == "RESOLVED":
        title = f"{hazard} RESOLVED [{location_str}]"
        body = (
            f"{hazard} RESOLVED\n"
            f"{location_str}\n\n"
            f"Operational evidence no longer meets active hazard thresholds.\n\n"
            f"Resolved at:\n{verified_time}"
        )
    elif ev == "EXPIRED":
        title = f"{hazard} EXPIRED [{location_str}]"
        body = (
            f"{hazard} EXPIRED\n"
            f"{location_str}\n\n"
            f"Official alert validity period has elapsed.\n\n"
            f"Expired at:\n{verified_time}"
        )
    else:
        title = f"{hazard} UPDATE — {severity} [{location_str}]"
        body = (
            f"{hazard} UPDATE — {severity}\n"
            f"{location_str}\n\n"
            f"{reason}\n\n"
            f"Last verified:\n{verified_time}"
        )

    return {"title": title, "body": body}


class NotificationService:
    """
    Central dispatcher for operational alert notifications.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def should_notify(self, event_type: str, old_severity: Optional[str], new_severity: str) -> bool:
        """
        Determines if the event transition qualifies for notification dispatch.
        Dispatches on NEW, ESCALATED, DEESCALATED, RESOLVED.
        Suppresses unchanged ACTIVE heartbeats.
        """
        ev = event_type.upper()
        if ev in ("NEW", "CREATED", "ESCALATED", "RESOLVED", "DEESCALATED", "EXPIRED"):
            return True
        if ev == "UPDATED":
            # Only notify if severity changed
            return old_severity is not None and old_severity.upper() != new_severity.upper()
        return False

    def dispatch(
        self,
        alert: Dict[str, Any],
        event_type: str,
        old_severity: Optional[str] = None,
        recipient_emails: Optional[List[str]] = None,
        push_tokens: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Dispatches notification across configured channels.
        Returns detailed delivery status per channel.
        """
        new_severity = alert.get("severity", "LOW")
        if not self.should_notify(event_type, old_severity, new_severity):
            return {
                "dispatched": False,
                "reason": f"Event type '{event_type}' with unchanged severity does not trigger dispatch."
            }

        content = build_notification_content(alert, event_type, old_severity)
        alert_id = alert["alert_id"]
        
        # 1. IN-APP Channel (Always active)
        in_app_record = save_in_app_notification(
            alert_id=alert_id,
            event_type=event_type,
            title=content["title"],
            body=content["body"],
            severity=new_severity,
            channel="IN_APP",
            status="DELIVERED",
            db_path=self.db_path
        )

        # 2. EMAIL Channel
        email_status = self._send_email(content["title"], content["body"], recipient_emails)

        # 3. PUSH Channel (FCM)
        push_status = self._send_push(content["title"], content["body"], push_tokens, alert)

        return {
            "dispatched": True,
            "event_type": event_type.upper(),
            "notification_title": content["title"],
            "channels": {
                "in_app": {"status": "DELIVERED", "notification_id": in_app_record["notification_id"]},
                "email": email_status,
                "push": push_status,
            }
        }

    def _send_email(self, title: str, body: str, recipient_emails: Optional[List[str]]) -> Dict[str, Any]:
        """Sends email via SMTP or gracefully reports unavailable credentials."""
        if not SMTP_HOST:
            return {"status": "UNAVAILABLE", "reason": "SMTP host not configured (SMTP_HOST is empty)."}
        
        recipients = [e for e in (recipient_emails or []) if e and "@" in e]
        if not recipients:
            return {"status": "SKIPPED", "reason": "No recipient email addresses provided."}

        try:
            msg = MIMEMultipart()
            msg["From"] = SMTP_FROM
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = title
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                if SMTP_USER and SMTP_PASSWORD:
                    server.starttls()
                    server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_FROM, recipients, msg.as_string())
            return {"status": "DELIVERED", "recipients_count": len(recipients)}
        except Exception as exc:
            return {"status": "FAILED", "error": str(exc)}

    def _send_push(
        self,
        title: str,
        body: str,
        push_tokens: Optional[List[str]],
        alert: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sends push notification via FCM or gracefully reports unavailable credentials."""
        if not FCM_SERVER_KEY:
            return {"status": "UNAVAILABLE", "reason": "FCM server key not configured (FCM_SERVER_KEY is empty)."}

        tokens = [t for t in (push_tokens or []) if t]
        if not tokens:
            return {"status": "SKIPPED", "reason": "No device push tokens registered."}

        try:
            headers = {
                "Authorization": f"key={FCM_SERVER_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "registration_ids": tokens,
                "notification": {"title": title, "body": body},
                "data": {
                    "alert_id": alert.get("alert_id"),
                    "hazard": alert.get("hazard"),
                    "severity": alert.get("severity"),
                }
            }
            resp = requests.post("https://fcm.googleapis.com/fcm/send", json=payload, headers=headers, timeout=5)
            if resp.status_code == 200:
                return {"status": "DELIVERED", "recipients_count": len(tokens)}
            return {"status": "FAILED", "http_status": resp.status_code, "response": resp.text}
        except Exception as exc:
            return {"status": "FAILED", "error": str(exc)}


# Module-level default notification service instance
default_notification_service = NotificationService()
