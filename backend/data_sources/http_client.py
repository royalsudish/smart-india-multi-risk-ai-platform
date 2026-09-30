import socket
import requests
import urllib3.util.connection as urllib3_conn
from backend.config.settings import APP_USER_AGENT, REQUEST_TIMEOUT_SECONDS

# Force IPv4 resolution to prevent dual-stack IPv6 TLS handshake timeouts on systems with unrouted IPv6
def _allowed_gai_family():
    return socket.AF_INET

urllib3_conn.allowed_gai_family = _allowed_gai_family

def session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": APP_USER_AGENT,
        "Accept": "application/json,*/*"
    })
    return s

def get_json(url, params=None, timeout=None, headers=None):
    s = session()
    if headers:
        s.headers.update(headers)
    r = s.get(url, params=params, timeout=timeout or REQUEST_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()

def post_form_json(url, data=None, timeout=None, headers=None):
    s = session()
    if headers:
        s.headers.update(headers)
    r = s.post(url, data=data, timeout=timeout or REQUEST_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()
