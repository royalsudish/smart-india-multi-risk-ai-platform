import requests
from backend.config.settings import APP_USER_AGENT, REQUEST_TIMEOUT_SECONDS

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
