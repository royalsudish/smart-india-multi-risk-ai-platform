from backend.config.settings import MOSDAC_MDAPI_ZIP

def info():
    return {
        "mdapi_zip": MOSDAC_MDAPI_ZIP,
        "manual": "https://www.mosdac.gov.in/downloadapi-manual",
        "integration_type": "downloadable Python utility",
        "requires_credentials": True,
        "expected_files_after_unzip": ["mdapi.py", "config.json"],
        "note": (
            "MOSDAC MDAPI is not treated here as a normal unauthenticated REST endpoint. "
            "Download the official mdapi.zip, configure credentials locally and never commit them."
        )
    }
