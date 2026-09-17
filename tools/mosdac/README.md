# MOSDAC MDAPI

The supplied URL points to the official downloadable `mdapi.zip`, not to a
simple public JSON REST endpoint.

Official package:

```text
https://www.mosdac.gov.in/software/mdapi.zip
```

Official manual:

```text
https://www.mosdac.gov.in/downloadapi-manual
```

The archive contains `mdapi.py` and `config.json`. MOSDAC credentials belong in
your local config/environment only. Do **not** commit usernames or passwords.

Recommended project flow:

1. Download and unzip the official package outside `git`.
2. Configure datasetId, time range and boundingBox in `config.json`.
3. Download required satellite products into a data directory.
4. Write a parser for the specific product you selected.
5. Convert that data into model features.

Because dataset/product IDs determine the parsing logic, this repo does not
pretend `mdapi.zip` is a generic unauthenticated REST API.
