import os

# Admin settings are validated at import time and OIDC is enabled by default,
# so any test that imports `statgpt.admin.auth` would need a full OIDC configuration.
os.environ.setdefault("OIDC_AUTH_ENABLED", "false")
