import os

# Admin authentication settings are validated at import time.
os.environ.setdefault("OIDC_AUTH_ENABLED", "false")
