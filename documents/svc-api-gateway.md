# API Gateway (edge-gateway)
**Owner team:** Edge Platform
**Purpose:** Single external entry point; authenticates and routes client requests to internal services.
**Upstream callers:** external clients (mobile apps, partner integrations)
**Downstream dependencies:** booking-svc (900 ms), pricing-svc (500 ms), rider-profile-svc (200 ms)
**Datastores:** none
**Third-party dependencies:** none
**Inbound request timeout:** 1000 ms (platform global; returns HTTP 504 on breach)
**Retry policy:** platform default
**Known failure modes:**
- Returns 504 to clients whenever any downstream critical path exceeds the 1000 ms budget.
  The gateway is almost never the root cause — a 504 here points at a slow service *below* it.
