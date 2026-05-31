# Dispatch Service (dispatch-svc)
**Owner team:** Rides Core
**Purpose:** Matches a rider to the best available nearby driver.
**Upstream callers:** booking-svc
**Downstream dependencies:** driver-location-svc (350 ms), matching-engine (300 ms)
**Datastores:** none (orchestrates other services)
**Third-party dependencies:** none
**Inbound request timeout:** 600 ms
**Retry policy:** platform default
**Known failure modes:**
- Calls driver-location-svc to fetch candidate driver positions before matching.
  During surge, driver-location-svc slows sharply, dispatch breaches its 600 ms
  inbound timeout, and the stall propagates up through booking-svc to the gateway.
