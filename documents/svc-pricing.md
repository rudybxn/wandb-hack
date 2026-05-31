# Pricing Service (pricing-svc)
**Owner team:** Marketplace
**Purpose:** Computes the fare estimate and final price for a trip.
**Upstream callers:** edge-gateway, booking-svc
**Downstream dependencies:** surge-engine (200 ms), geo-svc (250 ms)
**Datastores:** none
**Third-party dependencies:** none
**Inbound request timeout:** 500 ms
**Retry policy:** platform default
**Known failure modes:**
- If surge-engine returns a stale multiplier, quoted fares can drift from the
  charged amount; this is a correctness issue, not a latency one.
