# Notification Service (notification-svc)
**Owner team:** Engagement
**Purpose:** Sends push and SMS messages (ride confirmations, driver-arrival alerts) to riders and drivers.
**Upstream callers:** booking-svc, dispatch-svc, trip-ledger-svc
**Downstream dependencies:** none internal
**Datastores:** none
**Third-party dependencies:** PulseSMS
**Inbound request timeout:** 1200 ms
**Retry policy:** platform default for delivery-status GETs; sends are not retried
**Known failure modes:**
- During large promotional blasts, outbound message volume can exceed the PulseSMS
  rate limit (see third-party catalog). Excess messages are rejected and dropped
  rather than queued, so some riders never receive an alert.
