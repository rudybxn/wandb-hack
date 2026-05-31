# Booking Service (booking-svc)
**Owner team:** Rides Core
**Purpose:** Creates and manages ride bookings from request through driver assignment.
**Upstream callers:** edge-gateway
**Downstream dependencies:** dispatch-svc (400 ms), pricing-svc (250 ms), rider-profile-svc (150 ms)
**Datastores:** none (stateless; trip state is persisted via trip-ledger-svc)
**Third-party dependencies:** none
**Inbound request timeout:** 900 ms
**Retry policy:** platform default
**Known failure modes:**
- A new booking blocks on dispatch-svc to assign a driver. If dispatch is slow,
  the booking request stalls and can breach the gateway's 1000 ms budget.
- Pricing and rider-profile calls run in parallel and are rarely the bottleneck.
