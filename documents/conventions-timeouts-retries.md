# Platform Timeout & Retry Conventions

These rules apply platform-wide unless a service doc overrides them.

## Inbound timeouts
- **edge-gateway global budget:** 1000 ms. If a request to the gateway is not
  fully served within 1000 ms, the gateway returns **HTTP 504 Gateway Timeout**.
- Every other service declares its own inbound timeout in its service doc.

## Inter-service call timeouts
- **Default downstream call timeout:** 300 ms, unless a service doc overrides the
  per-call timeout for a specific dependency.
- A service's inbound timeout should be **greater than the sum of its critical-path
  downstream call timeouts**. Violations are a known source of premature cutoffs.

## Retries
- **Platform default:** up to 2 retries, exponential backoff with a 50 ms base,
  **only on idempotent GETs**.
- Writes (POST/PUT) are never retried automatically; the caller must reconcile.

## Cascade rule of thumb
A latency spike at a leaf dependency propagates upward: the leaf breaches a
caller's per-call timeout, the caller breaches its own inbound timeout, and so on
up to the gateway's 1000 ms budget, which surfaces to the client as a 504.
