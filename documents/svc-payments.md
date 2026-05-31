# Payments Service (payments-svc)
**Owner team:** Money
**Purpose:** Charges riders for completed trips and records the transaction.
**Upstream callers:** trip-ledger-svc (on trip completion)
**Downstream dependencies:** none internal
**Datastores:** DS-PAY-LEDGER (PostgreSQL 15)
**Third-party dependencies:** NorthPay
**Inbound request timeout:** 2000 ms
**Retry policy:** charges are NOT auto-retried (non-idempotent); failures are reconciled by a sweeper job
**Known failure modes:**
- Intermittent timeouts during peak hours. The service charges cards by calling
  NorthPay; the NorthPay integration config is in the third-party catalog. When a
  charge runs long, the resulting transaction state can be left ambiguous because
  the inbound deadline and the downstream call timeout are not aligned.
