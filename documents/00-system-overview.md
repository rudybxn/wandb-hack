# Meridian Rideshare — Platform Operations Wiki

Meridian is a fictional distributed rideshare platform. This wiki is the source of
truth for service ownership, dependencies, timeout budgets, and failure modes.
All names, IDs, and numbers below are internal to Meridian and are not shared with
any external system.

## How to read this wiki

- One markdown file per service: `svc-<name>.md`
- Shared infrastructure lives in catalog files:
  - `catalog-datastores.md` — every datastore by ID (engine, pool limits, latency, failure behavior)
  - `catalog-third-party.md` — every external API by ID (provider, SLA, configured timeout, rate limit)
  - `conventions-timeouts-retries.md` — platform-wide timeout and retry rules

Service docs reference datastores and third parties **by ID only**. The deep
configuration for an ID is always in the relevant catalog file, never duplicated
in the service doc.

## Service-doc schema

Every `svc-*.md` file uses exactly these fields:

```
# <Service Name> (<service-id>)
**Owner team:**          <team>
**Purpose:**             <one line>
**Upstream callers:**    <services that call this service>
**Downstream dependencies:** <services this service calls, with per-call timeout>
**Datastores:**          <datastore IDs, with engine inline; deep config in catalog>
**Third-party dependencies:** <external API IDs; config in catalog>
**Inbound request timeout:** <ms — request is abandoned past this>
**Retry policy:**        <override, or "platform default">
**Known failure modes:** <short list>
```

## Dependency graph (request flow)

```
[external client]
      │
      ▼
 edge-gateway ──► booking-svc ──► dispatch-svc ──► driver-location-svc ──► DS-GEO-3
      │                │              └──► matching-engine ──► driver-location-svc
      │                ├──► pricing-svc ──► surge-engine ──► DS-SURGE-CACHE
      │                │                 └──► geo-svc ──► CartoMaps
      │                └──► rider-profile-svc ──► DS-RIDER-1
      │
      ▼
 trip-ledger-svc ──► DS-LEDGER-2
      │
      ▼
 payments-svc ──► NorthPay,  DS-PAY-LEDGER
 notification-svc ──► PulseSMS
 demand-forecast-svc ──► DS-FEAT-1
```

## Services in this platform

Defined: edge-gateway, booking-svc, dispatch-svc, driver-location-svc,
pricing-svc, payments-svc, notification-svc.

To be added (same schema): matching-engine, surge-engine, geo-svc,
rider-profile-svc, driver-profile-svc, trip-ledger-svc, demand-forecast-svc.
