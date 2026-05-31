# Third-Party Integrations Catalog

Configuration for every external API. Service docs reference these by ID.

## NorthPay
- **Provider:** card-charge payment gateway
- **Used by:** payments-svc
- **Contractual SLA (p99):** 1800 ms
- **Meridian-configured call timeout:** 2500 ms
- **Rate limit:** 200 req/s
- **Failure behavior:** returns 402 on declined card, 503 on provider overload.
- **Note:** Real p99 has been observed spiking to 2200 ms during peak hours,
  still inside NorthPay's own 2500 ms call timeout.

## PulseSMS
- **Provider:** SMS / push notification delivery
- **Used by:** notification-svc
- **Contractual SLA (p99):** 900 ms
- **Meridian-configured call timeout:** 1500 ms
- **Rate limit:** 100 messages/s
- **Failure behavior:** returns 429 when the rate limit is exceeded; messages are
  dropped, not queued.

## CartoMaps
- **Provider:** geocoding and routing
- **Used by:** geo-svc
- **Contractual SLA (p99):** 250 ms
- **Meridian-configured call timeout:** 800 ms
- **Rate limit:** 500 req/s
- **Failure behavior:** returns 429 when the rate limit is exceeded.
