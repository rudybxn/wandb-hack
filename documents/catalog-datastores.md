# Datastore Catalog

Deep configuration for every datastore. Service docs reference these by ID.

## DS-GEO-3
- **Engine:** ScyllaDB (Cassandra-compatible), 3-node cluster
- **Connection pool max:** 40 concurrent connections per service instance
- **p99 read latency (nominal):** 35 ms
- **p99 read latency (pool saturated):** 600+ ms
- **Backs:** driver-location-svc
- **Failure mode:** Under surge, concurrent reads exceed the 40-connection pool.
  Excess reads queue, and p99 climbs past 600 ms — a latency cliff, not a gradual
  slope. This is the most common origin of cascading timeouts on the dispatch path.

## DS-LEDGER-2
- **Engine:** PostgreSQL 15, primary + 1 read replica
- **Connection pool max:** 100
- **p99 write latency:** 20 ms
- **Backs:** trip-ledger-svc
- **Failure mode:** Replica lag up to 4 s during bulk backfills.

## DS-RIDER-1
- **Engine:** PostgreSQL 15
- **Connection pool max:** 80
- **p99 read latency:** 15 ms
- **Backs:** rider-profile-svc

## DS-DRIVER-1
- **Engine:** PostgreSQL 15
- **Connection pool max:** 80
- **p99 read latency:** 15 ms
- **Backs:** driver-profile-svc

## DS-SURGE-CACHE
- **Engine:** Redis
- **TTL:** 30 s on surge multipliers
- **Backs:** surge-engine
- **Failure mode:** On a cache miss storm, surge-engine recomputes synchronously,
  adding ~180 ms per request.

## DS-FEAT-1
- **Engine:** Redis feature store
- **TTL:** 90 s on demand features
- **Max memory:** 8 GB (LRU eviction)
- **Backs:** demand-forecast-svc
- **Failure mode:** If the refresh job lags, features go stale; forecasts silently
  degrade rather than error.

## DS-PAY-LEDGER
- **Engine:** PostgreSQL 15
- **Connection pool max:** 60
- **p99 write latency:** 25 ms
- **Backs:** payments-svc
