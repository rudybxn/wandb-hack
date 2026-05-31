# Driver Location Service (driver-location-svc)
**Owner team:** Geo & Mapping
**Purpose:** Serves real-time driver GPS positions for matching and ETA.
**Upstream callers:** dispatch-svc, matching-engine
**Downstream dependencies:** none
**Datastores:** DS-GEO-3 (ScyllaDB)
**Third-party dependencies:** none
**Inbound request timeout:** 300 ms
**Retry policy:** platform default (GET reads are idempotent)
**Known failure modes:**
- This service is read-heavy and its latency is bounded by DS-GEO-3. Under surge,
  read concurrency rises and the datastore becomes the limiting factor. See the
  datastore catalog for DS-GEO-3's pool behavior — the connection-pool ceiling is
  the usual origin of dispatch-path latency cliffs.
