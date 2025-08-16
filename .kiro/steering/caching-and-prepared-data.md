---
inclusion: fileMatch
fileMatchPattern: ['Corpus/qualia/*.java']
---
# Prepared Data Cache

- Key types and disk store:
  - [DatasetCacheKey.java](mdc:Corpus/qualia/DatasetCacheKey.java)
  - [DatasetPreparedDiskStore.java](mdc:Corpus/qualia/DatasetPreparedDiskStore.java)
- In-memory LRU-like cache:
  - [SimpleCaffeineLikeCache.java](mdc:Corpus/qualia/SimpleCaffeineLikeCache.java)
- Wiring in model precompute path:
  - [HierarchicalBayesianModel.java](mdc:Corpus/qualia/HierarchicalBayesianModel.java)
- Config knobs source:
  - [CacheConfig.java](mdc:Corpus/qualia/CacheConfig.java)
- Metrics export target:
  - [MetricsRegistry.java](mdc:Corpus/qualia/MetricsRegistry.java)

Ops notes
- Values must be immutable/defensively copied.
- Enable/disable disk layer and set TTL/weights via env/system properties (see `CacheConfig`).

