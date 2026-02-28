package chromem

import (
	"math"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
)

const (
	defaultHNSWEnabled                    = true
	defaultHNSWM                          = 16
	defaultHNSWEFConstruction             = 200
	defaultHNSWEFSearch                   = 200
	defaultHNSWExactRerankTopN            = 0
	defaultHNSWTombstoneRebuildRatio      = 0.2
	defaultHNSWTombstoneRebuildMinDeleted = 2048
)

// HNSW tuning parameters are stored in atomic variables to allow dynamic updates at runtime without locking.
var hnswEnabledFlag atomic.Bool

// HNSW parameters:
// M controls the maximum number of connections per node per layer. Higher M can improve recall but increases index size and build time.
// EFConstruction controls the size of the candidate list during index construction. Higher values can improve recall but increase build time and memory usage.
// EFSearch controls the size of the candidate list during search. Higher values can improve recall but increase search latency.
// TombstoneRebuildRatio sets the ratio of deleted nodes to total nodes that triggers a graph compaction rebuild when exceeded.
// TombstoneRebuildMinDeleted sets the minimum number of deleted nodes required before ratio-based compaction can trigger.
var hnswMValue atomic.Int64
var hnswEFConstructionValue atomic.Int64
var hnswEFSearchValue atomic.Int64
var hnswExactRerankTopNValue atomic.Int64
var hnswTombstoneRebuildRatioBits atomic.Uint64
var hnswTombstoneRebuildMinDeletedValue atomic.Int64

// init reads HNSW tuning parameters from environment variables or uses defaults if not set or invalid.
func init() {
	hnswEnabledFlag.Store(readHNSWBoolEnvOrDefault("CHROMEM_HNSW_ENABLED", defaultHNSWEnabled))
	hnswMValue.Store(int64(readHNSWPositiveEnvOrDefault("CHROMEM_HNSW_M", defaultHNSWM)))
	hnswEFConstructionValue.Store(int64(readHNSWPositiveEnvOrDefault("CHROMEM_HNSW_EF_CONSTRUCTION", defaultHNSWEFConstruction)))
	hnswEFSearchValue.Store(int64(readHNSWPositiveEnvOrDefault("CHROMEM_HNSW_EF_SEARCH", defaultHNSWEFSearch)))
	hnswExactRerankTopNValue.Store(int64(readHNSWNonNegativeEnvOrDefault("CHROMEM_HNSW_EXACT_RERANK_TOPN", defaultHNSWExactRerankTopN)))
	hnswTombstoneRebuildRatioBits.Store(math.Float64bits(readHNSWFloatRangeEnvOrDefault("CHROMEM_HNSW_TOMBSTONE_REBUILD_RATIO", defaultHNSWTombstoneRebuildRatio, 0, 1)))
	hnswTombstoneRebuildMinDeletedValue.Store(int64(readHNSWNonNegativeEnvOrDefault("CHROMEM_HNSW_TOMBSTONE_REBUILD_MIN_DELETED", defaultHNSWTombstoneRebuildMinDeleted)))
}

// HNSW tuning parameter getters and setters allow dynamic updates at runtime. Setters validate inputs and ignore invalid values, while getters return the current effective values.
func readHNSWBoolEnvOrDefault(name string, defaultValue bool) bool {
	value, ok := os.LookupEnv(name)
	if !ok {
		return defaultValue
	}

	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return defaultValue
	}
}

func SetHNSWEnabled(enabled bool) {
	hnswEnabledFlag.Store(enabled)
}

func SetHNSWM(m int) {
	if m < 2 {
		return
	}
	hnswMValue.Store(int64(m))
}

func SetHNSWEFConstruction(ef int) {
	if ef < 1 {
		return
	}
	hnswEFConstructionValue.Store(int64(ef))
}

func SetHNSWEFSearch(ef int) {
	if ef < 1 {
		return
	}
	hnswEFSearchValue.Store(int64(ef))
}

// SetHNSWExactRerankTopN sets two-stage query reranking candidate count.
//
//	0  disables exact reranking.
//	>0 enables exact reranking on up to top-N ANN candidates.
func SetHNSWExactRerankTopN(topN int) {
	if topN < 0 {
		return
	}
	hnswExactRerankTopNValue.Store(int64(topN))
}

// SetHNSWTombstoneRebuildRatio sets the deleted-node ratio threshold that triggers
// a graph compaction rebuild. Values outside [0,1] are ignored.
//
//	0   disables ratio-based compaction trigger.
//	>0  enables trigger when deleted/total >= ratio and min deleted threshold is met.
func SetHNSWTombstoneRebuildRatio(ratio float64) {
	if ratio < 0 || ratio > 1 {
		return
	}
	hnswTombstoneRebuildRatioBits.Store(math.Float64bits(ratio))
}

// SetHNSWTombstoneRebuildMinDeleted sets the minimum number of deleted (tombstoned)
// nodes required before ratio-based compaction can trigger. Values < 0 are ignored.
func SetHNSWTombstoneRebuildMinDeleted(minDeleted int) {
	if minDeleted < 0 {
		return
	}
	hnswTombstoneRebuildMinDeletedValue.Store(int64(minDeleted))
}

func getHNSWEnabled() bool {
	return hnswEnabledFlag.Load()
}

func getHNSWM() int {
	return int(hnswMValue.Load())
}

func getHNSWEFConstruction() int {
	return int(hnswEFConstructionValue.Load())
}

func getHNSWEFSearch() int {
	return int(hnswEFSearchValue.Load())
}

func getHNSWExactRerankTopN() int {
	return int(hnswExactRerankTopNValue.Load())
}

func getHNSWTombstoneRebuildRatio() float64 {
	return math.Float64frombits(hnswTombstoneRebuildRatioBits.Load())
}

func getHNSWTombstoneRebuildMinDeleted() int {
	return int(hnswTombstoneRebuildMinDeletedValue.Load())
}

func readHNSWPositiveEnvOrDefault(name string, defaultValue int) int {
	value, ok := os.LookupEnv(name)
	if !ok {
		return defaultValue
	}

	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 1 {
		return defaultValue
	}

	return parsed
}

func readHNSWNonNegativeEnvOrDefault(name string, defaultValue int) int {
	value, ok := os.LookupEnv(name)
	if !ok {
		return defaultValue
	}

	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 0 {
		return defaultValue
	}

	return parsed
}

func readHNSWFloatRangeEnvOrDefault(name string, defaultValue float64, minValue, maxValue float64) float64 {
	value, ok := os.LookupEnv(name)
	if !ok {
		return defaultValue
	}

	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil || parsed < minValue || parsed > maxValue {
		return defaultValue
	}

	return parsed
}
