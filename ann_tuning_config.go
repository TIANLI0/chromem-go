package chromem

import (
	"os"
	"strconv"
	"strings"
	"sync/atomic"
)

const (
	annIndexTypeHNSW   = "hnsw"
	annIndexTypeIVF    = "ivf"
	annIndexTypePQ     = "pq"
	annIndexTypeIVFPQ  = "ivfpq"
	annIndexTypeBM25   = "bm25"
	annIndexTypeHybrid = "hybrid"
)

const (
	defaultANNIndexType = annIndexTypeHNSW

	defaultANNIVFNList  = 64
	defaultANNIVFNProbe = 8

	defaultANNPQM     = 8
	defaultANNPQNBits = 8

	defaultANNIVFPQNList  = 64
	defaultANNIVFPQNProbe = 8
	defaultANNIVFPQM      = 8
	defaultANNIVFPQNBits  = 8
)

var annIndexTypeValue atomic.Value
var annIVFNListValue atomic.Int64
var annIVFNProbeValue atomic.Int64
var annPQMValue atomic.Int64
var annPQNBitsValue atomic.Int64
var annIVFPQNListValue atomic.Int64
var annIVFPQNProbeValue atomic.Int64
var annIVFPQMValue atomic.Int64
var annIVFPQNBitsValue atomic.Int64

func init() {
	annIndexTypeValue.Store(readANNIndexTypeEnvOrDefault("CHROMEM_INDEX_TYPE", defaultANNIndexType))
	annIVFNListValue.Store(int64(readANNPositiveEnvOrDefault("CHROMEM_IVF_NLIST", defaultANNIVFNList)))
	annIVFNProbeValue.Store(int64(readANNPositiveEnvOrDefault("CHROMEM_IVF_NPROBE", defaultANNIVFNProbe)))
	annPQMValue.Store(int64(readANNPositiveEnvOrDefault("CHROMEM_PQ_M", defaultANNPQM)))
	annPQNBitsValue.Store(int64(readANNRangeEnvOrDefault("CHROMEM_PQ_NBITS", defaultANNPQNBits, 1, 8)))
	annIVFPQNListValue.Store(int64(readANNPositiveEnvOrDefault("CHROMEM_IVFPQ_NLIST", defaultANNIVFPQNList)))
	annIVFPQNProbeValue.Store(int64(readANNPositiveEnvOrDefault("CHROMEM_IVFPQ_NPROBE", defaultANNIVFPQNProbe)))
	annIVFPQMValue.Store(int64(readANNPositiveEnvOrDefault("CHROMEM_IVFPQ_M", defaultANNIVFPQM)))
	annIVFPQNBitsValue.Store(int64(readANNRangeEnvOrDefault("CHROMEM_IVFPQ_NBITS", defaultANNIVFPQNBits, 1, 8)))
}

func getANNIndexType() string {
	value, ok := annIndexTypeValue.Load().(string)
	if !ok || value == "" {
		return defaultANNIndexType
	}
	return value
}

func getANNIVFNList() int {
	return int(annIVFNListValue.Load())
}

func getANNIVFNProbe() int {
	return int(annIVFNProbeValue.Load())
}

func getANNPQM() int {
	return int(annPQMValue.Load())
}

func getANNPQNBits() int {
	return int(annPQNBitsValue.Load())
}

func getANNIVFPQNList() int {
	return int(annIVFPQNListValue.Load())
}

func getANNIVFPQNProbe() int {
	return int(annIVFPQNProbeValue.Load())
}

func getANNIVFPQM() int {
	return int(annIVFPQMValue.Load())
}

func getANNIVFPQNBits() int {
	return int(annIVFPQNBitsValue.Load())
}

func readANNIndexTypeEnvOrDefault(name, defaultValue string) string {
	value, ok := os.LookupEnv(name)
	if !ok {
		return defaultValue
	}
	normalized := strings.ToLower(strings.TrimSpace(value))
	switch normalized {
	case annIndexTypeHNSW, annIndexTypeIVF, annIndexTypePQ, annIndexTypeIVFPQ, annIndexTypeBM25, annIndexTypeHybrid:
		return normalized
	default:
		return defaultValue
	}
}

func readANNPositiveEnvOrDefault(name string, defaultValue int) int {
	value, ok := os.LookupEnv(name)
	if !ok {
		return defaultValue
	}
	parsed, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || parsed <= 0 {
		return defaultValue
	}
	return parsed
}

func readANNRangeEnvOrDefault(name string, defaultValue, minValue, maxValue int) int {
	value, ok := os.LookupEnv(name)
	if !ok {
		return defaultValue
	}
	parsed, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || parsed < minValue || parsed > maxValue {
		return defaultValue
	}
	return parsed
}
