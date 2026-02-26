package chromem

import (
	"os"
	"strconv"
	"sync/atomic"
)

const (
	defaultQuerySmallDocsThreshold        = 2048
	defaultQuerySequentialDocsThreshold   = 512
	defaultQueryHighDimThreshold          = 2048
	defaultQueryHighDimConcurrencyDivisor = 2
)

var querySmallDocsThreshold atomic.Int64
var querySequentialDocsThreshold atomic.Int64
var queryHighDimThreshold atomic.Int64
var queryHighDimConcurrencyDivisor atomic.Int64

func init() {
	querySmallDocsThreshold.Store(int64(readNonNegativeEnvOrDefault("CHROMEM_QUERY_SMALL_DOCS_THRESHOLD", defaultQuerySmallDocsThreshold)))
	querySequentialDocsThreshold.Store(int64(readNonNegativeEnvOrDefault("CHROMEM_QUERY_SEQUENTIAL_DOCS_THRESHOLD", defaultQuerySequentialDocsThreshold)))
	queryHighDimThreshold.Store(int64(readNonNegativeEnvOrDefault("CHROMEM_QUERY_HIGH_DIM_THRESHOLD", defaultQueryHighDimThreshold)))
	queryHighDimConcurrencyDivisor.Store(int64(readPositiveEnvOrDefault("CHROMEM_QUERY_HIGH_DIM_CONCURRENCY_DIVISOR", defaultQueryHighDimConcurrencyDivisor)))
}

func readNonNegativeEnvOrDefault(name string, defaultValue int) int {
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

func readPositiveEnvOrDefault(name string, defaultValue int) int {
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

// SetQuerySmallDocsThreshold sets the docs threshold below which query workers
// scale to runtime.NumCPU(). Values < 0 are ignored.
func SetQuerySmallDocsThreshold(threshold int) {
	if threshold < 0 {
		return
	}
	querySmallDocsThreshold.Store(int64(threshold))
}

// SetQuerySequentialDocsThreshold sets the docs threshold below which query and
// filter paths run sequentially. Values < 0 are ignored.
func SetQuerySequentialDocsThreshold(threshold int) {
	if threshold < 0 {
		return
	}
	querySequentialDocsThreshold.Store(int64(threshold))
}

// SetQueryHighDimThreshold sets the embedding dimension threshold above which
// query concurrency is reduced. Values < 0 are ignored.
func SetQueryHighDimThreshold(threshold int) {
	if threshold < 0 {
		return
	}
	queryHighDimThreshold.Store(int64(threshold))
}

// SetQueryHighDimConcurrencyDivisor sets the divisor that reduces query
// concurrency for high-dimensional embeddings. Values < 1 are ignored.
func SetQueryHighDimConcurrencyDivisor(divisor int) {
	if divisor < 1 {
		return
	}
	queryHighDimConcurrencyDivisor.Store(int64(divisor))
}

func getQuerySmallDocsThreshold() int {
	return int(querySmallDocsThreshold.Load())
}

func getQuerySequentialDocsThreshold() int {
	return int(querySequentialDocsThreshold.Load())
}

func getQueryHighDimThreshold() int {
	return int(queryHighDimThreshold.Load())
}

func getQueryHighDimConcurrencyDivisor() int {
	return int(queryHighDimConcurrencyDivisor.Load())
}
