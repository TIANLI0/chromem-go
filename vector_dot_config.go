package chromem

import (
	"os"
	"strconv"
	"sync/atomic"
)

const defaultDotProductSIMDMinLength = 1536 // 1536 is a heuristic based on benchmarks, but can be tuned via the environment variable CHROMEM_SIMD_MIN_LENGTH or the SetSIMDMinLength function.

var dotProductSIMDMinLength atomic.Int64

func init() {
	minLen := defaultDotProductSIMDMinLength
	if value, ok := os.LookupEnv("CHROMEM_SIMD_MIN_LENGTH"); ok {
		if parsed, err := strconv.Atoi(value); err == nil && parsed >= 0 {
			minLen = parsed
		}
	}

	dotProductSIMDMinLength.Store(int64(minLen))
}

// SetSIMDMinLength sets the minimum vector length at which SIMD is used.
// Values < 0 are ignored.
func SetSIMDMinLength(minLen int) {
	if minLen < 0 {
		return
	}
	dotProductSIMDMinLength.Store(int64(minLen))
}

func getSIMDMinLength() int {
	return int(dotProductSIMDMinLength.Load())
}
