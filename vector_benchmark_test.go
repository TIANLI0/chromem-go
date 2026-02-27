package chromem

import (
	"fmt"
	"testing"
)

var benchmarkDotProductSink float32
var benchmarkDotProductPairSinkA float32
var benchmarkDotProductPairSinkB float32

func benchmarkVectors(size int) ([]float32, []float32) {
	a := make([]float32, size)
	b := make([]float32, size)

	for i := range size {
		a[i] = float32((i%97)+1) / 97
		b[i] = float32((i%89)+1) / 89
	}

	return a, b
}

func BenchmarkDotProduct(b *testing.B) {
	oldMinLen := getSIMDMinLength()
	SetSIMDMinLength(defaultDotProductSIMDMinLength)
	b.Cleanup(func() {
		SetSIMDMinLength(oldMinLen)
	})

	sizes := []int{384, 768, 1023, 1024, 1536, 3072}

	for _, size := range sizes {
		a, v := benchmarkVectors(size)

		b.Run(fmt.Sprintf("scalar/size=%d", size), func(b *testing.B) {
			b.ReportAllocs()
			var sum float32
			for i := 0; i < b.N; i++ {
				sum = dotProductScalar(a, v)
			}
			benchmarkDotProductSink = sum
		})

		b.Run(fmt.Sprintf("simd/size=%d", size), func(b *testing.B) {
			if !dotProductSIMDEnabled() {
				b.Skip("SIMD disabled: build with GOEXPERIMENT=simd on amd64 AVX-capable CPU")
			}
			b.ReportAllocs()
			var sum float32
			for i := 0; i < b.N; i++ {
				sum = dotProductSIMD(a, v)
			}
			benchmarkDotProductSink = sum
		})

		b.Run(fmt.Sprintf("optimized/size=%d", size), func(b *testing.B) {
			b.ReportAllocs()
			var sum float32
			for i := 0; i < b.N; i++ {
				sum = dotProductOptimized(a, v)
			}
			benchmarkDotProductSink = sum
		})
	}
}

func BenchmarkDotProductPair(b *testing.B) {
	sizes := []int{384, 768, 1024, 1536, 3072}

	for _, size := range sizes {
		a, c := benchmarkVectors(size)
		bvec := make([]float32, size)
		for i := range size {
			bvec[i] = float32((i%83)+1) / 83
		}

		b.Run(fmt.Sprintf("two_scalar_passes/size=%d", size), func(b *testing.B) {
			b.ReportAllocs()
			var s1, s2 float32
			for i := 0; i < b.N; i++ {
				s1 = dotProductScalar(a, c)
				s2 = dotProductScalar(bvec, c)
			}
			benchmarkDotProductPairSinkA = s1
			benchmarkDotProductPairSinkB = s2
		})

		b.Run(fmt.Sprintf("fused_single_pass/size=%d", size), func(b *testing.B) {
			b.ReportAllocs()
			var s1, s2 float32
			for i := 0; i < b.N; i++ {
				s1, s2 = dotProductPairScalar(a, bvec, c)
			}
			benchmarkDotProductPairSinkA = s1
			benchmarkDotProductPairSinkB = s2
		})
	}
}
