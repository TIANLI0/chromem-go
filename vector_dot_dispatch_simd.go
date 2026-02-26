//go:build goexperiment.simd && amd64

package chromem

import "simd/archsimd"

func init() {
	if archsimd.X86.AVX() {
		dotProductSIMDFunc = dotProductSIMDArch
		dotProductSIMDEnabledFlag = true
	}
}

func dotProductSIMDArch(a, b []float32) float32 {
	var sum8 archsimd.Float32x8
	i := 0
	for ; i+8 <= len(a); i += 8 {
		ax := archsimd.LoadFloat32x8((*[8]float32)(a[i : i+8]))
		bx := archsimd.LoadFloat32x8((*[8]float32)(b[i : i+8]))
		sum8 = sum8.Add(ax.Mul(bx))
	}

	sum4 := sum8.GetLo().Add(sum8.GetHi())
	sum := sum4.GetElem(0) + sum4.GetElem(1) + sum4.GetElem(2) + sum4.GetElem(3)

	for ; i < len(a); i++ {
		sum += a[i] * b[i]
	}

	return sum
}
