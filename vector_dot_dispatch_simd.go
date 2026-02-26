//go:build goexperiment.simd && amd64

package chromem

import "simd/archsimd"

func init() {
	if archsimd.X86.AVX() {
		dotProductSIMDFunc = dotProductSIMDArch
		dotProductSIMDEnabledFlag = true
	}
}

// dotProductSIMDArch computes the dot product of two float32 slices using SIMD instructions.
// It processes 16 elements at a time using AVX, then handles any remaining elements.
// The slices must be of the same length.
func dotProductSIMDArch(a, b []float32) float32 {
	var sum80, sum81 archsimd.Float32x8
	i := 0
	for ; i+16 <= len(a); i += 16 {
		ax0 := archsimd.LoadFloat32x8((*[8]float32)(a[i : i+8]))    // Load 8 elements from a and b into SIMD registers
		bx0 := archsimd.LoadFloat32x8((*[8]float32)(b[i : i+8]))    // Compute the product of the first 8 elements and accumulate into sum80
		ax1 := archsimd.LoadFloat32x8((*[8]float32)(a[i+8 : i+16])) // Compute the product of the next 8 elements and accumulate into sum81
		bx1 := archsimd.LoadFloat32x8((*[8]float32)(b[i+8 : i+16])) // Add the products to the respective sums

		sum80 = sum80.Add(ax0.Mul(bx0)) // Add the products to the respective sums
		sum81 = sum81.Add(ax1.Mul(bx1)) // Add the products to the respective sums
	}

	// Combine the two sums into one and then horizontally add the elements to get the final dot product.
	sum8 := sum80.Add(sum81)
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
