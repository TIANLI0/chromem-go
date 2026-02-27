package chromem

// dotProductPairScalar computes two dot products against the same vector in a
// single pass: dot(a, c) and dot(b, c).
func dotProductPairScalar(a, b, c []float32) (float32, float32) {
	var sumA0, sumA1, sumA2, sumA3 float32
	var sumB0, sumB1, sumB2, sumB3 float32
	i := 0
	n := len(c)

	for ; i+4 <= n; i += 4 {
		c0 := c[i]
		c1 := c[i+1]
		c2 := c[i+2]
		c3 := c[i+3]

		sumA0 += a[i] * c0
		sumA1 += a[i+1] * c1
		sumA2 += a[i+2] * c2
		sumA3 += a[i+3] * c3

		sumB0 += b[i] * c0
		sumB1 += b[i+1] * c1
		sumB2 += b[i+2] * c2
		sumB3 += b[i+3] * c3
	}

	dotA := sumA0 + sumA1 + sumA2 + sumA3
	dotB := sumB0 + sumB1 + sumB2 + sumB3
	for ; i < n; i++ {
		ci := c[i]
		dotA += a[i] * ci
		dotB += b[i] * ci
	}

	return dotA, dotB
}
