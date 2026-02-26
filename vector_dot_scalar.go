package chromem

func dotProductScalar(a, b []float32) float32 {
	var sum0, sum1, sum2, sum3 float32
	i := 0
	n := len(a)

	for ; i+4 <= n; i += 4 {
		sum0 += a[i] * b[i]
		sum1 += a[i+1] * b[i+1]
		sum2 += a[i+2] * b[i+2]
		sum3 += a[i+3] * b[i+3]
	}

	dotProduct := sum0 + sum1 + sum2 + sum3
	for ; i < n; i++ {
		dotProduct += a[i] * b[i]
	}

	return dotProduct
}
