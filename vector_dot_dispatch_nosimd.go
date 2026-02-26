package chromem

var dotProductSIMDFunc = dotProductScalar
var dotProductSIMDEnabledFlag bool

func dotProductOptimized(a, b []float32) float32 {
	if dotProductSIMDEnabledFlag && len(a) >= getSIMDMinLength() {
		return dotProductSIMDFunc(a, b)
	}
	return dotProductScalar(a, b)
}

func dotProductSIMD(a, b []float32) float32 {
	return dotProductSIMDFunc(a, b)
}

func dotProductSIMDEnabled() bool {
	return dotProductSIMDEnabledFlag
}
