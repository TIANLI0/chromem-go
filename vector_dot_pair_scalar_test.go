package chromem

import (
	"math"
	"testing"
)

func TestDotProductPairScalar(t *testing.T) {
	a := []float32{0.1, 0.2, 0.3, 0.4, 0.5}
	b := []float32{0.5, 0.4, 0.3, 0.2, 0.1}
	c := []float32{1, 2, 3, 4, 5}

	gotA, gotB := dotProductPairScalar(a, b, c)
	wantA := dotProductScalar(a, c)
	wantB := dotProductScalar(b, c)

	if math.Abs(float64(gotA-wantA)) > 1e-6 {
		t.Fatalf("unexpected first dot product: got %v, want %v", gotA, wantA)
	}
	if math.Abs(float64(gotB-wantB)) > 1e-6 {
		t.Fatalf("unexpected second dot product: got %v, want %v", gotB, wantB)
	}
}
