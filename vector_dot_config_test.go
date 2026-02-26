package chromem

import "testing"

func TestSetSIMDMinLength(t *testing.T) {
	oldMinLen := getSIMDMinLength()
	t.Cleanup(func() {
		SetSIMDMinLength(oldMinLen)
	})

	SetSIMDMinLength(2048)
	if got := getSIMDMinLength(); got != 2048 {
		t.Fatalf("expected SIMD min length 2048, got %d", got)
	}

	SetSIMDMinLength(-1)
	if got := getSIMDMinLength(); got != 2048 {
		t.Fatalf("negative value should be ignored, expected 2048, got %d", got)
	}
}
