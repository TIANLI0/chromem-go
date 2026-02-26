package chromem

import "testing"

func TestQueryTuningSetters(t *testing.T) {
	oldSmall := getQuerySmallDocsThreshold()
	oldSequential := getQuerySequentialDocsThreshold()
	oldHighDim := getQueryHighDimThreshold()
	oldDivisor := getQueryHighDimConcurrencyDivisor()

	t.Cleanup(func() {
		SetQuerySmallDocsThreshold(oldSmall)
		SetQuerySequentialDocsThreshold(oldSequential)
		SetQueryHighDimThreshold(oldHighDim)
		SetQueryHighDimConcurrencyDivisor(oldDivisor)
	})

	SetQuerySmallDocsThreshold(4096)
	if got := getQuerySmallDocsThreshold(); got != 4096 {
		t.Fatalf("expected small docs threshold 4096, got %d", got)
	}

	SetQuerySequentialDocsThreshold(1024)
	if got := getQuerySequentialDocsThreshold(); got != 1024 {
		t.Fatalf("expected sequential docs threshold 1024, got %d", got)
	}

	SetQueryHighDimThreshold(3072)
	if got := getQueryHighDimThreshold(); got != 3072 {
		t.Fatalf("expected high dim threshold 3072, got %d", got)
	}

	SetQueryHighDimConcurrencyDivisor(3)
	if got := getQueryHighDimConcurrencyDivisor(); got != 3 {
		t.Fatalf("expected high dim divisor 3, got %d", got)
	}

	SetQuerySmallDocsThreshold(-1)
	if got := getQuerySmallDocsThreshold(); got != 4096 {
		t.Fatalf("negative small docs threshold should be ignored, got %d", got)
	}

	SetQuerySequentialDocsThreshold(-1)
	if got := getQuerySequentialDocsThreshold(); got != 1024 {
		t.Fatalf("negative sequential docs threshold should be ignored, got %d", got)
	}

	SetQueryHighDimThreshold(-1)
	if got := getQueryHighDimThreshold(); got != 3072 {
		t.Fatalf("negative high dim threshold should be ignored, got %d", got)
	}

	SetQueryHighDimConcurrencyDivisor(0)
	if got := getQueryHighDimConcurrencyDivisor(); got != 3 {
		t.Fatalf("non-positive divisor should be ignored, got %d", got)
	}
}
