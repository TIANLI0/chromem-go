package chromem

import (
	"runtime"
	"testing"
)

func TestQueryTuningSetters(t *testing.T) {
	oldSmall := getQuerySmallDocsThreshold()
	oldSequential := getQuerySequentialDocsThreshold()
	oldHighDim := getQueryHighDimThreshold()
	oldDivisor := getQueryHighDimConcurrencyDivisor()
	oldMaxConcurrency := getQueryMaxConcurrency()

	t.Cleanup(func() {
		SetQuerySmallDocsThreshold(oldSmall)
		SetQuerySequentialDocsThreshold(oldSequential)
		SetQueryHighDimThreshold(oldHighDim)
		SetQueryHighDimConcurrencyDivisor(oldDivisor)
		SetQueryMaxConcurrency(oldMaxConcurrency)
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

	SetQueryMaxConcurrency(6)
	if got := getQueryMaxConcurrency(); got != 6 {
		t.Fatalf("expected max concurrency 6, got %d", got)
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

	SetQueryMaxConcurrency(-1)
	if got := getQueryMaxConcurrency(); got != 6 {
		t.Fatalf("negative max concurrency should be ignored, got %d", got)
	}

	SetQueryMaxConcurrency(0)
	if got := getQueryMaxConcurrency(); got != 0 {
		t.Fatalf("expected max concurrency reset to 0, got %d", got)
	}
}

func TestQueryConcurrency(t *testing.T) {
	oldSmall := getQuerySmallDocsThreshold()
	oldHighDim := getQueryHighDimThreshold()
	oldDivisor := getQueryHighDimConcurrencyDivisor()
	oldMaxConcurrency := getQueryMaxConcurrency()

	t.Cleanup(func() {
		SetQuerySmallDocsThreshold(oldSmall)
		SetQueryHighDimThreshold(oldHighDim)
		SetQueryHighDimConcurrencyDivisor(oldDivisor)
		SetQueryMaxConcurrency(oldMaxConcurrency)
	})

	SetQuerySmallDocsThreshold(2048)
	SetQueryHighDimThreshold(1024)
	SetQueryHighDimConcurrencyDivisor(2)
	SetQueryMaxConcurrency(0)

	workers := runtime.GOMAXPROCS(0)
	if workers <= 0 {
		workers = runtime.NumCPU()
	}

	if got := queryConcurrency(0, 512); got != 0 {
		t.Fatalf("expected 0 concurrency for 0 docs, got %d", got)
	}

	if got, want := queryConcurrency(10_000, 512), min(10_000, workers); got != want {
		t.Fatalf("expected concurrency %d for non-high-dim query, got %d", want, got)
	}

	if got, want := queryConcurrency(10_000, 1536), min(10_000, max(workers/2, 1)); got != want {
		t.Fatalf("expected reduced concurrency %d for high-dim large query, got %d", want, got)
	}

	if got, want := queryConcurrency(512, 1536), min(512, workers); got != want {
		t.Fatalf("expected full concurrency %d for high-dim small query, got %d", want, got)
	}

	if workers >= 2 {
		if got, want := queryConcurrency(10_000, 4096), min(10_000, max(workers/8, 1)); got != want {
			t.Fatalf("expected adaptively reduced concurrency %d for very-high-dim query, got %d", want, got)
		}
	}

	SetQueryMaxConcurrency(3)
	if got, want := queryConcurrency(10_000, 512), min(10_000, 3); got != want {
		t.Fatalf("expected capped concurrency %d, got %d", want, got)
	}
}
