package chromem

import "testing"

func TestHNSWTuningSetters(t *testing.T) {
	oldEnabled := getHNSWEnabled()
	oldM := getHNSWM()
	oldEFC := getHNSWEFConstruction()
	oldEFS := getHNSWEFSearch()
	oldTombstoneRatio := getHNSWTombstoneRebuildRatio()
	oldTombstoneMinDeleted := getHNSWTombstoneRebuildMinDeleted()

	t.Cleanup(func() {
		SetHNSWEnabled(oldEnabled)
		SetHNSWM(oldM)
		SetHNSWEFConstruction(oldEFC)
		SetHNSWEFSearch(oldEFS)
		SetHNSWTombstoneRebuildRatio(oldTombstoneRatio)
		SetHNSWTombstoneRebuildMinDeleted(oldTombstoneMinDeleted)
	})

	SetHNSWEnabled(false)
	if got := getHNSWEnabled(); got {
		t.Fatal("expected HNSW disabled")
	}

	SetHNSWEnabled(true)
	if got := getHNSWEnabled(); !got {
		t.Fatal("expected HNSW enabled")
	}

	SetHNSWM(24)
	if got := getHNSWM(); got != 24 {
		t.Fatalf("expected HNSW M=24, got %d", got)
	}
	SetHNSWM(1)
	if got := getHNSWM(); got != 24 {
		t.Fatalf("invalid HNSW M should be ignored, got %d", got)
	}

	SetHNSWEFConstruction(256)
	if got := getHNSWEFConstruction(); got != 256 {
		t.Fatalf("expected HNSW efConstruction=256, got %d", got)
	}
	SetHNSWEFConstruction(0)
	if got := getHNSWEFConstruction(); got != 256 {
		t.Fatalf("invalid HNSW efConstruction should be ignored, got %d", got)
	}

	SetHNSWEFSearch(96)
	if got := getHNSWEFSearch(); got != 96 {
		t.Fatalf("expected HNSW efSearch=96, got %d", got)
	}
	SetHNSWEFSearch(0)
	if got := getHNSWEFSearch(); got != 96 {
		t.Fatalf("invalid HNSW efSearch should be ignored, got %d", got)
	}

	SetHNSWTombstoneRebuildRatio(0.35)
	if got := getHNSWTombstoneRebuildRatio(); got != 0.35 {
		t.Fatalf("expected HNSW tombstone rebuild ratio=0.35, got %v", got)
	}
	SetHNSWTombstoneRebuildRatio(-1)
	if got := getHNSWTombstoneRebuildRatio(); got != 0.35 {
		t.Fatalf("invalid HNSW tombstone rebuild ratio should be ignored, got %v", got)
	}

	SetHNSWTombstoneRebuildMinDeleted(1234)
	if got := getHNSWTombstoneRebuildMinDeleted(); got != 1234 {
		t.Fatalf("expected HNSW tombstone rebuild min deleted=1234, got %d", got)
	}
	SetHNSWTombstoneRebuildMinDeleted(-1)
	if got := getHNSWTombstoneRebuildMinDeleted(); got != 1234 {
		t.Fatalf("invalid HNSW tombstone rebuild min deleted should be ignored, got %d", got)
	}
}
