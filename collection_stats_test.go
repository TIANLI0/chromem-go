package chromem

import (
	"context"
	"testing"
)

func TestCollectionMemoryStats(t *testing.T) {
	ctx := context.Background()
	db := NewDB()

	embedFunc := func(_ context.Context, _ string) ([]float32, error) {
		return nil, nil
	}

	c, err := db.CreateCollection("stats-test", nil, embedFunc)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}

	docs := []Document{
		{
			ID:        "1",
			Embedding: normalizeVector([]float32{1, 2, 3}),
			Metadata:  map[string]string{"k": "v"},
			Content:   "hello",
		},
		{
			ID:        "2",
			Embedding: normalizeVector([]float32{4, 5}),
			Metadata:  map[string]string{"k2": "v2"},
			Content:   "world",
		},
	}

	for _, doc := range docs {
		if err := c.AddDocument(ctx, doc); err != nil {
			t.Fatal("expected no error, got", err)
		}
	}

	stats := c.MemoryStats()
	if stats.DocumentCount != 2 {
		t.Fatalf("expected document count 2, got %d", stats.DocumentCount)
	}

	expectedEmbeddingBytes := uint64((3 + 2) * 4)
	if stats.ApproxEmbeddingBytes != expectedEmbeddingBytes {
		t.Fatalf("expected embedding bytes %d, got %d", expectedEmbeddingBytes, stats.ApproxEmbeddingBytes)
	}

	if stats.ApproxMetadataBytes == 0 {
		t.Fatal("expected metadata bytes > 0")
	}
	if stats.ApproxContentBytes == 0 {
		t.Fatal("expected content bytes > 0")
	}

	expectedTotal := stats.ApproxEmbeddingBytes + stats.ApproxMetadataBytes + stats.ApproxContentBytes
	if stats.EstimatedTotalBytes != expectedTotal {
		t.Fatalf("expected total bytes %d, got %d", expectedTotal, stats.EstimatedTotalBytes)
	}

	if stats.RuntimeAllocBytes == 0 {
		t.Fatal("expected runtime alloc bytes > 0")
	}
	if stats.RuntimeHeapInuse == 0 {
		t.Fatal("expected runtime heap in-use bytes > 0")
	}
}
