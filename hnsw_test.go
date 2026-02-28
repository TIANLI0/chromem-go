package chromem

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestHNSWSelectNeighbors_DiversifyThenFill(t *testing.T) {
	idx := &hnswIndex{
		nodes: []hnswNode{
			{doc: &Document{ID: "a", Embedding: []float32{1, 0}}},
			{doc: &Document{ID: "b", Embedding: []float32{1, 0}}},
			{doc: &Document{ID: "c", Embedding: []float32{1, 0}}},
		},
	}

	candidates := []hnswCandidate{
		{id: 0, similarity: 0.80},
		{id: 1, similarity: 0.70},
		{id: 2, similarity: 0.60},
	}

	selected := idx.selectNeighbors(candidates, 2)
	if len(selected) != 2 {
		t.Fatalf("expected 2 neighbors, got %d", len(selected))
	}
	if selected[0] != 0 || selected[1] != 1 {
		t.Fatalf("expected [0 1], got %v", selected)
	}

	idx.setDeleted(1, true)
	selected = idx.selectNeighbors(candidates, 2)
	if len(selected) != 2 {
		t.Fatalf("expected 2 neighbors with deleted candidate skipped, got %d", len(selected))
	}
	if selected[0] != 0 || selected[1] != 2 {
		t.Fatalf("expected [0 2] when candidate 1 is deleted, got %v", selected)
	}
}

func TestCollection_QueryEmbedding_HNSWRebuildOnMutation(t *testing.T) {
	db := NewDB()
	c, err := db.CreateCollection("hnsw-rebuild", nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}

	ctx := context.Background()
	err = c.AddDocument(ctx, Document{ID: "a", Embedding: []float32{1, 0}, Content: "a"})
	if err != nil {
		t.Fatal("expected no error, got", err)
	}
	err = c.AddDocument(ctx, Document{ID: "b", Embedding: []float32{0, 1}, Content: "b"})
	if err != nil {
		t.Fatal("expected no error, got", err)
	}

	res, err := c.QueryEmbedding(ctx, []float32{1, 0}, 1, nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}
	if len(res) != 1 || res[0].ID != "a" {
		t.Fatalf("expected top result 'a', got %+v", res)
	}

	err = c.Delete(ctx, nil, nil, "a")
	if err != nil {
		t.Fatal("expected no error, got", err)
	}

	res, err = c.QueryEmbedding(ctx, []float32{1, 0}, 1, nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}
	if len(res) != 1 || res[0].ID != "b" {
		t.Fatalf("expected top result 'b' after delete, got %+v", res)
	}

	err = c.AddDocument(ctx, Document{ID: "c", Embedding: []float32{1, 0}, Content: "c"})
	if err != nil {
		t.Fatal("expected no error, got", err)
	}

	res, err = c.QueryEmbedding(ctx, []float32{1, 0}, 1, nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}
	if len(res) != 1 || res[0].ID != "c" {
		t.Fatalf("expected top result 'c' after add, got %+v", res)
	}
}

func TestCollection_HNSWIndexPersistenceAndMutationInvalidation(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	db, err := NewPersistentDBWithOptions(dir, PersistentDBOptions{Compress: false})
	if err != nil {
		t.Fatal("expected no error, got", err)
	}

	c, err := db.CreateCollection("hnsw-persist", nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}

	if err := c.AddDocument(ctx, Document{ID: "a", Embedding: []float32{1, 0}, Content: "a"}); err != nil {
		t.Fatal("expected no error, got", err)
	}
	if err := c.AddDocument(ctx, Document{ID: "b", Embedding: []float32{0, 1}, Content: "b"}); err != nil {
		t.Fatal("expected no error, got", err)
	}

	_, err = c.QueryEmbedding(ctx, []float32{1, 0}, 1, nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}

	indexPath := c.getHNSWIndexPath()
	if _, err := os.Stat(indexPath); err != nil {
		t.Fatalf("expected persisted hnsw index at %s, got error: %v", indexPath, err)
	}

	dbReloaded, err := NewPersistentDBWithOptions(filepath.Clean(dir), PersistentDBOptions{Compress: false})
	if err != nil {
		t.Fatal("expected no error, got", err)
	}

	cReloaded := dbReloaded.GetCollection("hnsw-persist", nil)
	if cReloaded == nil {
		t.Fatal("expected collection, got nil")
	}
	if cReloaded.hnsw != nil {
		t.Fatal("expected no in-memory hnsw before query")
	}

	_, err = cReloaded.QueryEmbedding(ctx, []float32{1, 0}, 1, nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}
	if cReloaded.hnsw == nil {
		t.Fatal("expected hnsw index to be loaded or rebuilt on query")
	}

	if err := cReloaded.AddDocument(ctx, Document{ID: "c", Embedding: []float32{1, 0}, Content: "c"}); err != nil {
		t.Fatal("expected no error, got", err)
	}
	if cReloaded.hnsw == nil {
		t.Fatal("expected in-memory hnsw to be incrementally updated after append")
	}
	if _, err := os.Stat(cReloaded.getHNSWIndexPath()); err != nil {
		t.Fatalf("expected persisted hnsw index to exist after incremental append, got err: %v", err)
	}
}

func TestCollection_HNSWIncrementalInsertWithoutGlobalRebuild(t *testing.T) {
	ctx := context.Background()
	db := NewDB()
	c, err := db.CreateCollection("hnsw-incremental", nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}

	if err := c.AddDocument(ctx, Document{ID: "a", Embedding: []float32{1, 0}}); err != nil {
		t.Fatal("expected no error, got", err)
	}
	if err := c.AddDocument(ctx, Document{ID: "b", Embedding: []float32{0, 1}}); err != nil {
		t.Fatal("expected no error, got", err)
	}

	_, err = c.QueryEmbedding(ctx, []float32{1, 0}, 1, nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}
	if c.hnsw == nil {
		t.Fatal("expected hnsw index to be built")
	}
	oldIndex := c.hnsw

	if err := c.AddDocument(ctx, Document{ID: "c", Embedding: []float32{1, 0}}); err != nil {
		t.Fatal("expected no error, got", err)
	}

	if c.hnsw == nil {
		t.Fatal("expected hnsw index to be incrementally updated")
	}
	if c.hnsw == oldIndex {
		t.Fatal("expected copy-on-write hnsw index replacement")
	}
	if c.hnswVersion.Load() != c.hnswIndexedVersion.Load() {
		t.Fatal("expected hnsw version and indexed version to stay in sync after incremental insert")
	}

	res, err := c.QueryEmbedding(ctx, []float32{1, 0}, 2, nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}
	foundC := false
	for _, item := range res {
		if item.ID == "c" {
			foundC = true
			break
		}
	}
	if !foundC {
		t.Fatalf("expected top-2 results to include 'c', got %+v", res)
	}
}

func TestCollection_HNSWOverwriteUsesIncrementalUpsert(t *testing.T) {
	ctx := context.Background()
	db := NewDB()
	c, err := db.CreateCollection("hnsw-overwrite", nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}

	if err := c.AddDocument(ctx, Document{ID: "a", Embedding: []float32{1, 0}}); err != nil {
		t.Fatal("expected no error, got", err)
	}
	if err := c.AddDocument(ctx, Document{ID: "b", Embedding: []float32{0, 1}}); err != nil {
		t.Fatal("expected no error, got", err)
	}
	if _, err := c.QueryEmbedding(ctx, []float32{1, 0}, 1, nil, nil); err != nil {
		t.Fatal("expected no error, got", err)
	}
	oldIndex := c.hnsw

	if err := c.AddDocument(ctx, Document{ID: "a", Embedding: []float32{0, 1}}); err != nil {
		t.Fatal("expected no error, got", err)
	}
	if c.hnsw == nil {
		t.Fatal("expected hnsw index to remain available after overwrite upsert")
	}
	if c.hnsw == oldIndex {
		t.Fatal("expected copy-on-write index replacement on overwrite")
	}

	res, err := c.QueryEmbedding(ctx, []float32{1, 0}, 2, nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}
	for _, item := range res {
		if item.ID == "a" {
			if item.Similarity > 0.1 {
				t.Fatalf("expected overwritten doc 'a' to be away from query after upsert, got similarity %v", item.Similarity)
			}
		}
	}
}

func TestCollection_HNSWDeleteUsesIncrementalTombstone(t *testing.T) {
	ctx := context.Background()
	db := NewDB()
	c, err := db.CreateCollection("hnsw-delete", nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}

	if err := c.AddDocument(ctx, Document{ID: "a", Embedding: []float32{1, 0}}); err != nil {
		t.Fatal("expected no error, got", err)
	}
	if err := c.AddDocument(ctx, Document{ID: "b", Embedding: []float32{0, 1}}); err != nil {
		t.Fatal("expected no error, got", err)
	}
	if _, err := c.QueryEmbedding(ctx, []float32{1, 0}, 1, nil, nil); err != nil {
		t.Fatal("expected no error, got", err)
	}
	oldIndex := c.hnsw

	if err := c.Delete(ctx, nil, nil, "a"); err != nil {
		t.Fatal("expected no error, got", err)
	}
	if c.hnsw == nil {
		t.Fatal("expected hnsw index to remain available after delete tombstone")
	}
	if c.hnsw == oldIndex {
		t.Fatal("expected copy-on-write index replacement on delete")
	}

	res, err := c.QueryEmbedding(ctx, []float32{1, 0}, 1, nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}
	if len(res) != 1 || res[0].ID != "b" {
		t.Fatalf("expected deleted doc to be excluded from hnsw results, got %+v", res)
	}
}

func TestCollection_HNSWTombstoneThresholdCompactsOnQuery(t *testing.T) {
	ctx := context.Background()

	oldRatio := getHNSWTombstoneRebuildRatio()
	oldMinDeleted := getHNSWTombstoneRebuildMinDeleted()
	t.Cleanup(func() {
		SetHNSWTombstoneRebuildRatio(oldRatio)
		SetHNSWTombstoneRebuildMinDeleted(oldMinDeleted)
	})
	SetHNSWTombstoneRebuildRatio(0.10)
	SetHNSWTombstoneRebuildMinDeleted(1)

	db := NewDB()
	c, err := db.CreateCollection("hnsw-compact", nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}

	if err := c.AddDocument(ctx, Document{ID: "a", Embedding: []float32{1, 0}}); err != nil {
		t.Fatal("expected no error, got", err)
	}
	if err := c.AddDocument(ctx, Document{ID: "b", Embedding: []float32{0, 1}}); err != nil {
		t.Fatal("expected no error, got", err)
	}
	if err := c.AddDocument(ctx, Document{ID: "c", Embedding: []float32{0.7, 0.7}}); err != nil {
		t.Fatal("expected no error, got", err)
	}

	if _, err := c.QueryEmbedding(ctx, []float32{1, 0}, 1, nil, nil); err != nil {
		t.Fatal("expected no error, got", err)
	}

	if err := c.Delete(ctx, nil, nil, "b"); err != nil {
		t.Fatal("expected no error, got", err)
	}

	c.hnswBuildLock.Lock()
	beforeNodes := len(c.hnsw.nodes)
	beforeDeleted := c.hnsw.deletedCount()
	c.hnswBuildLock.Unlock()
	if beforeDeleted == 0 {
		t.Fatal("expected tombstoned nodes before compaction")
	}

	if _, err := c.QueryEmbedding(ctx, []float32{1, 0}, 1, nil, nil); err != nil {
		t.Fatal("expected no error, got", err)
	}

	c.hnswBuildLock.Lock()
	afterNodes := len(c.hnsw.nodes)
	afterDeleted := c.hnsw.deletedCount()
	c.hnswBuildLock.Unlock()
	if afterDeleted != 0 {
		t.Fatalf("expected compaction to remove tombstones, got deleted=%d", afterDeleted)
	}
	if afterNodes >= beforeNodes {
		t.Fatalf("expected compaction to shrink node count, before=%d after=%d", beforeNodes, afterNodes)
	}
}
