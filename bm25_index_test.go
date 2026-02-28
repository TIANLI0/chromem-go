package chromem

import (
	"context"
	"testing"
)

func TestBM25IndexBuildSearch(t *testing.T) {
	docs := []*Document{
		{ID: "go-1", Content: "Go language concurrency patterns"},
		{ID: "py-1", Content: "Python data science notebook"},
		{ID: "go-2", Content: "Advanced Go channels and goroutines"},
	}

	idx := newBM25Index()
	idx.Build(docs)

	res := idx.Search("go concurrency", 2)
	if len(res) == 0 {
		t.Fatal("expected non-empty bm25 results")
	}
	if res[0].doc == nil || res[0].doc.ID != "go-1" {
		t.Fatalf("expected top result go-1, got %+v", res)
	}
}

func TestCollection_QueryWithOptions_BM25IndexType(t *testing.T) {
	oldIndexType := getANNIndexType()
	annIndexTypeValue.Store(annIndexTypeBM25)
	t.Cleanup(func() {
		annIndexTypeValue.Store(oldIndexType)
	})

	db := NewDB()
	c, err := db.CreateCollection("bm25-query", nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}

	ctx := context.Background()
	if err := c.AddDocument(ctx, Document{ID: "doc-go", Embedding: normalizeVector([]float32{1, 0}), Content: "go concurrency channels"}); err != nil {
		t.Fatal("expected no error, got", err)
	}
	if err := c.AddDocument(ctx, Document{ID: "doc-js", Embedding: normalizeVector([]float32{1, 0}), Content: "javascript frontend ui"}); err != nil {
		t.Fatal("expected no error, got", err)
	}

	res, err := c.QueryWithOptions(ctx, QueryOptions{
		QueryText:      "concurrency go",
		QueryEmbedding: normalizeVector([]float32{0, 1}),
		NResults:       1,
	})
	if err != nil {
		t.Fatal("expected no error, got", err)
	}
	if len(res) != 1 || res[0].ID != "doc-go" {
		t.Fatalf("expected bm25 to return doc-go, got %+v", res)
	}
}

func TestCollection_QueryWithOptions_HybridIndexType(t *testing.T) {
	oldIndexType := getANNIndexType()
	annIndexTypeValue.Store(annIndexTypeHybrid)
	t.Cleanup(func() {
		annIndexTypeValue.Store(oldIndexType)
	})

	db := NewDB()
	c, err := db.CreateCollection("hybrid-query", nil, nil)
	if err != nil {
		t.Fatal("expected no error, got", err)
	}

	ctx := context.Background()
	if err := c.AddDocument(ctx, Document{ID: "vec-doc", Embedding: normalizeVector([]float32{1, 0}), Content: "unrelated text"}); err != nil {
		t.Fatal("expected no error, got", err)
	}
	if err := c.AddDocument(ctx, Document{ID: "lex-doc", Embedding: normalizeVector([]float32{0, 1}), Content: "golang hnsw ann bm25"}); err != nil {
		t.Fatal("expected no error, got", err)
	}

	res, err := c.QueryWithOptions(ctx, QueryOptions{
		QueryText:      "hnsw bm25",
		QueryEmbedding: normalizeVector([]float32{1, 0}),
		NResults:       2,
	})
	if err != nil {
		t.Fatal("expected no error, got", err)
	}
	if len(res) != 2 {
		t.Fatalf("expected 2 results, got %+v", res)
	}

	foundVec := false
	foundLex := false
	for _, item := range res {
		switch item.ID {
		case "vec-doc":
			foundVec = true
		case "lex-doc":
			foundLex = true
		}
	}
	if !foundVec || !foundLex {
		t.Fatalf("expected hybrid results to include both vec-doc and lex-doc, got %+v", res)
	}
}
