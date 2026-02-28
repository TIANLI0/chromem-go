package chromem

import "testing"

func testANNDocs() []*Document {
	raw := [][]float32{
		{1, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0},
		{0.7, 0.7, 0, 0, 0, 0, 0, 0},
		{0.6, 0.2, 0.6, 0, 0, 0, 0, 0},
		{0, 0, 0, 1, 0, 0, 0, 0},
	}
	docs := make([]*Document, 0, len(raw))
	for i, vec := range raw {
		norm := normalizeVector(vec)
		docs = append(docs, &Document{ID: "doc-" + string(rune('A'+i)), Embedding: norm})
	}
	return docs
}

func containsDocID(items []hnswNeighbor, id string) bool {
	for _, item := range items {
		if item.doc != nil && item.doc.ID == id {
			return true
		}
	}
	return false
}

func TestIVFIndexBuildSearch(t *testing.T) {
	docs := testANNDocs()
	idx := newIVFIndex(len(docs[0].Embedding), 3, 2)
	if err := idx.Build(docs); err != nil {
		t.Fatalf("Build() error: %v", err)
	}

	query := normalizeVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})
	res, err := idx.Search(query, 3)
	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}
	if len(res) == 0 {
		t.Fatal("expected non-empty results")
	}
	if !containsDocID(res, "doc-A") {
		t.Fatalf("expected result set to contain doc-A, got %+v", res)
	}
}

func TestPQIndexBuildSearch(t *testing.T) {
	docs := testANNDocs()
	idx := newPQIndex(len(docs[0].Embedding), 4, 4)
	if err := idx.Build(docs); err != nil {
		t.Fatalf("Build() error: %v", err)
	}

	query := normalizeVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})
	res, err := idx.Search(query, 3)
	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}
	if len(res) == 0 {
		t.Fatal("expected non-empty results")
	}
	if !containsDocID(res, "doc-A") {
		t.Fatalf("expected result set to contain doc-A, got %+v", res)
	}
}

func TestIVFPQIndexBuildSearch(t *testing.T) {
	docs := testANNDocs()
	idx := newIVFPQIndex(len(docs[0].Embedding), 3, 2, 4, 4)
	if err := idx.Build(docs); err != nil {
		t.Fatalf("Build() error: %v", err)
	}

	query := normalizeVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})
	res, err := idx.Search(query, 3)
	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}
	if len(res) == 0 {
		t.Fatal("expected non-empty results")
	}
	if !containsDocID(res, "doc-A") {
		t.Fatalf("expected result set to contain doc-A, got %+v", res)
	}
}
