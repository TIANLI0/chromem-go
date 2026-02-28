package chromem

import (
	"cmp"
	"fmt"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"testing"
)

type annBenchExactResult struct {
	ids map[string]struct{}
}

type annBenchCandidate struct {
	id  string
	sim float32
}

func annBenchDataset(numDocs, dim, numQueries int) ([]*Document, [][]float32) {
	rng := rand.New(rand.NewSource(20260227))
	clusterCount := min(32, numDocs)

	centers := make([][]float32, clusterCount)
	for i := range clusterCount {
		center := make([]float32, dim)
		for d := range dim {
			center[d] = rng.Float32()*2 - 1
		}
		centers[i] = normalizeVector(center)
	}

	docs := make([]*Document, 0, numDocs)
	for i := range numDocs {
		center := centers[i%clusterCount]
		vec := make([]float32, dim)
		for d := range dim {
			vec[d] = center[d] + (rng.Float32()*2-1)*0.05
		}
		vec = normalizeVector(vec)
		docs = append(docs, &Document{ID: fmt.Sprintf("doc-%d", i), Embedding: vec})
	}

	queries := make([][]float32, 0, numQueries)
	for range numQueries {
		base := docs[rng.Intn(len(docs))].Embedding
		q := make([]float32, dim)
		for d := range dim {
			q[d] = base[d] + (rng.Float32()*2-1)*0.03
		}
		queries = append(queries, normalizeVector(q))
	}

	return docs, queries
}

func annBenchExactTopK(docs []*Document, query []float32, k int) annBenchExactResult {
	if k <= 0 {
		return annBenchExactResult{ids: map[string]struct{}{}}
	}
	cands := make([]annBenchCandidate, 0, len(docs))
	for _, doc := range docs {
		cands = append(cands, annBenchCandidate{id: doc.ID, sim: dotProductOptimized(query, doc.Embedding)})
	}
	slices.SortFunc(cands, func(a, b annBenchCandidate) int {
		return cmp.Compare(b.sim, a.sim)
	})
	limit := min(k, len(cands))
	out := annBenchExactResult{ids: make(map[string]struct{}, limit)}
	for i := range limit {
		out.ids[cands[i].id] = struct{}{}
	}
	return out
}

func annBenchRecallAtK(exact annBenchExactResult, got []hnswNeighbor) float64 {
	if len(exact.ids) == 0 {
		return 1
	}
	hit := 0
	for _, neighbor := range got {
		if neighbor.doc == nil {
			continue
		}
		if _, ok := exact.ids[neighbor.doc.ID]; ok {
			hit++
		}
	}
	return float64(hit) / float64(len(exact.ids))
}

func annBenchEnvInt(name string, fallback int) int {
	value, ok := os.LookupEnv(name)
	if !ok {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func BenchmarkANNIndex_Compare_10k_384d(b *testing.B) {
	numDocs := 10_000
	dim := 384
	numQueries := 128
	k := 10

	hnswM := annBenchEnvInt("CHROMEM_ANN_BENCH_HNSW_M", 16)
	hnswEFC := annBenchEnvInt("CHROMEM_ANN_BENCH_HNSW_EFC", 200)
	hnswEFS := annBenchEnvInt("CHROMEM_ANN_BENCH_HNSW_EFS", 100)

	ivfNList := annBenchEnvInt("CHROMEM_ANN_BENCH_IVF_NLIST", 128)
	ivfNProbe := annBenchEnvInt("CHROMEM_ANN_BENCH_IVF_NPROBE", 16)

	pqM := annBenchEnvInt("CHROMEM_ANN_BENCH_PQ_M", 16)
	pqNBits := annBenchEnvInt("CHROMEM_ANN_BENCH_PQ_NBITS", 8)

	ivfpqNList := annBenchEnvInt("CHROMEM_ANN_BENCH_IVFPQ_NLIST", 64)
	ivfpqNProbe := annBenchEnvInt("CHROMEM_ANN_BENCH_IVFPQ_NPROBE", 8)
	ivfpqM := annBenchEnvInt("CHROMEM_ANN_BENCH_IVFPQ_M", 8)
	ivfpqNBits := annBenchEnvInt("CHROMEM_ANN_BENCH_IVFPQ_NBITS", 6)

	docs, queries := annBenchDataset(numDocs, dim, numQueries)
	exact := make([]annBenchExactResult, len(queries))
	for i, q := range queries {
		exact[i] = annBenchExactTopK(docs, q, k)
	}

	b.Run("BruteForce", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = annBenchExactTopK(docs, queries[i%len(queries)], k)
		}
	})

	b.Run("HNSW", func(b *testing.B) {
		idx := newHNSWIndex(dim, hnswM, hnswEFC, hnswEFS)
		if err := idx.Build(docs); err != nil {
			b.Fatalf("build hnsw: %v", err)
		}
		recallSum := 0.0
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			qIdx := i % len(queries)
			res, err := idx.Search(queries[qIdx], k)
			if err != nil {
				b.Fatalf("search hnsw: %v", err)
			}
			recallSum += annBenchRecallAtK(exact[qIdx], res)
		}
		b.StopTimer()
		b.ReportMetric(recallSum/float64(b.N)*100, "recall@10_%")
	})

	b.Run("IVF", func(b *testing.B) {
		idx := newIVFIndex(dim, ivfNList, ivfNProbe)
		if err := idx.Build(docs); err != nil {
			b.Fatalf("build ivf: %v", err)
		}
		recallSum := 0.0
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			qIdx := i % len(queries)
			res, err := idx.Search(queries[qIdx], k)
			if err != nil {
				b.Fatalf("search ivf: %v", err)
			}
			recallSum += annBenchRecallAtK(exact[qIdx], res)
		}
		b.StopTimer()
		b.ReportMetric(recallSum/float64(b.N)*100, "recall@10_%")
	})

	b.Run("PQ", func(b *testing.B) {
		idx := newPQIndex(dim, pqM, pqNBits)
		if err := idx.Build(docs); err != nil {
			b.Fatalf("build pq: %v", err)
		}
		recallSum := 0.0
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			qIdx := i % len(queries)
			res, err := idx.Search(queries[qIdx], k)
			if err != nil {
				b.Fatalf("search pq: %v", err)
			}
			recallSum += annBenchRecallAtK(exact[qIdx], res)
		}
		b.StopTimer()
		b.ReportMetric(recallSum/float64(b.N)*100, "recall@10_%")
	})

	b.Run("IVFPQ", func(b *testing.B) {
		idx := newIVFPQIndex(dim, ivfpqNList, ivfpqNProbe, ivfpqM, ivfpqNBits)
		if err := idx.Build(docs); err != nil {
			b.Fatalf("build ivfpq: %v", err)
		}
		recallSum := 0.0
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			qIdx := i % len(queries)
			res, err := idx.Search(queries[qIdx], k)
			if err != nil {
				b.Fatalf("search ivfpq: %v", err)
			}
			recallSum += annBenchRecallAtK(exact[qIdx], res)
		}
		b.StopTimer()
		b.ReportMetric(recallSum/float64(b.N)*100, "recall@10_%")
	})
}

func BenchmarkHNSWBuild_10k_384d(b *testing.B) {
	numDocs := 10_000
	dim := 384

	hnswM := annBenchEnvInt("CHROMEM_ANN_BENCH_HNSW_M", 16)
	hnswEFC := annBenchEnvInt("CHROMEM_ANN_BENCH_HNSW_EFC", 200)
	hnswEFS := annBenchEnvInt("CHROMEM_ANN_BENCH_HNSW_EFS", 100)

	docs, _ := annBenchDataset(numDocs, dim, 1)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := newHNSWIndex(dim, hnswM, hnswEFC, hnswEFS)
		if err := idx.Build(docs); err != nil {
			b.Fatalf("build hnsw: %v", err)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(numDocs), "docs/op")
}
