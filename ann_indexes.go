package chromem

import (
	"cmp"
	"container/heap"
	"errors"
	"fmt"
	"math"
	"slices"
)

// annTopKHeap is a fixed-size min-heap used to keep top-k highest similarities.
// The heap root is the current worst item in the kept set.
type annTopKHeap []hnswNeighbor

func (h annTopKHeap) Len() int           { return len(h) }
func (h annTopKHeap) Less(i, j int) bool { return h[i].similarity < h[j].similarity }
func (h annTopKHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *annTopKHeap) Push(x any) {
	*h = append(*h, x.(hnswNeighbor))
}

func (h *annTopKHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func addTopK(topK *annTopKHeap, n hnswNeighbor, k int) {
	if k <= 0 {
		return
	}
	if topK.Len() < k {
		heap.Push(topK, n)
		return
	}
	if (*topK)[0].similarity < n.similarity {
		heap.Pop(topK)
		heap.Push(topK, n)
	}
}

func sortedTopK(topK annTopKHeap) []hnswNeighbor {
	out := make([]hnswNeighbor, 0, len(topK))
	for _, item := range topK {
		out = append(out, item)
	}
	slices.SortFunc(out, func(a, b hnswNeighbor) int {
		return cmp.Compare(b.similarity, a.similarity)
	})
	return out
}

// kMeansCosine runs lightweight k-means on normalized vectors using
// cosine/dot-product assignment. It is intentionally simple and deterministic
// enough for index training in this package.
func kMeansCosine(vectors [][]float32, k, maxIter int) [][]float32 {
	if len(vectors) == 0 || k <= 0 {
		return nil
	}
	if k > len(vectors) {
		k = len(vectors)
	}
	if maxIter <= 0 {
		maxIter = 20
	}

	dim := len(vectors[0])
	centroids := make([][]float32, k)
	step := max(1, len(vectors)/k)
	for i := 0; i < k; i++ {
		idx := min(i*step, len(vectors)-1)
		centroids[i] = slices.Clone(vectors[idx])
	}

	assign := make([]int, len(vectors))
	for iter := 0; iter < maxIter; iter++ {
		changed := false

		for i, vec := range vectors {
			bestC := 0
			bestSim := float32(-2)
			for c, centroid := range centroids {
				sim := dotProductOptimized(vec, centroid)
				if sim > bestSim {
					bestSim = sim
					bestC = c
				}
			}
			if assign[i] != bestC {
				assign[i] = bestC
				changed = true
			}
		}

		sums := make([][]float32, k)
		counts := make([]int, k)
		for c := range k {
			sums[c] = make([]float32, dim)
		}
		for i, vec := range vectors {
			c := assign[i]
			counts[c]++
			for d, v := range vec {
				sums[c][d] += v
			}
		}

		for c := range k {
			if counts[c] == 0 {
				centroids[c] = slices.Clone(vectors[(iter+c)%len(vectors)])
				continue
			}
			inv := float32(1.0 / float64(counts[c]))
			for d := range sums[c] {
				sums[c][d] *= inv
			}
			if !isNormalized(sums[c]) {
				sums[c] = normalizeVector(sums[c])
			}
			centroids[c] = sums[c]
		}

		if !changed {
			break
		}
	}

	return centroids
}

// ivfIndex implements IVF (Inverted File Index) with k-means clustering.
// It accelerates search by scanning only the most relevant clusters.
type ivfIndex struct {
	dim     int
	nlist   int
	nprobe  int
	docs    []*Document
	vectors [][]float32
	centers [][]float32
	lists   [][]int
}

func newIVFIndex(dim, nlist, nprobe int) *ivfIndex {
	if nlist <= 0 {
		nlist = 64
	}
	if nprobe <= 0 {
		nprobe = max(1, int(math.Sqrt(float64(nlist))))
	}
	if nprobe > nlist {
		nprobe = nlist
	}
	return &ivfIndex{dim: dim, nlist: nlist, nprobe: nprobe}
}

func (idx *ivfIndex) Build(docs []*Document) error {
	idx.docs = docs
	idx.vectors = idx.vectors[:0]
	for _, doc := range docs {
		if len(doc.Embedding) != idx.dim {
			return fmt.Errorf("document '%s' embedding has dimension %d, expected %d", doc.ID, len(doc.Embedding), idx.dim)
		}
		idx.vectors = append(idx.vectors, doc.Embedding)
	}
	if len(idx.vectors) == 0 {
		idx.centers = nil
		idx.lists = nil
		return nil
	}

	k := min(idx.nlist, len(idx.vectors))
	idx.centers = kMeansCosine(idx.vectors, k, 20)
	idx.lists = make([][]int, len(idx.centers))
	for i, vec := range idx.vectors {
		cid := idx.nearestCenter(vec)
		idx.lists[cid] = append(idx.lists[cid], i)
	}
	return nil
}

func (idx *ivfIndex) nearestCenter(vec []float32) int {
	best := 0
	bestSim := float32(-2)
	for i, center := range idx.centers {
		sim := dotProductOptimized(vec, center)
		if sim > bestSim {
			best = i
			bestSim = sim
		}
	}
	return best
}

func (idx *ivfIndex) Search(query []float32, k int) ([]hnswNeighbor, error) {
	if len(query) != idx.dim {
		return nil, fmt.Errorf("query embedding has dimension %d, expected %d", len(query), idx.dim)
	}
	if k <= 0 || len(idx.docs) == 0 || len(idx.centers) == 0 {
		return nil, nil
	}

	type centerScore struct {
		id  int
		sim float32
	}
	cands := make([]centerScore, 0, len(idx.centers))
	for i, center := range idx.centers {
		cands = append(cands, centerScore{id: i, sim: dotProductOptimized(query, center)})
	}
	slices.SortFunc(cands, func(a, b centerScore) int {
		return cmp.Compare(b.sim, a.sim)
	})

	nprobe := min(idx.nprobe, len(cands))
	top := make(annTopKHeap, 0, k)
	for i := range nprobe {
		for _, docIdx := range idx.lists[cands[i].id] {
			doc := idx.docs[docIdx]
			sim := dotProductOptimized(query, doc.Embedding)
			addTopK(&top, hnswNeighbor{doc: doc, similarity: sim}, k)
		}
	}

	return sortedTopK(top), nil
}

// pqIndex implements Product Quantization for compressed ANN search.
// It compresses vectors into subspace centroid codes.
type pqIndex struct {
	dim      int
	m        int
	nbits    int
	ksub     int
	dsub     int
	docs     []*Document
	codebook [][]float32
	codes    [][]uint16
}

func newPQIndex(dim, m, nbits int) *pqIndex {
	if m <= 0 {
		m = 8
	}
	if dim%m != 0 {
		for cand := m; cand >= 2; cand-- {
			if dim%cand == 0 {
				m = cand
				break
			}
		}
		if dim%m != 0 {
			m = 1
		}
	}
	if nbits <= 0 || nbits > 8 {
		nbits = 8
	}
	return &pqIndex{
		dim:   dim,
		m:     m,
		nbits: nbits,
		ksub:  1 << nbits,
		dsub:  dim / m,
	}
}

func (idx *pqIndex) Build(docs []*Document) error {
	idx.docs = docs
	idx.codes = make([][]uint16, len(docs))
	if len(docs) == 0 {
		idx.codebook = nil
		return nil
	}

	for _, doc := range docs {
		if len(doc.Embedding) != idx.dim {
			return fmt.Errorf("document '%s' embedding has dimension %d, expected %d", doc.ID, len(doc.Embedding), idx.dim)
		}
	}

	idx.codebook = make([][]float32, idx.m)
	for sub := 0; sub < idx.m; sub++ {
		subVectors := make([][]float32, len(docs))
		start := sub * idx.dsub
		end := start + idx.dsub
		for i, doc := range docs {
			subVectors[i] = doc.Embedding[start:end]
		}

		subCenters := kMeansCosine(subVectors, min(idx.ksub, len(subVectors)), 20)
		if len(subCenters) == 0 {
			return errors.New("pq training failed: empty subspace codebook")
		}

		flat := make([]float32, len(subCenters)*idx.dsub)
		for i, center := range subCenters {
			copy(flat[i*idx.dsub:(i+1)*idx.dsub], center)
		}
		idx.codebook[sub] = flat
	}

	for i, doc := range docs {
		idx.codes[i] = idx.encode(doc.Embedding)
	}
	return nil
}

func (idx *pqIndex) encode(vector []float32) []uint16 {
	code := make([]uint16, idx.m)
	for sub := 0; sub < idx.m; sub++ {
		start := sub * idx.dsub
		end := start + idx.dsub
		subVec := vector[start:end]
		book := idx.codebook[sub]
		k := len(book) / idx.dsub
		best := 0
		bestSim := float32(-2)
		for c := range k {
			centroid := book[c*idx.dsub : (c+1)*idx.dsub]
			sim := dotProductOptimized(subVec, centroid)
			if sim > bestSim {
				bestSim = sim
				best = c
			}
		}
		code[sub] = uint16(best)
	}
	return code
}

func (idx *pqIndex) Search(query []float32, k int) ([]hnswNeighbor, error) {
	if len(query) != idx.dim {
		return nil, fmt.Errorf("query embedding has dimension %d, expected %d", len(query), idx.dim)
	}
	if k <= 0 || len(idx.docs) == 0 {
		return nil, nil
	}

	tables := make([][]float32, idx.m)
	for sub := 0; sub < idx.m; sub++ {
		start := sub * idx.dsub
		end := start + idx.dsub
		subQuery := query[start:end]
		book := idx.codebook[sub]
		cCount := len(book) / idx.dsub
		tables[sub] = make([]float32, cCount)
		for c := range cCount {
			centroid := book[c*idx.dsub : (c+1)*idx.dsub]
			tables[sub][c] = dotProductOptimized(subQuery, centroid)
		}
	}

	top := make(annTopKHeap, 0, k)
	for i, code := range idx.codes {
		sim := float32(0)
		for sub, c := range code {
			sim += tables[sub][int(c)]
		}
		addTopK(&top, hnswNeighbor{doc: idx.docs[i], similarity: sim}, k)
	}

	return sortedTopK(top), nil
}

// ivfPQCode stores one IVFPQ-compressed residual and its document index.
type ivfPQCode struct {
	docIndex int
	code     []uint16
}

// ivfpqIndex implements IVFPQ (IVF + residual PQ).
// It provides coarse partitioning plus aggressive compression.
type ivfpqIndex struct {
	dim      int
	nlist    int
	nprobe   int
	m        int
	nbits    int
	ksub     int
	dsub     int
	docs     []*Document
	centers  [][]float32
	codebook [][]float32
	lists    [][]ivfPQCode
}

func newIVFPQIndex(dim, nlist, nprobe, m, nbits int) *ivfpqIndex {
	if m <= 0 {
		m = 8
	}
	if dim%m != 0 {
		for cand := m; cand >= 2; cand-- {
			if dim%cand == 0 {
				m = cand
				break
			}
		}
		if dim%m != 0 {
			m = 1
		}
	}
	if nlist <= 0 {
		nlist = 64
	}
	if nprobe <= 0 {
		nprobe = max(1, int(math.Sqrt(float64(nlist))))
	}
	if nprobe > nlist {
		nprobe = nlist
	}
	if nbits <= 0 || nbits > 8 {
		nbits = 8
	}

	return &ivfpqIndex{
		dim:    dim,
		nlist:  nlist,
		nprobe: nprobe,
		m:      m,
		nbits:  nbits,
		ksub:   1 << nbits,
		dsub:   dim / m,
	}
}

func (idx *ivfpqIndex) Build(docs []*Document) error {
	idx.docs = docs
	if len(docs) == 0 {
		idx.centers = nil
		idx.codebook = nil
		idx.lists = nil
		return nil
	}
	for _, doc := range docs {
		if len(doc.Embedding) != idx.dim {
			return fmt.Errorf("document '%s' embedding has dimension %d, expected %d", doc.ID, len(doc.Embedding), idx.dim)
		}
	}

	vectors := make([][]float32, len(docs))
	for i, doc := range docs {
		vectors[i] = doc.Embedding
	}

	k := min(idx.nlist, len(vectors))
	idx.centers = kMeansCosine(vectors, k, 20)
	assign := make([]int, len(vectors))
	for i, vec := range vectors {
		assign[i] = idx.nearestCenter(vec)
	}

	residuals := make([][]float32, len(vectors))
	for i, vec := range vectors {
		center := idx.centers[assign[i]]
		r := make([]float32, idx.dim)
		for d := range vec {
			r[d] = vec[d] - center[d]
		}
		residuals[i] = r
	}

	idx.codebook = make([][]float32, idx.m)
	for sub := 0; sub < idx.m; sub++ {
		subVectors := make([][]float32, len(residuals))
		start := sub * idx.dsub
		end := start + idx.dsub
		for i, residual := range residuals {
			subVectors[i] = residual[start:end]
		}
		subCenters := kMeansCosine(subVectors, min(idx.ksub, len(subVectors)), 20)
		if len(subCenters) == 0 {
			return errors.New("ivfpq training failed: empty residual codebook")
		}
		flat := make([]float32, len(subCenters)*idx.dsub)
		for i, center := range subCenters {
			copy(flat[i*idx.dsub:(i+1)*idx.dsub], center)
		}
		idx.codebook[sub] = flat
	}

	idx.lists = make([][]ivfPQCode, len(idx.centers))
	for i, residual := range residuals {
		cid := assign[i]
		idx.lists[cid] = append(idx.lists[cid], ivfPQCode{docIndex: i, code: idx.encodeResidual(residual)})
	}
	return nil
}

func (idx *ivfpqIndex) nearestCenter(vec []float32) int {
	best := 0
	bestSim := float32(-2)
	for i, center := range idx.centers {
		sim := dotProductOptimized(vec, center)
		if sim > bestSim {
			best = i
			bestSim = sim
		}
	}
	return best
}

func (idx *ivfpqIndex) encodeResidual(residual []float32) []uint16 {
	code := make([]uint16, idx.m)
	for sub := 0; sub < idx.m; sub++ {
		start := sub * idx.dsub
		end := start + idx.dsub
		subVec := residual[start:end]
		book := idx.codebook[sub]
		k := len(book) / idx.dsub
		best := 0
		bestSim := float32(-2)
		for c := range k {
			centroid := book[c*idx.dsub : (c+1)*idx.dsub]
			sim := dotProductOptimized(subVec, centroid)
			if sim > bestSim {
				bestSim = sim
				best = c
			}
		}
		code[sub] = uint16(best)
	}
	return code
}

func (idx *ivfpqIndex) Search(query []float32, k int) ([]hnswNeighbor, error) {
	if len(query) != idx.dim {
		return nil, fmt.Errorf("query embedding has dimension %d, expected %d", len(query), idx.dim)
	}
	if k <= 0 || len(idx.docs) == 0 || len(idx.centers) == 0 {
		return nil, nil
	}

	type centerScore struct {
		id  int
		sim float32
	}
	centers := make([]centerScore, 0, len(idx.centers))
	for i, center := range idx.centers {
		centers = append(centers, centerScore{id: i, sim: dotProductOptimized(query, center)})
	}
	slices.SortFunc(centers, func(a, b centerScore) int {
		return cmp.Compare(b.sim, a.sim)
	})

	nprobe := min(idx.nprobe, len(centers))
	top := make(annTopKHeap, 0, k)
	for i := range nprobe {
		cid := centers[i].id
		center := idx.centers[cid]

		residualQuery := make([]float32, idx.dim)
		for d := 0; d < idx.dim; d++ {
			residualQuery[d] = query[d] - center[d]
		}

		tables := make([][]float32, idx.m)
		for sub := 0; sub < idx.m; sub++ {
			start := sub * idx.dsub
			end := start + idx.dsub
			subQ := residualQuery[start:end]
			book := idx.codebook[sub]
			kSub := len(book) / idx.dsub
			tables[sub] = make([]float32, kSub)
			for c := range kSub {
				centroid := book[c*idx.dsub : (c+1)*idx.dsub]
				tables[sub][c] = dotProductOptimized(subQ, centroid)
			}
		}

		baseSim := dotProductOptimized(query, center)
		for _, item := range idx.lists[cid] {
			sim := baseSim
			for sub, code := range item.code {
				sim += tables[sub][int(code)]
			}
			addTopK(&top, hnswNeighbor{doc: idx.docs[item.docIndex], similarity: sim}, k)
		}
	}

	return sortedTopK(top), nil
}
