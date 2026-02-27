package chromem

import (
	"cmp"
	"container/heap"
	"fmt"
	"math/bits"
	"math/rand"
	"slices"
)

// hnswNeighbor represents a search result with its similarity score.
type hnswNeighbor struct {
	doc        *Document
	similarity float32
}

// hnswNode stores one document in the HNSW graph.
//
// neighbors is indexed by level and contains node IDs connected at that level.
// Level 0 is the densest/base layer, higher levels are progressively sparser.
type hnswNode struct {
	doc       *Document
	level     int
	neighbors [][]int
}

// hnswIndex is an in-memory Hierarchical Navigable Small World graph index.
//
// It is used for approximate nearest-neighbor search over normalized embeddings
// using dot-product similarity.
type hnswIndex struct {
	// dim is the embedding dimension expected for all indexed/query vectors.
	dim int
	// m is the maximum number of neighbors kept per node per level.
	m int
	// efConstruction controls candidate breadth during insertion/build.
	efConstruction int
	// efSearch controls candidate breadth during querying.
	efSearch int

	// nodes stores all graph nodes; node ID equals its index in this slice.
	nodes []hnswNode
	// deletedBitmap marks tombstoned nodes in a compact bitset form.
	// Bit i corresponds to node i.
	deletedBitmap []uint64
	// latestNodeByDocID points each live document ID to its most recent node.
	latestNodeByDocID map[string]int
	// entryPoint is the current top-layer entry node ID, or -1 when empty.
	entryPoint int
	// maxLevel is the highest level currently present in the graph.
	maxLevel int
	// rng is used to sample levels for newly inserted nodes.
	rng *rand.Rand
}

func (h *hnswIndex) clone() *hnswIndex {
	if h == nil {
		return nil
	}

	nodes := make([]hnswNode, len(h.nodes))
	for i, node := range h.nodes {
		neighbors := make([][]int, len(node.neighbors))
		for level := range node.neighbors {
			neighbors[level] = slices.Clone(node.neighbors[level])
		}
		nodes[i] = hnswNode{
			doc:       node.doc,
			level:     node.level,
			neighbors: neighbors,
		}
	}

	return &hnswIndex{
		dim:               h.dim,
		m:                 h.m,
		efConstruction:    h.efConstruction,
		efSearch:          h.efSearch,
		nodes:             nodes,
		deletedBitmap:     slices.Clone(h.deletedBitmap),
		latestNodeByDocID: mapsClone(h.latestNodeByDocID),
		entryPoint:        h.entryPoint,
		maxLevel:          h.maxLevel,
		rng:               rand.New(rand.NewSource(h.rng.Int63())),
	}
}

func mapsClone[K comparable, V any](m map[K]V) map[K]V {
	if m == nil {
		return nil
	}
	out := make(map[K]V, len(m))
	for key, value := range m {
		out[key] = value
	}
	return out
}

// newHNSWIndex creates an index with sane defaults for invalid/small values.
//
// If m < 2, m defaults to 16.
// If efConstruction < m, efConstruction defaults to m*8.
// If efSearch < m, efSearch defaults to m*4.
func newHNSWIndex(dim, m, efConstruction, efSearch int) *hnswIndex {
	if m < 2 {
		m = 16
	}
	if efConstruction < m {
		efConstruction = m * 8
	}
	if efSearch < m {
		efSearch = m * 4
	}

	return &hnswIndex{
		dim:            dim,
		m:              m,
		efConstruction: efConstruction,
		efSearch:       efSearch,
		entryPoint:     -1,
		maxLevel:       -1,
		rng:            rand.New(rand.NewSource(42)),
	}
}

// Build rebuilds the HNSW graph from the provided documents.
//
// Existing graph state is discarded. Every document must have a non-empty
// embedding with dimension equal to index dim.
func (h *hnswIndex) Build(docs []*Document) error {
	h.nodes = h.nodes[:0]
	h.deletedBitmap = h.deletedBitmap[:0]
	h.latestNodeByDocID = make(map[string]int, len(docs))
	h.entryPoint = -1
	h.maxLevel = -1

	for _, doc := range docs {
		if len(doc.Embedding) == 0 {
			return fmt.Errorf("document '%s' embedding is empty", doc.ID)
		}
		if len(doc.Embedding) != h.dim {
			return fmt.Errorf("document '%s' embedding has dimension %d, expected %d", doc.ID, len(doc.Embedding), h.dim)
		}
		if err := h.insert(doc); err != nil {
			return err
		}
	}

	return nil
}

// Search returns up to k nearest neighbors for query using approximate search.
//
// The query embedding dimension must match index dim. Results are ordered by
// descending similarity.
func (h *hnswIndex) Search(query []float32, k int) ([]hnswNeighbor, error) {
	if len(query) != h.dim {
		return nil, fmt.Errorf("query embedding has dimension %d, expected %d", len(query), h.dim)
	}
	if k <= 0 || len(h.nodes) == 0 {
		return nil, nil
	}
	if len(h.latestNodeByDocID) == 0 {
		return nil, nil
	}

	entry := h.entryPoint
	if entry < 0 {
		return nil, nil
	}

	current := entry
	currentSim := dotProductOptimized(query, h.nodes[current].doc.Embedding)
	for level := h.maxLevel; level > 0; level-- {
		improved := true
		for improved {
			improved = false
			for _, nid := range h.nodes[current].neighbors[level] {
				sim := dotProductOptimized(query, h.nodes[nid].doc.Embedding)
				if sim > currentSim {
					current = nid
					currentSim = sim
					improved = true
				}
			}
		}
	}

	ef := max(max(k, h.efSearch), k*4)
	candidates := h.searchLayer(query, []int{current}, ef, 0)
	if len(candidates) == 0 {
		return nil, nil
	}

	neighbors := make([]hnswNeighbor, 0, min(k, len(candidates)))
	for i := 0; i < len(candidates) && len(neighbors) < k; i++ {
		if h.isDeleted(candidates[i].id) {
			continue
		}
		node := h.nodes[candidates[i].id]
		neighbors = append(neighbors, hnswNeighbor{doc: node.doc, similarity: candidates[i].similarity})
	}

	return neighbors, nil
}

// insert adds one document to the graph.
//
// The algorithm first greedily descends from upper layers to find a good entry
// region, then performs a broader search/link procedure on each layer down to 0.
func (h *hnswIndex) insert(doc *Document) error {
	level := h.randomLevel()
	nodeID := len(h.nodes)
	h.nodes = append(h.nodes, hnswNode{
		doc:       doc,
		level:     level,
		neighbors: make([][]int, level+1),
	})
	h.ensureDeletedBitmapSize(len(h.nodes))
	if h.latestNodeByDocID == nil {
		h.latestNodeByDocID = make(map[string]int)
	}
	h.latestNodeByDocID[doc.ID] = nodeID

	if h.entryPoint == -1 {
		h.entryPoint = nodeID
		h.maxLevel = level
		return nil
	}

	query := doc.Embedding
	ep := h.entryPoint

	for l := h.maxLevel; l > level; l-- {
		best := h.searchLayer(query, []int{ep}, 1, l)
		if len(best) > 0 {
			ep = best[0].id
		}
	}

	limitLevel := min(level, h.maxLevel)
	for l := limitLevel; l >= 0; l-- {
		candidates := h.searchLayer(query, []int{ep}, h.efConstruction, l)
		selected := h.selectNeighbors(candidates, h.m)
		for _, neighborID := range selected {
			h.link(nodeID, neighborID, l)
			h.link(neighborID, nodeID, l)
		}
		if len(candidates) > 0 {
			ep = candidates[0].id
		}
	}

	if level > h.maxLevel {
		h.entryPoint = nodeID
		h.maxLevel = level
	}

	return nil
}

func (h *hnswIndex) upsert(doc *Document) error {
	if h.latestNodeByDocID != nil {
		if oldNodeID, ok := h.latestNodeByDocID[doc.ID]; ok {
			h.setDeleted(oldNodeID, true)
		}
	}
	return h.insert(doc)
}

func (h *hnswIndex) markDeleted(docID string) bool {
	if h.latestNodeByDocID == nil {
		return false
	}
	nodeID, ok := h.latestNodeByDocID[docID]
	if !ok {
		return false
	}
	h.setDeleted(nodeID, true)
	delete(h.latestNodeByDocID, docID)
	return true
}

func (h *hnswIndex) deletedCount() int {
	count := 0
	for _, word := range h.deletedBitmap {
		count += bits.OnesCount64(word)
	}
	return count
}

func (h *hnswIndex) ensureDeletedBitmapSize(nodeCount int) {
	needed := (nodeCount + 63) / 64
	if len(h.deletedBitmap) >= needed {
		return
	}
	h.deletedBitmap = append(h.deletedBitmap, make([]uint64, needed-len(h.deletedBitmap))...)
}

func (h *hnswIndex) setDeleted(nodeID int, deleted bool) {
	if nodeID < 0 {
		return
	}
	h.ensureDeletedBitmapSize(nodeID + 1)
	wordIndex := nodeID / 64
	bitOffset := uint(nodeID % 64)
	mask := uint64(1) << bitOffset
	if deleted {
		h.deletedBitmap[wordIndex] |= mask
	} else {
		h.deletedBitmap[wordIndex] &^= mask
	}
}

func (h *hnswIndex) isDeleted(nodeID int) bool {
	if nodeID < 0 {
		return false
	}
	wordIndex := nodeID / 64
	if wordIndex >= len(h.deletedBitmap) {
		return false
	}
	bitOffset := uint(nodeID % 64)
	return (h.deletedBitmap[wordIndex] & (uint64(1) << bitOffset)) != 0
}

// link adds a directed edge from fromID to toID on the given level.
//
// Duplicate edges are ignored. If the neighbor list exceeds m, neighbors are
// trimmed to the top-m most similar nodes to fromID.
func (h *hnswIndex) link(fromID, toID, level int) {
	neighbors := h.nodes[fromID].neighbors[level]
	if slices.Contains(neighbors, toID) {
		return
	}
	neighbors = append(neighbors, toID)

	if len(neighbors) > h.m {
		baseVec := h.nodes[fromID].doc.Embedding
		slices.SortFunc(neighbors, func(a, b int) int {
			sa := dotProductOptimized(baseVec, h.nodes[a].doc.Embedding)
			sb := dotProductOptimized(baseVec, h.nodes[b].doc.Embedding)
			return cmp.Compare(sb, sa)
		})
		neighbors = neighbors[:h.m]
	}

	h.nodes[fromID].neighbors[level] = neighbors
}

// selectNeighbors picks up to limit best candidates for query by similarity.
//
// Returned IDs are ordered by descending similarity.
func (h *hnswIndex) selectNeighbors(candidates []hnswCandidate, limit int) []int {
	if len(candidates) <= limit {
		ids := make([]int, 0, len(candidates))
		for _, candidate := range candidates {
			ids = append(ids, candidate.id)
		}
		return ids
	}

	slices.SortFunc(candidates, func(a, b hnswCandidate) int {
		return cmp.Compare(b.similarity, a.similarity)
	})
	ids := make([]int, 0, limit)
	for i := 0; i < limit; i++ {
		ids = append(ids, candidates[i].id)
	}
	return ids
}

// hnswMinHeap keeps the current best set as a min-heap by similarity.
//
// The heap root is the worst item in the best set, enabling O(log n)
// replacement when a better candidate is found.
type hnswMinHeap []hnswCandidate

func (h hnswMinHeap) Len() int           { return len(h) }
func (h hnswMinHeap) Less(i, j int) bool { return h[i].similarity < h[j].similarity }
func (h hnswMinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *hnswMinHeap) Push(x any) {
	*h = append(*h, x.(hnswCandidate))
}

func (h *hnswMinHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// hnswMaxHeap stores exploration candidates as a max-heap by similarity.
//
// The next expanded candidate is always the most promising known one.
type hnswMaxHeap []hnswCandidate

func (h hnswMaxHeap) Len() int           { return len(h) }
func (h hnswMaxHeap) Less(i, j int) bool { return h[i].similarity > h[j].similarity }
func (h hnswMaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *hnswMaxHeap) Push(x any) {
	*h = append(*h, x.(hnswCandidate))
}

func (h *hnswMaxHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// hnswCandidate is a node considered during layer search.
type hnswCandidate struct {
	id         int
	similarity float32
}

// searchLayer performs the HNSW best-first search within a single level.
//
// entryPoints seeds the search frontier. ef bounds the size of the maintained
// best set. Returned node IDs are sorted by descending similarity.
func (h *hnswIndex) searchLayer(query []float32, entryPoints []int, ef int, level int) []hnswCandidate {
	visited := make(map[int]struct{}, ef*2)
	best := make(hnswMinHeap, 0, ef)
	candidates := make(hnswMaxHeap, 0, ef)

	for _, ep := range entryPoints {
		if ep < 0 || ep >= len(h.nodes) || level > h.nodes[ep].level {
			continue
		}
		sim := dotProductOptimized(query, h.nodes[ep].doc.Embedding)
		cand := hnswCandidate{id: ep, similarity: sim}
		heap.Push(&best, cand)
		heap.Push(&candidates, cand)
		visited[ep] = struct{}{}
	}

	if len(candidates) == 0 {
		return nil
	}

	for len(candidates) > 0 {
		current := heap.Pop(&candidates).(hnswCandidate)
		worstBest := best[0]
		if len(best) >= ef && current.similarity < worstBest.similarity {
			break
		}

		for _, nid := range h.nodes[current.id].neighbors[level] {
			if _, ok := visited[nid]; ok {
				continue
			}
			visited[nid] = struct{}{}

			sim := dotProductOptimized(query, h.nodes[nid].doc.Embedding)
			candidate := hnswCandidate{id: nid, similarity: sim}

			if len(best) < ef || sim > best[0].similarity {
				heap.Push(&candidates, candidate)
				heap.Push(&best, candidate)
				if len(best) > ef {
					heap.Pop(&best)
				}
			}
		}
	}

	res := make([]hnswCandidate, 0, len(best))
	for _, item := range best {
		res = append(res, item)
	}

	slices.SortFunc(res, func(a, b hnswCandidate) int {
		return cmp.Compare(b.similarity, a.similarity)
	})

	return res
}

// randomLevel samples a layer level for a new node.
//
// This implementation uses a geometric distribution with p=0.5 and a hard cap
// of 32 levels.
func (h *hnswIndex) randomLevel() int {
	level := 0
	for level < 32 && h.rng.Float64() < 0.5 {
		level++
	}
	return level
}
