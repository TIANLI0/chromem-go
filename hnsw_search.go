package chromem

import (
	"container/heap"
	"fmt"
	"runtime"
	"slices"
	"sync"
)

var hnswVisitedPool = sync.Pool{
	New: func() any {
		return &hnswVisitedState{}
	},
}

// acquireHNSWVisitedState returns a reusable visited state with at least size slots.
func acquireHNSWVisitedState(size int) *hnswVisitedState {
	state := hnswVisitedPool.Get().(*hnswVisitedState)
	if cap(state.marks) < size {
		state.marks = make([]uint32, size)
	} else {
		state.marks = state.marks[:size]
	}

	state.epoch++
	if state.epoch == 0 {
		clear(state.marks)
		state.epoch = 1
	}

	return state
}

// releaseHNSWVisitedState returns a visited state to the pool.
func releaseHNSWVisitedState(state *hnswVisitedState) {
	if state == nil {
		return
	}
	hnswVisitedPool.Put(state)
}

// hnswCandidate is a node considered during layer search.
type hnswCandidate struct {
	id         int
	similarity float32
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

// Heap pools reduce allocations in hot search paths.
var hnswMinHeapPool = sync.Pool{
	New: func() any {
		h := make(hnswMinHeap, 0, 64)
		return &h
	},
}

var hnswMaxHeapPool = sync.Pool{
	New: func() any {
		h := make(hnswMaxHeap, 0, 64)
		return &h
	},
}

// acquireHNSWMinHeap returns a cleared min-heap buffer with requested capacity.
func acquireHNSWMinHeap(capacity int) *hnswMinHeap {
	h := hnswMinHeapPool.Get().(*hnswMinHeap)
	if cap(*h) < capacity {
		*h = make(hnswMinHeap, 0, capacity)
	} else {
		*h = (*h)[:0]
	}
	return h
}

// releaseHNSWMinHeap returns a min-heap buffer to the pool.
func releaseHNSWMinHeap(h *hnswMinHeap) {
	if h == nil {
		return
	}
	*h = (*h)[:0]
	hnswMinHeapPool.Put(h)
}

// acquireHNSWMaxHeap returns a cleared max-heap buffer with requested capacity.
func acquireHNSWMaxHeap(capacity int) *hnswMaxHeap {
	h := hnswMaxHeapPool.Get().(*hnswMaxHeap)
	if cap(*h) < capacity {
		*h = make(hnswMaxHeap, 0, capacity)
	} else {
		*h = (*h)[:0]
	}
	return h
}

// releaseHNSWMaxHeap returns a max-heap buffer to the pool.
func releaseHNSWMaxHeap(h *hnswMaxHeap) {
	if h == nil {
		return
	}
	*h = (*h)[:0]
	hnswMaxHeapPool.Put(h)
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

	start := h.searchStartNode(query)
	if start < 0 {
		return nil, nil
	}

	entryPoints := h.collectSearchEntryPoints(query, start, max(h.m, 8))
	if len(h.shardEntryPoints) > 1 {
		seedCap := max(max(h.m*2, 16), len(h.shardEntryPoints))
		merged := make([]int, 0, len(entryPoints)+len(h.shardEntryPoints)+1)
		seen := make(map[int]struct{}, len(entryPoints)+len(h.shardEntryPoints)+1)

		addSeed := func(id int) {
			if id < 0 || id >= len(h.nodes) {
				return
			}
			if _, ok := seen[id]; ok {
				return
			}
			seen[id] = struct{}{}
			merged = append(merged, id)
		}

		addSeed(start)
		for _, id := range entryPoints {
			addSeed(id)
		}
		for _, id := range h.shardEntryPoints {
			if len(merged) >= seedCap {
				break
			}
			addSeed(id)
		}

		if len(merged) > 0 {
			entryPoints = merged
		}
	}
	if len(entryPoints) == 0 {
		entryPoints = []int{start}
	}

	ef := max(max(k, h.efSearch), k*4)
	if len(h.shardEntryPoints) > 1 {
		ef = max(ef, h.efSearch*len(h.shardEntryPoints))
		ef = min(ef, len(h.nodes))
	}
	candidates := h.searchLayer(query, entryPoints, ef, 0)
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

// collectSearchEntryPoints collects several strong level-0 seeds around start.
// These seeds improve recall for merged-shard or heterogeneous neighborhoods.
func (h *hnswIndex) collectSearchEntryPoints(query []float32, start, maxEntries int) []int {
	if start < 0 || start >= len(h.nodes) {
		return nil
	}
	if maxEntries <= 0 {
		maxEntries = 1
	}

	type scoredID struct {
		id  int
		sim float32
	}

	seen := make(map[int]struct{}, maxEntries*4)
	pool := make([]scoredID, 0, maxEntries*4)
	candidateIDs := make([]int, 0, maxEntries*4)
	pushIfNew := func(id int) {
		if id < 0 || id >= len(h.nodes) {
			return
		}
		if _, ok := seen[id]; ok {
			return
		}
		seen[id] = struct{}{}
		candidateIDs = append(candidateIDs, id)
	}

	pushIfNew(start)
	for _, nid := range h.nodes[start].neighbors[0] {
		pushIfNew(nid)
	}

	for _, nid := range h.nodes[start].neighbors[0] {
		for _, nnid := range h.nodes[nid].neighbors[0] {
			if len(candidateIDs) >= maxEntries*4 {
				break
			}
			pushIfNew(nnid)
		}
		if len(candidateIDs) >= maxEntries*4 {
			break
		}
	}

	scored := h.scoreNodeIDs(query, candidateIDs)
	for _, item := range scored {
		pool = append(pool, scoredID{id: item.id, sim: item.similarity})
	}

	slices.SortFunc(pool, func(a, b scoredID) int {
		if a.sim > b.sim {
			return -1
		}
		if a.sim < b.sim {
			return 1
		}
		return 0
	})

	if len(pool) > maxEntries {
		pool = pool[:maxEntries]
	}

	out := make([]int, 0, len(pool))
	for _, item := range pool {
		out = append(out, item.id)
	}

	return out
}

// searchLayer performs the HNSW best-first search within a single level.
//
// entryPoints seeds the search frontier. ef bounds the size of the maintained
// best set. Returned node IDs are sorted by descending similarity.
func (h *hnswIndex) searchLayer(query []float32, entryPoints []int, ef int, level int) []hnswCandidate {
	if ef <= 0 || len(h.nodes) == 0 {
		return nil
	}

	visitedState := acquireHNSWVisitedState(len(h.nodes))
	defer releaseHNSWVisitedState(visitedState)
	visited := visitedState.marks
	epoch := visitedState.epoch
	best := acquireHNSWMinHeap(ef)
	defer releaseHNSWMinHeap(best)
	candidates := acquireHNSWMaxHeap(max(ef, len(entryPoints)))
	defer releaseHNSWMaxHeap(candidates)

	for _, ep := range entryPoints {
		if ep < 0 || ep >= len(h.nodes) || level > h.nodes[ep].level {
			continue
		}
		if visited[ep] == epoch {
			continue
		}
		visited[ep] = epoch
		sim := hnswDot(query, h.nodeEmbedding(ep))
		cand := hnswCandidate{id: ep, similarity: sim}
		heap.Push(candidates, cand)
		if !h.isDeleted(ep) {
			heap.Push(best, cand)
		}
	}

	if len(*candidates) == 0 {
		return nil
	}

	for len(*candidates) > 0 {
		current := heap.Pop(candidates).(hnswCandidate)
		if len(*best) >= ef && current.similarity < (*best)[0].similarity {
			break
		}
		if level > h.nodes[current.id].level {
			continue
		}

		for _, nid := range h.nodes[current.id].neighbors[level] {
			if nid < 0 || nid >= len(h.nodes) {
				continue
			}
			if visited[nid] == epoch {
				continue
			}
			visited[nid] = epoch
			candidate := hnswCandidate{id: nid, similarity: hnswDot(query, h.nodeEmbedding(nid))}
			worstBestSim := float32(-2)
			if len(*best) > 0 {
				worstBestSim = (*best)[0].similarity
			}
			shouldExpand := len(*best) < ef || candidate.similarity > worstBestSim

			if shouldExpand || h.isDeleted(candidate.id) {
				heap.Push(candidates, candidate)
			}
			if !h.isDeleted(candidate.id) && shouldExpand {
				heap.Push(best, candidate)
				if len(*best) > ef {
					heap.Pop(best)
				}
			}
		}
	}

	if len(*best) == 0 {
		return nil
	}

	res := make([]hnswCandidate, len(*best))
	for i := len(res) - 1; i >= 0; i-- {
		res[i] = heap.Pop(best).(hnswCandidate)
	}

	return res
}

// searchStartNode performs greedy descent from entry point to obtain a strong
// level-0 start node for the final best-first search.
func (h *hnswIndex) searchStartNode(query []float32) int {
	if len(h.nodes) == 0 {
		return -1
	}

	current := h.entryPoint
	if current < 0 || current >= len(h.nodes) {
		return h.firstLiveNode()
	}

	currentSim := hnswDot(query, h.nodeEmbedding(current))
	for level := h.maxLevel; level > 0; level-- {
		improved := true
		for improved {
			improved = false
			if level > h.nodes[current].level {
				break
			}
			for _, nid := range h.nodes[current].neighbors[level] {
				sim := hnswDot(query, h.nodeEmbedding(nid))
				if sim > currentSim {
					current = nid
					currentSim = sim
					improved = true
				}
			}
		}
	}

	if !h.isDeleted(current) {
		return current
	}

	bestID := -1
	bestSim := float32(-2)
	for _, nid := range h.nodes[current].neighbors[0] {
		if h.isDeleted(nid) {
			continue
		}
		sim := hnswDot(query, h.nodeEmbedding(nid))
		if sim > bestSim {
			bestSim = sim
			bestID = nid
		}
	}
	if bestID >= 0 {
		return bestID
	}

	return h.firstLiveNode()
}

// firstLiveNode returns any non-tombstoned node from latestNodeByDocID.
func (h *hnswIndex) firstLiveNode() int {
	for _, nodeID := range h.latestNodeByDocID {
		if nodeID >= 0 && nodeID < len(h.nodes) && !h.isDeleted(nodeID) {
			return nodeID
		}
	}
	return -1
}

// nodeEmbedding returns node embedding from arena-backed slice when available.
func (h *hnswIndex) nodeEmbedding(nodeID int) []float32 {
	if nodeID < 0 || nodeID >= len(h.nodes) {
		return nil
	}
	embedding := h.nodes[nodeID].embedding
	if len(embedding) > 0 {
		return embedding
	}
	if h.nodes[nodeID].doc == nil {
		return nil
	}
	return h.nodes[nodeID].doc.Embedding
}

// hnswWorkerCount selects worker count for current work size.
func hnswWorkerCount(workItems int) int {
	if workItems < hnswParallelMinWorkItems {
		return 1
	}

	workers := runtime.GOMAXPROCS(0)
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	if workers <= 1 {
		return 1
	}

	return min(workItems, workers)
}

// hnswWorkerRange returns [start,end) chunk for a given worker index.
func hnswWorkerRange(total, workers, workerIndex int) (int, int) {
	start := workerIndex * total / workers
	end := (workerIndex + 1) * total / workers
	return start, end
}

// scoreNodeIDs computes query-to-node similarities for ids.
func (h *hnswIndex) scoreNodeIDs(query []float32, ids []int) []hnswCandidate {
	return h.scoreNodeIDsWithBaseInto(query, ids, nil)
}

// scoreNodeIDsWithBase computes base-to-node similarities for ids.
func (h *hnswIndex) scoreNodeIDsWithBase(base []float32, ids []int) []hnswCandidate {
	return h.scoreNodeIDsWithBaseInto(base, ids, nil)
}

// scoreNodeIDsWithBaseInto writes scored candidates into dst (reused when possible).
func (h *hnswIndex) scoreNodeIDsWithBaseInto(base []float32, ids []int, dst []hnswCandidate) []hnswCandidate {
	if len(ids) == 0 {
		return dst[:0]
	}

	if len(ids) < hnswParallelScoreMinItems {
		if cap(dst) < len(ids) {
			dst = make([]hnswCandidate, 0, len(ids))
		} else {
			dst = dst[:0]
		}
		for _, id := range ids {
			if id < 0 || id >= len(h.nodes) {
				continue
			}
			sim := hnswDot(base, h.nodeEmbedding(id))
			dst = append(dst, hnswCandidate{id: id, similarity: sim})
		}
		return dst
	}

	workers := hnswWorkerCount(len(ids))
	if workers <= 1 {
		if cap(dst) < len(ids) {
			dst = make([]hnswCandidate, 0, len(ids))
		} else {
			dst = dst[:0]
		}
		for _, id := range ids {
			if id < 0 || id >= len(h.nodes) {
				continue
			}
			sim := hnswDot(base, h.nodeEmbedding(id))
			dst = append(dst, hnswCandidate{id: id, similarity: sim})
		}
		return dst
	}

	type scoredItem struct {
		ok   bool
		cand hnswCandidate
	}
	items := make([]scoredItem, len(ids))

	chunkSize := max(len(ids)/(workers*2), hnswParallelMinWorkItems)
	var wg sync.WaitGroup
	for worker := range workers {
		workerIndex := worker
		wg.Go(func() {
			start, end := hnswWorkerRange(len(ids), workers, workerIndex)
			for i := start; i < end; i += chunkSize {
				chunkEnd := min(i+chunkSize, end)
				for idx := i; idx < chunkEnd; idx++ {
					id := ids[idx]
					if id < 0 || id >= len(h.nodes) {
						continue
					}
					sim := hnswDot(base, h.nodeEmbedding(id))
					items[idx] = scoredItem{ok: true, cand: hnswCandidate{id: id, similarity: sim}}
				}
			}
		})
	}

	wg.Wait()

	if cap(dst) < len(ids) {
		dst = make([]hnswCandidate, 0, len(ids))
	} else {
		dst = dst[:0]
	}
	for _, item := range items {
		if !item.ok {
			continue
		}
		dst = append(dst, item.cand)
	}

	return dst
}

// hnswDot is the HNSW distance kernel: SIMD when available, scalar fallback.
func hnswDot(a, b []float32) float32 {
	if dotProductSIMDEnabled() {
		return dotProductSIMD(a, b)
	}
	return dotProductScalar(a, b)
}
