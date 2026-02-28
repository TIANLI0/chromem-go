package chromem

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
)

// Build rebuilds the HNSW graph from the provided documents.
//
// Existing graph state is discarded. Every document must have a non-empty
// embedding with dimension equal to index dim.
func (h *hnswIndex) Build(docs []*Document) error {
	shards := getHNSWBuildParallelShards()
	if shards > 1 && len(docs) >= shards*hnswParallelBuildMinDocsPerShard {
		return h.buildParallelShards(docs, shards)
	}

	return h.buildSequential(docs)
}

// buildSequential constructs the full graph in one shard.
func (h *hnswIndex) buildSequential(docs []*Document) error {
	if cap(h.nodes) < len(docs) {
		h.nodes = make([]hnswNode, 0, len(docs))
	} else {
		h.nodes = h.nodes[:0]
	}
	h.embeddingArena = h.embeddingArena[:0]
	h.deletedBitmap = h.deletedBitmap[:0]
	h.latestNodeByDocID = make(map[string]int, len(docs))
	h.entryPoint = -1
	h.maxLevel = -1
	h.shardEntryPoints = nil

	if err := validateHNSWDocs(docs, h.dim); err != nil {
		return err
	}

	if len(docs) > 0 {
		h.embeddingArena = make([]float32, len(docs)*h.dim)
		workers := hnswWorkerCount(len(docs))
		if workers <= 1 {
			for i, doc := range docs {
				offset := i * h.dim
				copy(h.embeddingArena[offset:offset+h.dim], doc.Embedding)
			}
		} else {
			var wg sync.WaitGroup
			for worker := range workers {
				workerIndex := worker
				wg.Go(func() {
					start, end := hnswWorkerRange(len(docs), workers, workerIndex)
					for i := start; i < end; i++ {
						offset := i * h.dim
						copy(h.embeddingArena[offset:offset+h.dim], docs[i].Embedding)
					}
				})
			}
			wg.Wait()
		}
	}

	for i, doc := range docs {
		embeddingOffset := i * h.dim
		embedding := h.embeddingArena[embeddingOffset : embeddingOffset+h.dim]
		if err := h.insertWithEmbedding(doc, embedding, embeddingOffset); err != nil {
			return err
		}
	}

	return nil
}

// getHNSWBuildParallelShards reads build shard count from env.
func getHNSWBuildParallelShards() int {
	value, ok := os.LookupEnv("CHROMEM_HNSW_BUILD_PARALLEL_SHARDS")
	if !ok {
		return 1
	}
	parsed, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || parsed <= 1 {
		return 1
	}
	return parsed
}

// getHNSWBuildParallelRefineEnabled toggles post-merge graph refinement.
func getHNSWBuildParallelRefineEnabled() bool {
	value, ok := os.LookupEnv("CHROMEM_HNSW_BUILD_PARALLEL_REFINE")
	if !ok {
		return true
	}
	normalized := strings.ToLower(strings.TrimSpace(value))
	switch normalized {
	case "0", "false", "off", "no":
		return false
	default:
		return true
	}
}

// buildParallelShards builds multiple subgraphs in parallel and merges them.
func (h *hnswIndex) buildParallelShards(docs []*Document, requestedShards int) error {
	if err := validateHNSWDocs(docs, h.dim); err != nil {
		return err
	}

	shards := min(requestedShards, len(docs))
	if shards <= 1 {
		return h.buildSequential(docs)
	}

	type shardResult struct {
		index *hnswIndex
		err   error
	}
	results := make([]shardResult, shards)

	workers := runtime.GOMAXPROCS(0)
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	workers = min(shards, max(workers, 1))

	var wg sync.WaitGroup
	jobs := make(chan int, shards)
	for shardID := range shards {
		jobs <- shardID
	}
	close(jobs)

	for worker := 0; worker < workers; worker++ {
		wg.Go(func() {
			for shardID := range jobs {
				start, end := hnswWorkerRange(len(docs), shards, shardID)
				if start >= end {
					continue
				}
				sub := newHNSWIndex(h.dim, h.m, h.efConstruction, h.efSearch)
				err := sub.buildSequential(docs[start:end])
				results[shardID] = shardResult{index: sub, err: err}
			}
		})
	}
	wg.Wait()

	for _, item := range results {
		if item.err != nil {
			return item.err
		}
	}

	totalNodes := 0
	totalArena := 0
	for _, item := range results {
		if item.index == nil {
			continue
		}
		totalNodes += len(item.index.nodes)
		totalArena += len(item.index.embeddingArena)
	}

	h.nodes = make([]hnswNode, 0, totalNodes)
	h.embeddingArena = make([]float32, 0, totalArena)
	h.deletedBitmap = h.deletedBitmap[:0]
	h.latestNodeByDocID = make(map[string]int, len(docs))
	h.entryPoint = -1
	h.maxLevel = -1

	entryPoints := make([]int, 0, shards)
	shardRanges := make([]hnswShardRange, 0, shards)
	for _, item := range results {
		sub := item.index
		if sub == nil || len(sub.nodes) == 0 {
			continue
		}

		nodeOffset := len(h.nodes)
		shardStart := nodeOffset
		embOffset := len(h.embeddingArena)
		h.embeddingArena = append(h.embeddingArena, sub.embeddingArena...)

		for i := range sub.nodes {
			node := sub.nodes[i]

			if node.embeddingOffset >= 0 {
				newEmbeddingOffset := embOffset + node.embeddingOffset
				node.embeddingOffset = newEmbeddingOffset
				node.embedding = h.embeddingArena[newEmbeddingOffset : newEmbeddingOffset+h.dim]
			}

			for level := range node.neighbors {
				for j := range node.neighbors[level] {
					node.neighbors[level][j] += nodeOffset
				}
			}

			h.nodes = append(h.nodes, node)
		}

		for docID, localNodeID := range sub.latestNodeByDocID {
			h.latestNodeByDocID[docID] = nodeOffset + localNodeID
		}

		globalEntry := nodeOffset + sub.entryPoint
		entryPoints = append(entryPoints, globalEntry)
		shardRanges = append(shardRanges, hnswShardRange{start: shardStart, end: len(h.nodes)})
		if sub.maxLevel > h.maxLevel {
			h.maxLevel = sub.maxLevel
			h.entryPoint = globalEntry
		}
	}

	h.ensureDeletedBitmapSize(len(h.nodes))
	h.connectShardAnchors(entryPoints, shardRanges)
	h.shardEntryPoints = slices.Clone(entryPoints)

	if getHNSWBuildParallelRefineEnabled() && len(h.nodes) >= hnswParallelRefineMinNodes {
		h.refineMergedShardGraph()
	}

	if h.rng == nil {
		h.rng = rand.New(rand.NewSource(42))
	}

	return nil
}

// refineMergedShardGraph improves merged level-0 neighborhood quality.
func (h *hnswIndex) refineMergedShardGraph() {
	if len(h.nodes) == 0 {
		return
	}

	entryPoints := h.shardEntryPoints
	if len(entryPoints) == 0 && h.entryPoint >= 0 {
		entryPoints = []int{h.entryPoint}
	}
	if len(entryPoints) == 0 {
		return
	}

	neighborLimit := h.m * 2
	refineEF := max(h.efConstruction, max(h.efSearch*4, neighborLimit*8))
	refineEF = min(refineEF, len(h.nodes))

	for nodeID := range h.nodes {
		if h.isDeleted(nodeID) {
			continue
		}

		query := h.nodeEmbedding(nodeID)
		candidates := h.searchLayer(query, entryPoints, refineEF, 0)
		if len(candidates) == 0 {
			continue
		}

		existing := h.nodes[nodeID].neighbors[0]
		mergedIDs := make([]int, 0, len(existing)+len(candidates))
		seen := make(map[int]struct{}, len(existing)+len(candidates))
		for _, id := range existing {
			if id == nodeID || id < 0 || id >= len(h.nodes) {
				continue
			}
			if _, ok := seen[id]; ok {
				continue
			}
			seen[id] = struct{}{}
			mergedIDs = append(mergedIDs, id)
		}
		for _, candidate := range candidates {
			id := candidate.id
			if id == nodeID || id < 0 || id >= len(h.nodes) {
				continue
			}
			if _, ok := seen[id]; ok {
				continue
			}
			seen[id] = struct{}{}
			mergedIDs = append(mergedIDs, id)
		}
		if len(mergedIDs) == 0 {
			continue
		}

		scored := h.scoreNodeIDsWithBase(query, mergedIDs)
		newNeighbors := h.selectNeighbors(scored, neighborLimit)
		h.nodes[nodeID].neighbors[0] = newNeighbors
		for _, nid := range newNeighbors {
			h.link(nid, nodeID, 0)
		}
	}
}

// connectShardAnchors adds cross-shard bridges to improve navigability.
func (h *hnswIndex) connectShardAnchors(entryPoints []int, ranges []hnswShardRange) {
	if len(ranges) <= 1 {
		return
	}

	const anchorsPerShard = 32
	const bridgesPerAnchor = 2

	type anchor struct {
		id    int
		shard int
	}

	shardAnchors := make([][]int, len(ranges))
	for shardID, r := range ranges {
		if r.end <= r.start {
			continue
		}
		count := r.end - r.start
		step := max(1, count/anchorsPerShard)
		anchors := make([]int, 0, anchorsPerShard+1)
		for id := r.start; id < r.end && len(anchors) < anchorsPerShard; id += step {
			anchors = append(anchors, id)
		}
		if shardID < len(entryPoints) {
			entryID := entryPoints[shardID]
			exists := slices.Contains(anchors, entryID)
			if !exists && entryID >= r.start && entryID < r.end {
				anchors = append(anchors, entryID)
			}
		}
		shardAnchors[shardID] = anchors
	}

	allAnchors := make([]anchor, 0, len(ranges)*anchorsPerShard)
	for shardID, anchors := range shardAnchors {
		for _, id := range anchors {
			allAnchors = append(allAnchors, anchor{id: id, shard: shardID})
		}
	}

	for _, from := range allAnchors {
		fromVec := h.nodeEmbedding(from.id)
		candidates := make([]hnswCandidate, 0, len(allAnchors))
		for _, to := range allAnchors {
			if to.shard == from.shard || to.id == from.id {
				continue
			}
			sim := hnswDot(fromVec, h.nodeEmbedding(to.id))
			candidates = append(candidates, hnswCandidate{id: to.id, similarity: sim})
		}

		for _, toID := range h.selectNeighbors(candidates, min(bridgesPerAnchor, len(candidates))) {
			h.link(from.id, toID, 0)
			h.link(toID, from.id, 0)
		}
	}
}

// validateHNSWDocs validates embedding presence and dimensionality.
func validateHNSWDocs(docs []*Document, dim int) error {
	if len(docs) == 0 {
		return nil
	}

	workers := hnswWorkerCount(len(docs))
	if workers <= 1 {
		for _, doc := range docs {
			if len(doc.Embedding) == 0 {
				return fmt.Errorf("document '%s' embedding is empty", doc.ID)
			}
			if len(doc.Embedding) != dim {
				return fmt.Errorf("document '%s' embedding has dimension %d, expected %d", doc.ID, len(doc.Embedding), dim)
			}
		}
		return nil
	}

	errCh := make(chan error, 1)
	var once sync.Once
	setErr := func(err error) {
		once.Do(func() {
			errCh <- err
		})
	}

	chunkSize := max(len(docs)/(workers*2), 64)
	var wg sync.WaitGroup
	for worker := range workers {
		workerIndex := worker
		wg.Go(func() {
			start, end := hnswWorkerRange(len(docs), workers, workerIndex)
			for i := start; i < end; i += chunkSize {
				if len(errCh) > 0 {
					return
				}
				chunkEnd := min(i+chunkSize, end)
				for _, doc := range docs[i:chunkEnd] {
					if len(doc.Embedding) == 0 {
						setErr(fmt.Errorf("document '%s' embedding is empty", doc.ID))
						return
					}
					if len(doc.Embedding) != dim {
						setErr(fmt.Errorf("document '%s' embedding has dimension %d, expected %d", doc.ID, len(doc.Embedding), dim))
						return
					}
				}
			}
		})
	}

	wg.Wait()
	if len(errCh) > 0 {
		return <-errCh
	}

	return nil
}
