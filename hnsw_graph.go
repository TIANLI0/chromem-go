package chromem

import (
	"cmp"
	"fmt"
	"math/bits"
	"slices"
)

// insert adds one document to the graph.
//
// The algorithm first greedily descends from upper layers to find a good entry
// region, then performs a broader search/link procedure on each layer down to 0.
func (h *hnswIndex) insert(doc *Document) error {
	return h.insertWithEmbedding(doc, nil, -1)
}

// insertWithEmbedding inserts a node and links it from top layers to level 0.
// When embeddingOffset < 0, the embedding is copied into the arena first.
func (h *hnswIndex) insertWithEmbedding(doc *Document, embedding []float32, embeddingOffset int) error {
	if len(embedding) == 0 {
		embedding = doc.Embedding
	}
	if len(embedding) != h.dim {
		return fmt.Errorf("document '%s' embedding has dimension %d, expected %d", doc.ID, len(embedding), h.dim)
	}

	if embeddingOffset < 0 {
		embeddingOffset = len(h.embeddingArena)
		h.embeddingArena = append(h.embeddingArena, embedding...)
		embedding = h.embeddingArena[embeddingOffset : embeddingOffset+h.dim]
	}

	level := h.randomLevel()
	nodeID := len(h.nodes)
	h.nodes = append(h.nodes, hnswNode{
		doc:             doc,
		embedding:       embedding,
		embeddingOffset: embeddingOffset,
		level:           level,
		neighbors:       make([][]int, level+1),
	})
	h.ensureDeletedBitmapSize(len(h.nodes))
	if h.latestNodeByDocID == nil {
		h.latestNodeByDocID = make(map[string]int)
	}
	h.latestNodeByDocID[doc.ID] = nodeID
	if len(h.latestNodeByDocID) == 1 {
		h.entryPoint = nodeID
		h.maxLevel = level
		return nil
	}

	if h.entryPoint == -1 {
		h.entryPoint = nodeID
		h.maxLevel = level
		return nil
	}

	query := embedding
	ep := h.entryPoint

	for l := h.maxLevel; l > level; l-- {
		best := h.searchLayer(query, []int{ep}, 1, l)
		if len(best) > 0 {
			ep = best[0].id
		}
	}

	limitLevel := min(level, h.maxLevel)
	for l := limitLevel; l >= 0; l-- {
		efLevel := h.efConstruction
		if l > 0 {
			efLevel = max(h.m*2, h.efConstruction/2)
		}
		candidates := h.searchLayer(query, []int{ep}, efLevel, l)
		neighborLimit := h.m
		if l == 0 {
			neighborLimit = h.m * 2
		}
		selected := h.selectNeighborsPresorted(candidates, neighborLimit)
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

// upsert tombstones the previous node for doc ID (if any) and inserts a new one.
func (h *hnswIndex) upsert(doc *Document) error {
	if h.latestNodeByDocID != nil {
		if oldNodeID, ok := h.latestNodeByDocID[doc.ID]; ok {
			h.setDeleted(oldNodeID, true)
		}
	}
	return h.insert(doc)
}

// markDeleted tombstones the latest node of a document ID.
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

// deletedCount returns number of tombstoned nodes.
func (h *hnswIndex) deletedCount() int {
	count := 0
	for _, word := range h.deletedBitmap {
		count += bits.OnesCount64(word)
	}
	return count
}

// ensureDeletedBitmapSize grows the tombstone bitmap to cover nodeCount nodes.
func (h *hnswIndex) ensureDeletedBitmapSize(nodeCount int) {
	needed := (nodeCount + 63) / 64
	if len(h.deletedBitmap) >= needed {
		return
	}
	h.deletedBitmap = append(h.deletedBitmap, make([]uint64, needed-len(h.deletedBitmap))...)
}

// setDeleted marks or clears tombstone bit for nodeID.
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

// isDeleted reports whether nodeID is tombstoned.
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

	neighborLimit := h.m
	if level == 0 {
		neighborLimit = h.m * 2
	}

	if len(neighbors) > neighborLimit {
		baseVec := h.nodeEmbedding(fromID)
		candidates := h.scoreNodeIDsWithBase(baseVec, neighbors)
		neighbors = h.selectNeighbors(candidates, neighborLimit)
	}

	h.nodes[fromID].neighbors[level] = neighbors
}

// selectNeighbors picks up to limit candidates using HNSW-style diversification.
//
// We first keep high-similarity candidates that are not overly redundant with
// already selected neighbors (heuristic from the HNSW paper). If the heuristic
// under-fills the target size, we fill the remainder by pure similarity order.
func (h *hnswIndex) selectNeighbors(candidates []hnswCandidate, limit int) []int {
	return h.selectNeighborsInternal(candidates, limit, false)
}

// selectNeighborsPresorted is like selectNeighbors, but expects candidates
// already sorted by descending similarity.
func (h *hnswIndex) selectNeighborsPresorted(candidates []hnswCandidate, limit int) []int {
	return h.selectNeighborsInternal(candidates, limit, true)
}

// selectNeighborsInternal applies diversification then similarity fill.
func (h *hnswIndex) selectNeighborsInternal(candidates []hnswCandidate, limit int, presorted bool) []int {
	if len(candidates) == 0 || limit <= 0 {
		return nil
	}

	if !presorted {
		slices.SortFunc(candidates, func(a, b hnswCandidate) int {
			return cmp.Compare(b.similarity, a.similarity)
		})
	}

	selected := make([]int, 0, min(limit, len(candidates)))

	containsSelected := func(id int) bool {
		for _, existing := range selected {
			if existing == id {
				return true
			}
		}
		return false
	}

	for _, candidate := range candidates {
		if h.isDeleted(candidate.id) {
			continue
		}

		candidateVec := h.nodeEmbedding(candidate.id)
		redundant := false
		comparisons := 0
		for _, selectedID := range selected {
			if comparisons >= hnswDiversifyMaxComparisons {
				break
			}
			selectedVec := h.nodeEmbedding(selectedID)
			simToSelected := hnswDot(candidateVec, selectedVec)
			if simToSelected > candidate.similarity {
				redundant = true
				break
			}
			comparisons++
		}
		if redundant {
			continue
		}

		selected = append(selected, candidate.id)
		if len(selected) >= limit {
			break
		}
	}

	if len(selected) < limit {
		for _, candidate := range candidates {
			if h.isDeleted(candidate.id) {
				continue
			}
			if containsSelected(candidate.id) {
				continue
			}
			selected = append(selected, candidate.id)
			if len(selected) >= limit {
				break
			}
		}
	}

	return selected
}

// randomLevel samples a layer level for a new node.
//
// This implementation uses a geometric distribution with p=1/max(m,2) and a
// hard cap of 16 levels.
func (h *hnswIndex) randomLevel() int {
	probability := 1.0 / float64(max(h.m, 2))
	level := 0
	for level < 16 && h.rng.Float64() < probability {
		level++
	}
	return level
}
