package chromem

import (
	"maps"
	"math/rand"
	"slices"
)

// Internal thresholds used by build/search parallel paths and neighbor diversification.
const hnswParallelMinWorkItems = 128
const hnswParallelScoreMinItems = 512
const hnswDiversifyMaxComparisons = 8
const hnswParallelBuildMinDocsPerShard = 1024
const hnswParallelRefineMinNodes = 2048

// hnswVisitedState provides epoch-based visited marking for graph traversal,
// avoiding per-search map allocations.
type hnswVisitedState struct {
	marks []uint32
	epoch uint32
}

type hnswShardRange struct {
	start int
	end   int
}

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
	doc             *Document
	embedding       []float32
	embeddingOffset int
	level           int
	neighbors       [][]int
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
	// embeddingArena stores node embeddings densely for better cache locality.
	embeddingArena []float32
	// deletedBitmap marks tombstoned nodes in a compact bitset form.
	// Bit i corresponds to node i.
	deletedBitmap []uint64
	// latestNodeByDocID points each live document ID to its most recent node.
	latestNodeByDocID map[string]int
	// entryPoint is the current top-layer entry node ID, or -1 when empty.
	entryPoint int
	// maxLevel is the highest level currently present in the graph.
	maxLevel int
	// shardEntryPoints stores per-shard start nodes for optional multi-start search.
	shardEntryPoints []int
	// rng is used to sample levels for newly inserted nodes.
	rng *rand.Rand
}

func (h *hnswIndex) clone() *hnswIndex {
	if h == nil {
		return nil
	}

	embeddingArena := slices.Clone(h.embeddingArena)
	nodes := make([]hnswNode, len(h.nodes))
	for i, node := range h.nodes {
		neighbors := make([][]int, len(node.neighbors))
		for level := range node.neighbors {
			neighbors[level] = slices.Clone(node.neighbors[level])
		}

		var embedding []float32
		if node.embeddingOffset >= 0 && node.embeddingOffset+h.dim <= len(embeddingArena) {
			embedding = embeddingArena[node.embeddingOffset : node.embeddingOffset+h.dim]
		} else {
			embedding = slices.Clone(node.embedding)
		}

		nodes[i] = hnswNode{
			doc:             node.doc,
			embedding:       embedding,
			embeddingOffset: node.embeddingOffset,
			level:           node.level,
			neighbors:       neighbors,
		}
	}

	return &hnswIndex{
		dim:               h.dim,
		m:                 h.m,
		efConstruction:    h.efConstruction,
		efSearch:          h.efSearch,
		nodes:             nodes,
		embeddingArena:    embeddingArena,
		deletedBitmap:     slices.Clone(h.deletedBitmap),
		latestNodeByDocID: mapsClone(h.latestNodeByDocID),
		entryPoint:        h.entryPoint,
		maxLevel:          h.maxLevel,
		shardEntryPoints:  slices.Clone(h.shardEntryPoints),
		rng:               rand.New(rand.NewSource(h.rng.Int63())),
	}
}

func mapsClone[K comparable, V any](m map[K]V) map[K]V {
	if m == nil {
		return nil
	}
	out := make(map[K]V, len(m))
	maps.Copy(out, m)
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
