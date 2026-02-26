package chromem

import "runtime"

// CollectionMemoryStats provides memory observations for a collection and the current process.
type CollectionMemoryStats struct {
	DocumentCount int

	ApproxEmbeddingBytes uint64
	ApproxMetadataBytes  uint64
	ApproxContentBytes   uint64
	EstimatedTotalBytes  uint64

	RuntimeAllocBytes  uint64
	RuntimeHeapInuse   uint64
	RuntimeHeapObjects uint64
	RuntimeNumGC       uint32
}

// MemoryStats returns an approximate memory footprint for collection data
// and selected runtime memory counters.
func (c *Collection) MemoryStats() CollectionMemoryStats {
	c.documentsLock.RLock()
	docCount := len(c.documents)

	var embeddingBytes uint64
	var metadataBytes uint64
	var contentBytes uint64
	for _, doc := range c.documents {
		embeddingBytes += uint64(len(doc.Embedding)) * 4
		contentBytes += uint64(len(doc.Content))
		for k, v := range doc.Metadata {
			metadataBytes += uint64(len(k) + len(v))
		}
	}
	c.documentsLock.RUnlock()

	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)

	estimatedTotal := embeddingBytes + metadataBytes + contentBytes

	return CollectionMemoryStats{
		DocumentCount: docCount,

		ApproxEmbeddingBytes: embeddingBytes,
		ApproxMetadataBytes:  metadataBytes,
		ApproxContentBytes:   contentBytes,
		EstimatedTotalBytes:  estimatedTotal,

		RuntimeAllocBytes:  mem.Alloc,
		RuntimeHeapInuse:   mem.HeapInuse,
		RuntimeHeapObjects: mem.HeapObjects,
		RuntimeNumGC:       mem.NumGC,
	}
}
