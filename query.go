package chromem

import (
	"cmp"
	"container/heap"
	"context"
	"fmt"
	"maps"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
)

var supportedFilters = []string{"$contains", "$not_contains"}

type docSim struct {
	doc        *Document
	similarity float32
}

// docMaxHeap is a max-heap of docSims, based on similarity.
// See https://pkg.go.dev/container/heap@go1.22#example-package-IntHeap
type docMaxHeap []docSim

func (h docMaxHeap) Len() int           { return len(h) }
func (h docMaxHeap) Less(i, j int) bool { return h[i].similarity < h[j].similarity }
func (h docMaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *docMaxHeap) Push(x any) {
	// Push and Pop use pointer receivers because they modify the slice's length,
	// not just its contents.
	*h = append(*h, x.(docSim))
}

func (h *docMaxHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// maxDocSims manages a max-heap of docSims with a fixed size, keeping the n highest
// similarities. It's not safe for concurrent use.
type maxDocSims struct {
	h    docMaxHeap
	size int
}

// newMaxDocSims creates a new nMaxDocs with a fixed size.
func newMaxDocSims(size int) *maxDocSims {
	return &maxDocSims{
		h:    make(docMaxHeap, 0, size),
		size: size,
	}
}

// add inserts a new docSim into the heap, keeping only the top n similarities.
func (d *maxDocSims) add(doc docSim) {
	if d.h.Len() < d.size {
		heap.Push(&d.h, doc)
	} else if d.h.Len() > 0 && d.h[0].similarity < doc.similarity {
		// Replace the smallest similarity if the new doc's similarity is higher
		heap.Pop(&d.h)
		heap.Push(&d.h, doc)
	}
}

// values returns the docSims in the heap, sorted by similarity (descending).
func (d *maxDocSims) values() []docSim {
	slices.SortFunc(d.h, func(i, j docSim) int {
		return cmp.Compare(j.similarity, i.similarity)
	})
	return d.h
}

var documentSlicePool = sync.Pool{
	New: func() any {
		return make([]*Document, 0, 256)
	},
}

func queryConcurrency(numDocs int, vectorDim int) int {
	if numDocs <= 0 {
		return 0
	}

	numCPUs := runtime.NumCPU()
	if numCPUs <= 1 {
		return 1
	}

	concurrency := max(numCPUs/2, 1)
	if numDocs < getQuerySmallDocsThreshold() {
		concurrency = numCPUs
	}

	if vectorDim >= getQueryHighDimThreshold() {
		concurrency = max(concurrency/getQueryHighDimConcurrencyDivisor(), 1)
	}

	return min(numDocs, concurrency)
}

func queryChunkSize(numDocs int) int {
	switch {
	case numDocs >= 100_000:
		return 1024
	case numDocs >= 25_000:
		return 512
	case numDocs >= 5_000:
		return 256
	default:
		return 128
	}
}

func docsFromMap(docs map[string]*Document) []*Document {
	return slices.Collect(maps.Values(docs))
}

// filterDocs filters a map of documents by metadata and content.
// It does this concurrently.
func filterDocs(docs map[string]*Document, where, whereDocument map[string]string) []*Document {
	numDocs := len(docs)
	concurrency := queryConcurrency(numDocs, 0)
	if concurrency == 0 {
		return nil
	}

	docsSlice := docsFromMap(docs)

	if len(where) == 0 && len(whereDocument) == 0 {
		return docsSlice
	}

	if concurrency == 1 || numDocs < getQuerySequentialDocsThreshold() {
		filteredDocs := documentSlicePool.Get().([]*Document)
		filteredDocs = filteredDocs[:0]
		for _, doc := range docsSlice {
			if documentMatchesFilters(doc, where, whereDocument) {
				filteredDocs = append(filteredDocs, doc)
			}
		}
		if len(filteredDocs) == 0 {
			documentSlicePool.Put(filteredDocs[:0])
			return nil
		}
		res := slices.Clone(filteredDocs)
		documentSlicePool.Put(filteredDocs[:0])
		return res
	}

	chunkSize := queryChunkSize(numDocs)
	var nextIndex atomic.Int64
	resultsChan := make(chan []*Document, concurrency)

	wg := sync.WaitGroup{}
	for range concurrency {
		wg.Go(func() {
			localMatches := documentSlicePool.Get().([]*Document)
			localMatches = localMatches[:0]
			for {
				start := int(nextIndex.Add(int64(chunkSize)) - int64(chunkSize))
				if start >= numDocs {
					break
				}

				end := min(start+chunkSize, numDocs)
				for _, doc := range docsSlice[start:end] {
					if documentMatchesFilters(doc, where, whereDocument) {
						localMatches = append(localMatches, doc)
					}
				}
			}
			resultsChan <- localMatches
		})
	}

	wg.Wait()
	close(resultsChan)

	filteredDocs := make([]*Document, 0, numDocs)
	for localMatches := range resultsChan {
		filteredDocs = append(filteredDocs, localMatches...)
		documentSlicePool.Put(localMatches[:0])
	}

	// With filteredDocs being initialized as potentially large slice, let's return
	// nil instead of the empty slice.
	if len(filteredDocs) == 0 {
		filteredDocs = nil
	}
	return filteredDocs
}

// documentMatchesFilters checks if a document matches the given filters.
// When calling this function, the whereDocument keys must already be validated!
func documentMatchesFilters(document *Document, where, whereDocument map[string]string) bool {
	// A document's metadata must have *all* the fields in the where clause.
	for k, v := range where {
		// TODO: Do we want to check for existence of the key? I.e. should
		// a where clause with empty string as value match a document's
		// metadata that doesn't have the key at all?
		if document.Metadata[k] != v {
			return false
		}
	}

	// A document must satisfy *all* filters, until we support the `$or` operator.
	for k, v := range whereDocument {
		switch k {
		case "$contains":
			if !strings.Contains(document.Content, v) {
				return false
			}
		case "$not_contains":
			if strings.Contains(document.Content, v) {
				return false
			}
		default:
			// No handling (error) required because we already validated the
			// operators. This simplifies the concurrency logic (no err var
			// and lock, no context to cancel).
		}
	}

	return true
}

func getMostSimilarDocs(ctx context.Context, queryVectors, negativeVector []float32, negativeFilterThreshold float32, docs []*Document, n int) ([]docSim, error) {
	numDocs := len(docs)
	concurrency := queryConcurrency(numDocs, len(queryVectors))
	if concurrency == 0 {
		return nil, nil
	}

	if concurrency == 1 || numDocs < getQuerySequentialDocsThreshold() {
		localMaxDocs := newMaxDocSims(n)
		for _, doc := range docs {
			sim, err := dotProduct(queryVectors, doc.Embedding)
			if err != nil {
				return nil, fmt.Errorf("couldn't calculate similarity for document '%s': %w", doc.ID, err)
			}

			if negativeFilterThreshold > 0 {
				nsim, err := dotProduct(negativeVector, doc.Embedding)
				if err != nil {
					return nil, fmt.Errorf("couldn't calculate negative similarity for document '%s': %w", doc.ID, err)
				}

				if nsim > negativeFilterThreshold {
					continue
				}
			}

			localMaxDocs.add(docSim{doc: doc, similarity: sim})
		}

		return localMaxDocs.values(), nil
	}

	var sharedErr error
	sharedErrLock := sync.Mutex{}
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	setSharedErr := func(err error) {
		sharedErrLock.Lock()
		defer sharedErrLock.Unlock()
		// Another goroutine might have already set the error.
		if sharedErr == nil {
			sharedErr = err
			// Cancel the operation for all other goroutines.
			cancel(sharedErr)
		}
	}

	wg := sync.WaitGroup{}
	resultsChan := make(chan []docSim, concurrency)
	chunkSize := queryChunkSize(numDocs)
	var nextIndex atomic.Int64
	for i := range concurrency {

		wg.Add(1)
		go func(workerIndex int) {
			defer wg.Done()
			localMaxDocs := newMaxDocSims(n)
			for {
				if ctx.Err() != nil {
					break
				}

				start := int(nextIndex.Add(int64(chunkSize)) - int64(chunkSize))
				if start >= numDocs {
					break
				}

				end := min(start+chunkSize, numDocs)
				for _, doc := range docs[start:end] {
					if ctx.Err() != nil {
						break
					}

					// As the vectors are normalized, the dot product is the cosine similarity.
					sim, err := dotProduct(queryVectors, doc.Embedding)
					if err != nil {
						setSharedErr(fmt.Errorf("couldn't calculate similarity for document '%s': %w", doc.ID, err))
						break
					}

					if negativeFilterThreshold > 0 {
						nsim, err := dotProduct(negativeVector, doc.Embedding)
						if err != nil {
							setSharedErr(fmt.Errorf("couldn't calculate negative similarity for document '%s': %w", doc.ID, err))
							break
						}

						if nsim > negativeFilterThreshold {
							continue
						}
					}

					localMaxDocs.add(docSim{doc: doc, similarity: sim})
				}
			}
			resultsChan <- localMaxDocs.values()
		}(i)
	}

	wg.Wait()
	close(resultsChan)

	if sharedErr != nil {
		return nil, sharedErr
	}

	nMaxDocs := newMaxDocSims(n)
	for workerTopK := range resultsChan {
		for _, doc := range workerTopK {
			nMaxDocs.add(doc)
		}
	}

	return nMaxDocs.values(), nil
}
