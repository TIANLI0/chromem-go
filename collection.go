package chromem

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"path/filepath"
	"slices"
	"sync"
	"sync/atomic"
)

// Collection represents a collection of documents.
// It also has a configured embedding function, which is used when adding documents
// that don't have embeddings yet.
type Collection struct {
	Name string

	metadata      map[string]string
	documents     map[string]*Document
	documentsList []*Document
	docsListValid bool
	documentsLock sync.RWMutex
	embed         EmbeddingFunc

	persistDirectory        string
	compress                bool
	streamEmbeddingsOnQuery bool
	hnsw                    *hnswIndex
	hnswBuildLock           sync.Mutex
	hnswVersion             atomic.Uint64
	hnswIndexedVersion      atomic.Uint64

	// ⚠️ When adding fields here, consider adding them to the persistence struct
	// versions in [DB.Export] and [DB.Import] as well!
}

const hnswIndexFileName = "00000001.hidx"

type persistedHNSWNode struct {
	DocID     string
	Level     int
	Neighbors [][]int
}

type persistedHNSWIndex struct {
	Dim            int
	M              int
	EFConstruction int
	EFSearch       int
	EntryPoint     int
	MaxLevel       int
	Deleted        []bool
	DeletedBitmap  []uint64
	Nodes          []persistedHNSWNode
}

// NegativeMode represents the mode to use for the negative text.
// See QueryOptions for more information.
type NegativeMode string

const (
	// NEGATIVE_MODE_FILTER filters out results based on the similarity between the
	// negative embedding and the document embeddings.
	// NegativeFilterThreshold controls the threshold for filtering. Documents with
	// similarity above the threshold will be removed from the results.
	NEGATIVE_MODE_FILTER NegativeMode = "filter"

	// NEGATIVE_MODE_SUBTRACT subtracts the negative embedding from the query embedding.
	// This is the default behavior.
	NEGATIVE_MODE_SUBTRACT NegativeMode = "subtract"

	// The default threshold for the negative filter.
	DEFAULT_NEGATIVE_FILTER_THRESHOLD = 0.5
)

// QueryOptions represents the options for a query.
type QueryOptions struct {
	// The text to search for.
	QueryText string

	// The embedding of the query to search for. It must be created
	// with the same embedding model as the document embeddings in the collection.
	// The embedding will be normalized if it's not the case yet.
	// If both QueryText and QueryEmbedding are set, QueryEmbedding will be used.
	QueryEmbedding []float32

	// The number of results to return.
	NResults int

	// Conditional filtering on metadata.
	Where map[string]string

	// Conditional filtering on documents.
	WhereDocument map[string]string

	// Negative is the negative query options.
	// They can be used to exclude certain results from the query.
	Negative NegativeQueryOptions
}

type NegativeQueryOptions struct {
	// Mode is the mode to use for the negative text.
	Mode NegativeMode

	// Text is the text to exclude from the results.
	Text string

	// Embedding is the embedding of the negative text. It must be created
	// with the same embedding model as the document embeddings in the collection.
	// The embedding will be normalized if it's not the case yet.
	// If both Text and Embedding are set, Embedding will be used.
	Embedding []float32

	// FilterThreshold is the threshold for the negative filter. Used when Mode is NEGATIVE_MODE_FILTER.
	FilterThreshold float32
}

// We don't export this yet to keep the API surface to the bare minimum.
// Users create collections via [Client.CreateCollection].
func newCollection(name string, metadata map[string]string, embed EmbeddingFunc, dbDir string, compress bool, streamEmbeddingsOnQuery bool) (*Collection, error) {
	// We copy the metadata to avoid data races in case the caller modifies the
	// map after creating the collection while we range over it.
	m := make(map[string]string, len(metadata))
	maps.Copy(m, metadata)

	c := &Collection{
		Name: name,

		metadata:                m,
		documents:               make(map[string]*Document),
		embed:                   embed,
		streamEmbeddingsOnQuery: streamEmbeddingsOnQuery,
	}
	c.hnswVersion.Store(1)

	// Persistence
	if dbDir != "" {
		safeName := hash2hex(name)
		c.persistDirectory = filepath.Join(dbDir, safeName)
		c.compress = compress
		return c, c.persistMetadata()
	}

	return c, nil
}

// Add embeddings to the datastore.
//
//   - ids: The ids of the embeddings you wish to add
//   - embeddings: The embeddings to add. If nil, embeddings will be computed based
//     on the contents using the embeddingFunc set for the Collection. Optional.
//   - metadatas: The metadata to associate with the embeddings. When querying,
//     you can filter on this metadata. Optional.
//   - contents: The contents to associate with the embeddings.
//
// This is a Chroma-like method. For a more Go-idiomatic one, see [Collection.AddDocuments].
func (c *Collection) Add(ctx context.Context, ids []string, embeddings [][]float32, metadatas []map[string]string, contents []string) error {
	return c.AddConcurrently(ctx, ids, embeddings, metadatas, contents, 1)
}

// AddConcurrently is like Add, but adds embeddings concurrently.
// This is mostly useful when you don't pass any embeddings, so they have to be created.
// Upon error, concurrently running operations are canceled and the error is returned.
//
// This is a Chroma-like method. For a more Go-idiomatic one, see [Collection.AddDocuments].
func (c *Collection) AddConcurrently(ctx context.Context, ids []string, embeddings [][]float32, metadatas []map[string]string, contents []string, concurrency int) error {
	if len(ids) == 0 {
		return errors.New("ids are empty")
	}
	if len(embeddings) == 0 && len(contents) == 0 {
		return errors.New("either embeddings or contents must be filled")
	}
	if len(embeddings) != 0 {
		if len(embeddings) != len(ids) {
			return errors.New("ids and embeddings must have the same length")
		}
	} else {
		// Assign empty slice, so we can simply access via index later
		embeddings = make([][]float32, len(ids))
	}
	if len(metadatas) != 0 {
		if len(ids) != len(metadatas) {
			return errors.New("when metadatas is not empty it must have the same length as ids")
		}
	} else {
		// Assign empty slice, so we can simply access via index later
		metadatas = make([]map[string]string, len(ids))
	}
	if len(contents) != 0 {
		if len(contents) != len(ids) {
			return errors.New("ids and contents must have the same length")
		}
	} else {
		// Assign empty slice, so we can simply access via index later
		contents = make([]string, len(ids))
	}
	if concurrency < 1 {
		return errors.New("concurrency must be at least 1")
	}

	// Convert Chroma-style parameters into a slice of documents.
	docs := make([]Document, 0, len(ids))
	for i, id := range ids {
		docs = append(docs, Document{
			ID:        id,
			Metadata:  metadatas[i],
			Embedding: embeddings[i],
			Content:   contents[i],
		})
	}

	return c.AddDocuments(ctx, docs, concurrency)
}

// AddDocuments adds documents to the collection with the specified concurrency.
// If the documents don't have embeddings, they will be created using the collection's
// embedding function.
// Upon error, concurrently running operations are canceled and the error is returned.
func (c *Collection) AddDocuments(ctx context.Context, documents []Document, concurrency int) error {
	if len(documents) == 0 {
		// TODO: Should this be a no-op instead?
		return errors.New("documents slice is nil or empty")
	}
	if concurrency < 1 {
		return errors.New("concurrency must be at least 1")
	}
	// For other validations we rely on AddDocument.

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

	var wg sync.WaitGroup
	semaphore := make(chan struct{}, concurrency)
	for _, doc := range documents {
		wg.Add(1)
		go func(doc Document) {
			defer wg.Done()

			// Don't even start if another goroutine already failed.
			if ctx.Err() != nil {
				return
			}

			// Wait here while $concurrency other goroutines are creating documents.
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			err := c.AddDocument(ctx, doc)
			if err != nil {
				setSharedErr(fmt.Errorf("couldn't add document '%s': %w", doc.ID, err))
				return
			}
		}(doc)
	}

	wg.Wait()

	return sharedErr
}

// AddDocument adds a document to the collection.
// If the document doesn't have an embedding, it will be created using the collection's
// embedding function.
func (c *Collection) AddDocument(ctx context.Context, doc Document) error {
	if doc.ID == "" {
		return errors.New("document ID is empty")
	}
	if len(doc.Embedding) == 0 && doc.Content == "" {
		return errors.New("either document embedding or content must be filled")
	}

	// We copy the metadata to avoid data races in case the caller modifies the
	// map after creating the document while we range over it.
	m := make(map[string]string, len(doc.Metadata))
	maps.Copy(m, doc.Metadata)

	// Create embedding if they don't exist, otherwise normalize if necessary
	if len(doc.Embedding) == 0 {
		embedding, err := c.embed(ctx, doc.Content)
		if err != nil {
			return fmt.Errorf("couldn't create embedding of document: %w", err)
		}
		doc.Embedding = embedding
	} else {
		if !isNormalized(doc.Embedding) {
			doc.Embedding = normalizeVector(doc.Embedding)
		}
	}

	c.documentsLock.Lock()
	doc.payloadLoaded = true
	if c.persistDirectory != "" {
		doc.persistPath = c.getDocPath(doc.ID)
	}
	_, existed := c.documents[doc.ID]
	// We don't defer the unlock because we want to do it earlier.
	c.documents[doc.ID] = &doc
	c.docsListValid = false
	c.documentsLock.Unlock()

	// Persist the document
	if c.persistDirectory != "" {
		docPath := doc.persistPath
		err := persistToFile(docPath, doc, c.compress, "")
		if err != nil {
			return fmt.Errorf("couldn't persist document to %q: %w", docPath, err)
		}
		if c.streamEmbeddingsOnQuery {
			c.documentsLock.Lock()
			if storedDoc, ok := c.documents[doc.ID]; ok {
				storedDoc.Embedding = nil
			}
			c.documentsLock.Unlock()
		}
	}

	upserted, err := c.tryIncrementalHNSWUpsert(&doc)
	if err != nil {
		return err
	}
	if !upserted {
		c.markHNSWDirty()
		c.removePersistedHNSWIndexBestEffort()
	} else if existed {
		// Keep docsList snapshot and index versions coherent after overwrite.
		// Incremental upsert already advances index versions.
	} else {
		// Nothing else to do.
	}

	return nil
}

// ListIDs returns the IDs of all documents in the collection.
func (c *Collection) ListIDs(_ context.Context) []string {
	c.documentsLock.RLock()
	defer c.documentsLock.RUnlock()

	ids := make([]string, 0, len(c.documents))
	for id := range c.documents {
		ids = append(ids, id)
	}

	return ids
}

// ListDocuments returns all documents in the collection. The returned documents
// are a deep copy of the original ones, so you can modify them without affecting
// the collection.
func (c *Collection) ListDocuments(_ context.Context) ([]*Document, error) {
	if err := c.ensureAllDocumentEmbeddingsLoaded(); err != nil {
		return nil, err
	}
	if err := c.ensureAllDocumentPayloadLoaded(); err != nil {
		return nil, err
	}

	c.documentsLock.RLock()
	defer c.documentsLock.RUnlock()

	results := make([]*Document, 0, len(c.documents))
	for _, doc := range c.documents {
		docCopy := cloneDocument(doc) // Deep copy
		results = append(results, docCopy)
	}
	return results, nil
}

// ListDocumentsShallow returns all documents in the collection. The returned documents'
// metadata and embeddings point to the original data, so modifying them will be
// reflected in the collection.
func (c *Collection) ListDocumentsShallow(_ context.Context) ([]*Document, error) {
	if err := c.ensureAllDocumentEmbeddingsLoaded(); err != nil {
		return nil, err
	}
	if err := c.ensureAllDocumentPayloadLoaded(); err != nil {
		return nil, err
	}

	c.documentsLock.RLock()
	defer c.documentsLock.RUnlock()

	results := make([]*Document, 0, len(c.documents))
	for _, doc := range c.documents {
		docCopy := *doc // Shallow copy
		results = append(results, &docCopy)
	}
	return results, nil
}

// ListDocumentsPartial returns a partial version of all documents in the collection,
// containing only the ID and content, but not the embedding or metadata values.
func (c *Collection) ListDocumentsPartial(_ context.Context) ([]*Document, error) {
	if err := c.ensureAllDocumentEmbeddingsLoaded(); err != nil {
		return nil, err
	}
	if err := c.ensureAllDocumentPayloadLoaded(); err != nil {
		return nil, err
	}

	c.documentsLock.RLock()
	defer c.documentsLock.RUnlock()

	results := make([]*Document, 0, len(c.documents))
	for _, doc := range c.documents {
		partialDoc := makePartialDocument(doc) // Shallow copy
		results = append(results, partialDoc)
	}
	return results, nil
}

// GetByID returns a document by its ID.
// The returned document is a copy of the original document, so it can be safely
// modified without affecting the collection.
func (c *Collection) GetByID(_ context.Context, id string) (Document, error) {
	if id == "" {
		return Document{}, errors.New("document ID is empty")
	}

	c.documentsLock.RLock()
	doc, ok := c.documents[id]
	c.documentsLock.RUnlock()

	if !ok {
		return Document{}, fmt.Errorf("document with ID '%v' not found", id)
	}

	if _, err := c.ensureDocumentEmbeddingLoaded(doc, true); err != nil {
		return Document{}, err
	}

	if err := c.ensureDocumentPayloadLoaded(doc); err != nil {
		return Document{}, err
	}
	res := cloneDocument(doc)
	return *res, nil
}

// GetByMetadata returns a set of documents, filtered by their metadata.
// The metadata tags must match the params specified in the where argument in both
// key and value.
// The returned documents are a deep copy of the original document, so they can
// be safely modified without affecting the collection.
func (c *Collection) GetByMetadata(_ context.Context, where map[string]string) ([]*Document, error) {
	if err := c.ensureAllDocumentEmbeddingsLoaded(); err != nil {
		return nil, err
	}
	if err := c.ensureAllDocumentPayloadLoaded(); err != nil {
		return nil, err
	}

	c.documentsLock.RLock()
	defer c.documentsLock.RUnlock()

	var results []*Document
	for _, doc := range c.documents {
		match := true
		for key, value := range where {
			if docVal, ok := doc.Metadata[key]; !ok || docVal != value {
				match = false
				break
			}
		}
		if match {
			docCopy := cloneDocument(doc) // Deep copy
			results = append(results, docCopy)
		}
	}
	return results, nil
}

// Delete removes document(s) from the collection.
//
//   - where: Conditional filtering on metadata. Optional.
//   - whereDocument: Conditional filtering on documents. Optional.
//   - ids: The ids of the documents to delete. If empty, all documents are deleted.
func (c *Collection) Delete(_ context.Context, where, whereDocument map[string]string, ids ...string) error {
	// must have at least one of where, whereDocument or ids
	if len(where) == 0 && len(whereDocument) == 0 && len(ids) == 0 {
		return fmt.Errorf("must have at least one of where, whereDocument or ids")
	}

	if len(c.documents) == 0 {
		return nil
	}

	for k := range whereDocument {
		if !slices.Contains(supportedFilters, k) {
			return errors.New("unsupported whereDocument operator")
		}
	}

	if where != nil || whereDocument != nil {
		if err := c.ensureAllDocumentPayloadLoaded(); err != nil {
			return err
		}
	}

	var docIDs []string

	c.documentsLock.Lock()

	if where != nil || whereDocument != nil {
		// metadata + content filters
		docs := c.getDocumentsListLocked()
		for _, doc := range docs {
			if !documentMatchesFilters(doc, where, whereDocument) {
				continue
			}
			docIDs = append(docIDs, doc.ID)
		}
	} else {
		docIDs = ids
	}

	// No-op if no docs are left
	if len(docIDs) == 0 {
		c.documentsLock.Unlock()
		return nil
	}

	for _, docID := range docIDs {
		delete(c.documents, docID)

		// Remove the document from disk
		if c.persistDirectory != "" {
			docPath := c.getDocPath(docID)
			err := removeFile(docPath)
			if err != nil {
				c.documentsLock.Unlock()
				return fmt.Errorf("couldn't remove document at %q: %w", docPath, err)
			}
		}
	}
	c.docsListValid = false
	c.documentsLock.Unlock()

	deleted, err := c.tryIncrementalHNSWDelete(docIDs)
	if err != nil {
		return err
	}
	if !deleted {
		c.markHNSWDirty()
		c.removePersistedHNSWIndexBestEffort()
	}

	return nil
}

func (c *Collection) markHNSWDirtyLocked() {
	c.hnsw = nil
	c.hnswVersion.Add(1)
}

func (c *Collection) markHNSWDirty() {
	c.documentsLock.Lock()
	c.markHNSWDirtyLocked()
	c.documentsLock.Unlock()
}

func (c *Collection) tryIncrementalHNSWUpsert(doc *Document) (bool, error) {
	if doc == nil || len(doc.Embedding) == 0 {
		return false, nil
	}
	if !getHNSWEnabled() || c.streamEmbeddingsOnQuery {
		return false, nil
	}

	c.hnswBuildLock.Lock()
	defer c.hnswBuildLock.Unlock()

	if c.hnsw == nil {
		return false, nil
	}
	if c.hnswIndexedVersion.Load() != c.hnswVersion.Load() {
		return false, nil
	}
	if len(doc.Embedding) != c.hnsw.dim {
		return false, nil
	}

	nextIndex := c.hnsw.clone()
	if err := nextIndex.upsert(doc); err != nil {
		return false, fmt.Errorf("couldn't upsert document '%s' into hnsw index: %w", doc.ID, err)
	}

	newVersion := c.hnswVersion.Add(1)
	c.hnsw = nextIndex
	c.hnswIndexedVersion.Store(newVersion)
	if err := c.persistHNSWIndex(nextIndex); err != nil {
		return false, fmt.Errorf("couldn't persist hnsw index: %w", err)
	}

	return true, nil
}

func (c *Collection) tryIncrementalHNSWDelete(docIDs []string) (bool, error) {
	if len(docIDs) == 0 || !getHNSWEnabled() || c.streamEmbeddingsOnQuery {
		return false, nil
	}

	c.hnswBuildLock.Lock()
	defer c.hnswBuildLock.Unlock()

	if c.hnsw == nil {
		return false, nil
	}
	if c.hnswIndexedVersion.Load() != c.hnswVersion.Load() {
		return false, nil
	}

	nextIndex := c.hnsw.clone()
	changed := false
	for _, docID := range docIDs {
		if nextIndex.markDeleted(docID) {
			changed = true
		}
	}
	if !changed {
		return true, nil
	}

	newVersion := c.hnswVersion.Add(1)
	c.hnsw = nextIndex
	c.hnswIndexedVersion.Store(newVersion)
	if err := c.persistHNSWIndex(nextIndex); err != nil {
		return false, fmt.Errorf("couldn't persist hnsw index: %w", err)
	}

	return true, nil
}

func (c *Collection) removePersistedHNSWIndexBestEffort() {
	if c.persistDirectory == "" {
		return
	}
	_ = removeFile(c.getHNSWIndexPath())
}

// getDocumentsListLocked returns a cached snapshot of all document pointers.
// Caller must hold either documentsLock.RLock or documentsLock.Lock.
func (c *Collection) getDocumentsListLocked() []*Document {
	if c.docsListValid {
		return c.documentsList
	}

	docs := make([]*Document, 0, len(c.documents))
	for _, doc := range c.documents {
		docs = append(docs, doc)
	}

	c.documentsList = docs
	c.docsListValid = true
	return c.documentsList
}

func (c *Collection) getDocumentsListSnapshot() []*Document {
	c.documentsLock.RLock()
	if c.docsListValid {
		docs := c.documentsList
		c.documentsLock.RUnlock()
		return docs
	}
	c.documentsLock.RUnlock()

	c.documentsLock.Lock()
	docs := c.getDocumentsListLocked()
	c.documentsLock.Unlock()
	return docs
}

// Count returns the number of documents in the collection.
func (c *Collection) Count() int {
	c.documentsLock.RLock()
	defer c.documentsLock.RUnlock()
	return len(c.documents)
}

// Result represents a single result from a query.
type Result struct {
	ID        string
	Metadata  map[string]string
	Embedding []float32
	Content   string

	// The cosine similarity between the query and the document.
	// The higher the value, the more similar the document is to the query.
	// The value is in the range [-1, 1].
	Similarity float32
}

// Query performs an exhaustive nearest neighbor search on the collection.
//
//   - queryText: The text to search for. Its embedding will be created using the
//     collection's embedding function.
//   - nResults: The maximum number of results to return. Must be > 0.
//     There can be fewer results if a filter is applied.
//   - where: Conditional filtering on metadata. Optional.
//   - whereDocument: Conditional filtering on documents. Optional.
func (c *Collection) Query(ctx context.Context, queryText string, nResults int, where, whereDocument map[string]string) ([]Result, error) {
	if queryText == "" {
		return nil, errors.New("queryText is empty")
	}

	queryVector, err := c.embed(ctx, queryText)
	if err != nil {
		return nil, fmt.Errorf("couldn't create embedding of query: %w", err)
	}

	return c.QueryEmbedding(ctx, queryVector, nResults, where, whereDocument)
}

// QueryWithOptions performs an exhaustive nearest neighbor search on the collection.
//
//   - options: The options for the query. See [QueryOptions] for more information.
func (c *Collection) QueryWithOptions(ctx context.Context, options QueryOptions) ([]Result, error) {
	if options.QueryText == "" && len(options.QueryEmbedding) == 0 {
		return nil, errors.New("QueryText and QueryEmbedding options are empty")
	}

	var err error
	queryVector := options.QueryEmbedding
	if len(queryVector) == 0 {
		queryVector, err = c.embed(ctx, options.QueryText)
		if err != nil {
			return nil, fmt.Errorf("couldn't create embedding of query: %w", err)
		}
	}

	negativeFilterThreshold := options.Negative.FilterThreshold
	negativeVector := options.Negative.Embedding
	if len(negativeVector) == 0 && options.Negative.Text != "" {
		negativeVector, err = c.embed(ctx, options.Negative.Text)
		if err != nil {
			return nil, fmt.Errorf("couldn't create embedding of negative: %w", err)
		}
	}

	if len(negativeVector) != 0 {
		if !isNormalized(negativeVector) {
			negativeVector = normalizeVector(negativeVector)
		}

		switch options.Negative.Mode {
		case NEGATIVE_MODE_SUBTRACT:
			queryVector = subtractVector(queryVector, negativeVector)
			queryVector = normalizeVector(queryVector)
		case NEGATIVE_MODE_FILTER:
			if negativeFilterThreshold == 0 {
				negativeFilterThreshold = DEFAULT_NEGATIVE_FILTER_THRESHOLD
			}
		default:
			return nil, fmt.Errorf("unsupported negative mode: %q", options.Negative.Mode)
		}
	}

	result, err := c.queryEmbedding(ctx, queryVector, negativeVector, negativeFilterThreshold, options.NResults, options.Where, options.WhereDocument)
	if err != nil {
		return nil, err
	}

	return result, nil
}

// QueryEmbedding performs an exhaustive nearest neighbor search on the collection.
//
//   - queryEmbedding: The embedding of the query to search for. It must be created
//     with the same embedding model as the document embeddings in the collection.
//     The embedding will be normalized if it's not the case yet.
//   - nResults: The maximum number of results to return. Must be > 0.
//     There can be fewer results if a filter is applied.
//   - where: Conditional filtering on metadata. Optional.
//   - whereDocument: Conditional filtering on documents. Optional.
func (c *Collection) QueryEmbedding(ctx context.Context, queryEmbedding []float32, nResults int, where, whereDocument map[string]string) ([]Result, error) {
	return c.queryEmbedding(ctx, queryEmbedding, nil, 0, nResults, where, whereDocument)
}

// queryEmbedding performs an exhaustive nearest neighbor search on the collection.
func (c *Collection) queryEmbedding(ctx context.Context, queryEmbedding, negativeEmbeddings []float32, negativeFilterThreshold float32, nResults int, where, whereDocument map[string]string) ([]Result, error) {
	if len(queryEmbedding) == 0 {
		return nil, errors.New("queryEmbedding is empty")
	}
	if nResults <= 0 {
		return nil, errors.New("nResults must be > 0")
	}

	// Validate whereDocument operators
	for k := range whereDocument {
		if !slices.Contains(supportedFilters, k) {
			return nil, errors.New("unsupported operator")
		}
	}

	docs := c.getDocumentsListSnapshot()
	if nResults > len(docs) {
		return nil, errors.New("nResults must be <= the number of documents in the collection")
	}

	if len(docs) == 0 {
		return nil, nil
	}

	filteredDocs := docs
	shouldReleaseFilteredDocs := false
	if len(where) > 0 || len(whereDocument) > 0 {
		if err := c.ensureAllDocumentPayloadLoaded(); err != nil {
			return nil, fmt.Errorf("couldn't load document payloads for filtering: %w", err)
		}
		filteredDocs, shouldReleaseFilteredDocs = filterDocs(docs, where, whereDocument)
	}
	if shouldReleaseFilteredDocs {
		defer releaseDocumentSlice(filteredDocs)
	}

	// No need to continue if the filters got rid of all documents
	if len(filteredDocs) == 0 {
		return nil, nil
	}

	// Normalize embedding if not the case yet. We only support cosine similarity
	// for now and all documents were already normalized when added to the collection.
	if !isNormalized(queryEmbedding) {
		queryEmbedding = normalizeVector(queryEmbedding)
	}

	// If the filtering already reduced the number of documents to fewer than nResults,
	// we only need to find the most similar docs among the filtered ones.
	resLen := min(len(filteredDocs), nResults)

	// For the remaining documents, get the most similar docs.
	var nMaxDocs []docSim
	var err error
	if c.canUseHNSW(where, whereDocument, negativeEmbeddings, negativeFilterThreshold) {
		nMaxDocs, err = c.getMostSimilarDocsHNSW(queryEmbedding, resLen, filteredDocs)
		if err != nil {
			return nil, fmt.Errorf("couldn't query hnsw index: %w", err)
		}
	}
	if len(nMaxDocs) == 0 {
		if c.streamEmbeddingsOnQuery {
			nMaxDocs, err = c.getMostSimilarDocsStreaming(ctx, queryEmbedding, negativeEmbeddings, negativeFilterThreshold, filteredDocs, resLen)
		} else {
			nMaxDocs, err = getMostSimilarDocs(ctx, queryEmbedding, negativeEmbeddings, negativeFilterThreshold, filteredDocs, resLen)
		}
	}
	if err != nil {
		return nil, fmt.Errorf("couldn't get most similar docs: %w", err)
	}

	res := make([]Result, 0, len(nMaxDocs))
	for i := range nMaxDocs {
		doc := nMaxDocs[i].doc
		embedding := doc.Embedding
		if len(embedding) == 0 {
			embedding, err = c.ensureDocumentEmbeddingLoaded(doc, !c.streamEmbeddingsOnQuery)
			if err != nil {
				return nil, fmt.Errorf("couldn't load document embedding for '%s': %w", doc.ID, err)
			}
		}
		if err := c.ensureDocumentPayloadLoaded(doc); err != nil {
			return nil, fmt.Errorf("couldn't load document payload for '%s': %w", doc.ID, err)
		}
		res = append(res, Result{
			ID:         doc.ID,
			Metadata:   doc.Metadata,
			Embedding:  embedding,
			Content:    doc.Content,
			Similarity: nMaxDocs[i].similarity,
		})
	}

	return res, nil
}

func (c *Collection) canUseHNSW(where, whereDocument map[string]string, negativeEmbeddings []float32, negativeFilterThreshold float32) bool {
	if !getHNSWEnabled() {
		return false
	}
	if c.streamEmbeddingsOnQuery {
		return false
	}
	if len(where) > 0 || len(whereDocument) > 0 {
		return false
	}
	if len(negativeEmbeddings) > 0 || negativeFilterThreshold > 0 {
		return false
	}
	return true
}

func (c *Collection) getMostSimilarDocsHNSW(queryEmbedding []float32, nResults int, docs []*Document) ([]docSim, error) {
	if nResults <= 0 || len(docs) == 0 {
		return nil, nil
	}

	if err := c.ensureHNSWIndexReady(); err != nil {
		return nil, err
	}

	c.hnswBuildLock.Lock()
	idx := c.hnsw
	c.hnswBuildLock.Unlock()
	if idx == nil {
		return nil, nil
	}

	if c.shouldCompactHNSW(idx) {
		if err := c.compactHNSWIndex(); err != nil {
			return nil, err
		}
		c.hnswBuildLock.Lock()
		idx = c.hnsw
		c.hnswBuildLock.Unlock()
		if idx == nil {
			return nil, nil
		}
	}

	neighbors, err := idx.Search(queryEmbedding, nResults)
	if err != nil {
		return nil, err
	}
	if len(neighbors) < nResults {
		return nil, nil
	}

	out := make([]docSim, 0, len(neighbors))
	for _, n := range neighbors {
		out = append(out, docSim{doc: n.doc, similarity: n.similarity})
	}

	return out, nil
}

// shouldCompactHNSW returns true when the index has accumulated enough tombstones
// that query quality/latency can degrade due to stale graph topology.
//
// We use a two-condition trigger:
//  1. deleted nodes >= CHROMEM_HNSW_TOMBSTONE_REBUILD_MIN_DELETED
//  2. deleted/total >= CHROMEM_HNSW_TOMBSTONE_REBUILD_RATIO
//
// This avoids frequent rebuilds on small collections while still recovering graph
// quality for long-lived, mutation-heavy workloads.
func (c *Collection) shouldCompactHNSW(idx *hnswIndex) bool {
	if idx == nil {
		return false
	}

	ratioThreshold := getHNSWTombstoneRebuildRatio()
	if ratioThreshold <= 0 {
		return false
	}

	deleted := idx.deletedCount()
	if deleted == 0 || deleted < getHNSWTombstoneRebuildMinDeleted() {
		return false
	}

	total := len(idx.nodes)
	if total == 0 {
		return false
	}

	ratio := float64(deleted) / float64(total)
	return ratio >= ratioThreshold
}

// compactHNSWIndex compacts the current index by rebuilding it from live documents
// only, removing tombstoned historical nodes.
//
// This function is intentionally called lazily from the query path so write-heavy
// phases can stay fast (incremental upserts/deletes), while read phases regain
// graph quality once tombstones grow beyond thresholds.
func (c *Collection) compactHNSWIndex() error {
	c.hnswBuildLock.Lock()
	defer c.hnswBuildLock.Unlock()

	idx := c.hnsw
	if !c.shouldCompactHNSW(idx) {
		return nil
	}

	if err := c.ensureAllDocumentEmbeddingsLoaded(); err != nil {
		return err
	}

	c.documentsLock.RLock()
	docSnapshot := slices.Clone(c.getDocumentsListLocked())
	c.documentsLock.RUnlock()
	if len(docSnapshot) == 0 {
		c.hnsw = nil
		c.hnswIndexedVersion.Store(c.hnswVersion.Load())
		return nil
	}

	dim := len(docSnapshot[0].Embedding)
	if dim == 0 {
		return errors.New("couldn't compact hnsw index: first document embedding is empty")
	}

	newIndex := newHNSWIndex(dim, getHNSWM(), getHNSWEFConstruction(), getHNSWEFSearch())
	if err := newIndex.Build(docSnapshot); err != nil {
		return fmt.Errorf("couldn't compact hnsw index: %w", err)
	}

	currentVersion := c.hnswVersion.Load()
	c.hnsw = newIndex
	c.hnswIndexedVersion.Store(currentVersion)
	if err := c.persistHNSWIndex(newIndex); err != nil {
		return fmt.Errorf("couldn't persist compacted hnsw index: %w", err)
	}

	return nil
}

func (c *Collection) ensureHNSWIndexReady() error {
	currentVersion := c.hnswVersion.Load()
	if c.hnswIndexedVersion.Load() == currentVersion {
		c.hnswBuildLock.Lock()
		ready := c.hnsw != nil
		c.hnswBuildLock.Unlock()
		if ready {
			return nil
		}
	}

	c.hnswBuildLock.Lock()
	defer c.hnswBuildLock.Unlock()

	currentVersion = c.hnswVersion.Load()
	if c.hnswIndexedVersion.Load() == currentVersion && c.hnsw != nil {
		return nil
	}

	if err := c.ensureAllDocumentEmbeddingsLoaded(); err != nil {
		return err
	}

	c.documentsLock.RLock()
	docSnapshot := slices.Clone(c.getDocumentsListLocked())
	c.documentsLock.RUnlock()
	if len(docSnapshot) == 0 {
		c.hnsw = nil
		c.hnswIndexedVersion.Store(currentVersion)
		return nil
	}

	dim := len(docSnapshot[0].Embedding)
	if dim == 0 {
		return errors.New("couldn't build hnsw index: first document embedding is empty")
	}

	loaded, err := c.tryLoadPersistedHNSWIndex(docSnapshot, dim, currentVersion)
	if err != nil {
		return err
	}
	if loaded {
		return nil
	}

	idx := newHNSWIndex(dim, getHNSWM(), getHNSWEFConstruction(), getHNSWEFSearch())
	if err := idx.Build(docSnapshot); err != nil {
		return err
	}

	currentVersion = c.hnswVersion.Load()
	c.hnsw = idx
	c.hnswIndexedVersion.Store(currentVersion)
	_ = c.persistHNSWIndex(idx)

	return nil
}

func (c *Collection) tryLoadPersistedHNSWIndex(docSnapshot []*Document, dim int, currentVersion uint64) (bool, error) {
	if c.persistDirectory == "" {
		return false, nil
	}

	persisted := persistedHNSWIndex{}
	if err := readFromFile(c.getHNSWIndexPath(), &persisted, ""); err != nil {
		return false, nil
	}

	if persisted.Dim != dim || len(persisted.Nodes) != len(docSnapshot) {
		c.removePersistedHNSWIndexBestEffort()
		return false, nil
	}
	deletedBitmap := persisted.DeletedBitmap
	if len(deletedBitmap) == 0 && len(persisted.Deleted) > 0 {
		for i, isDeleted := range persisted.Deleted {
			if !isDeleted {
				continue
			}
			wordIndex := i / 64
			for len(deletedBitmap) <= wordIndex {
				deletedBitmap = append(deletedBitmap, 0)
			}
			deletedBitmap[wordIndex] |= uint64(1) << uint(i%64)
		}
	}
	requiredWords := (len(persisted.Nodes) + 63) / 64
	if len(deletedBitmap) < requiredWords {
		deletedBitmap = append(deletedBitmap, make([]uint64, requiredWords-len(deletedBitmap))...)
	}

	docByID := make(map[string]*Document, len(docSnapshot))
	for _, doc := range docSnapshot {
		docByID[doc.ID] = doc
	}

	nodes := make([]hnswNode, len(persisted.Nodes))
	for i, node := range persisted.Nodes {
		doc, ok := docByID[node.DocID]
		if !ok {
			c.removePersistedHNSWIndexBestEffort()
			return false, nil
		}
		if node.Level < 0 || len(node.Neighbors) != node.Level+1 {
			c.removePersistedHNSWIndexBestEffort()
			return false, nil
		}

		neighbors := make([][]int, len(node.Neighbors))
		for level := range node.Neighbors {
			neighbors[level] = slices.Clone(node.Neighbors[level])
			for _, neighborID := range neighbors[level] {
				if neighborID < 0 || neighborID >= len(persisted.Nodes) {
					c.removePersistedHNSWIndexBestEffort()
					return false, nil
				}
			}
		}

		nodes[i] = hnswNode{doc: doc, level: node.Level, neighbors: neighbors}
	}

	if persisted.EntryPoint < -1 || persisted.EntryPoint >= len(nodes) {
		c.removePersistedHNSWIndexBestEffort()
		return false, nil
	}

	idx := newHNSWIndex(
		persisted.Dim,
		persisted.M,
		persisted.EFConstruction,
		persisted.EFSearch,
	)
	idx.nodes = nodes
	idx.deletedBitmap = slices.Clone(deletedBitmap)
	idx.entryPoint = persisted.EntryPoint
	idx.maxLevel = persisted.MaxLevel
	idx.latestNodeByDocID = make(map[string]int, len(nodes))
	for i, node := range nodes {
		if idx.isDeleted(i) {
			continue
		}
		idx.latestNodeByDocID[node.doc.ID] = i
	}

	c.hnsw = idx
	c.hnswIndexedVersion.Store(currentVersion)
	return true, nil
}

func (c *Collection) persistHNSWIndex(idx *hnswIndex) error {
	if c.persistDirectory == "" || idx == nil {
		return nil
	}

	nodes := make([]persistedHNSWNode, 0, len(idx.nodes))
	for _, node := range idx.nodes {
		neighbors := make([][]int, len(node.neighbors))
		for i := range node.neighbors {
			neighbors[i] = slices.Clone(node.neighbors[i])
		}
		nodes = append(nodes, persistedHNSWNode{
			DocID:     node.doc.ID,
			Level:     node.level,
			Neighbors: neighbors,
		})
	}

	persisted := persistedHNSWIndex{
		Dim:            idx.dim,
		M:              idx.m,
		EFConstruction: idx.efConstruction,
		EFSearch:       idx.efSearch,
		EntryPoint:     idx.entryPoint,
		MaxLevel:       idx.maxLevel,
		DeletedBitmap:  slices.Clone(idx.deletedBitmap),
		Nodes:          nodes,
	}

	return persistToFile(c.getHNSWIndexPath(), persisted, c.compress, "")
}

func (c *Collection) ensureDocumentPayloadLoaded(doc *Document) error {
	if doc == nil || doc.payloadLoaded {
		return nil
	}
	if doc.persistPath == "" {
		doc.payloadLoaded = true
		return nil
	}

	c.documentsLock.Lock()
	defer c.documentsLock.Unlock()

	if doc.payloadLoaded {
		return nil
	}

	tmp := struct {
		Metadata map[string]string
		Content  string
	}{}
	if err := readFromFile(doc.persistPath, &tmp, ""); err != nil {
		return fmt.Errorf("couldn't read document payload from %q: %w", doc.persistPath, err)
	}

	doc.Metadata = tmp.Metadata
	doc.Content = tmp.Content
	doc.payloadLoaded = true
	return nil
}

func (c *Collection) ensureDocumentEmbeddingLoaded(doc *Document, cache bool) ([]float32, error) {
	if doc == nil {
		return nil, errors.New("document is nil")
	}
	if len(doc.Embedding) > 0 {
		return doc.Embedding, nil
	}
	if doc.persistPath == "" {
		return nil, fmt.Errorf("document '%s' has no embedding in memory and no persistence path", doc.ID)
	}

	tmp := struct {
		Embedding []float32
	}{}
	if err := readFromFile(doc.persistPath, &tmp, ""); err != nil {
		return nil, fmt.Errorf("couldn't read document embedding from %q: %w", doc.persistPath, err)
	}
	if len(tmp.Embedding) == 0 {
		return nil, fmt.Errorf("document '%s' embedding is empty", doc.ID)
	}

	if cache {
		c.documentsLock.Lock()
		if len(doc.Embedding) == 0 {
			doc.Embedding = tmp.Embedding
		}
		embedding := doc.Embedding
		c.documentsLock.Unlock()
		return embedding, nil
	}

	return tmp.Embedding, nil
}

func (c *Collection) ensureAllDocumentPayloadLoaded() error {
	docs := c.getDocumentsListSnapshot()

	for _, doc := range docs {
		if err := c.ensureDocumentPayloadLoaded(doc); err != nil {
			return err
		}
	}
	return nil
}

func (c *Collection) ensureAllDocumentEmbeddingsLoaded() error {
	docs := c.getDocumentsListSnapshot()

	for _, doc := range docs {
		if _, err := c.ensureDocumentEmbeddingLoaded(doc, true); err != nil {
			return err
		}
	}
	return nil
}

func (c *Collection) getMostSimilarDocsStreaming(ctx context.Context, queryVectors, negativeVector []float32, negativeFilterThreshold float32, docs []*Document, n int) ([]docSim, error) {
	numDocs := len(docs)
	vectorDim := len(queryVectors)
	if negativeFilterThreshold > 0 && len(negativeVector) != vectorDim {
		return nil, fmt.Errorf("couldn't calculate negative similarity: vectors must have the same length")
	}

	concurrency := queryConcurrency(numDocs, len(queryVectors))
	if concurrency == 0 {
		return nil, nil
	}

	if concurrency == 1 || numDocs < getQuerySequentialDocsThreshold() {
		localMaxDocs := newMaxDocSims(n)
		for _, doc := range docs {
			embedding, err := c.ensureDocumentEmbeddingLoaded(doc, false)
			if err != nil {
				return nil, err
			}
			if len(embedding) != vectorDim {
				return nil, fmt.Errorf("couldn't calculate similarity for document '%s': vectors must have the same length", doc.ID)
			}

			sim := float32(0)
			if negativeFilterThreshold > 0 {
				nsim := float32(0)
				sim, nsim = dotProductPairScalar(queryVectors, negativeVector, embedding)
				if nsim > negativeFilterThreshold {
					continue
				}
			} else {
				sim = dotProductOptimized(queryVectors, embedding)
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
		if sharedErr == nil {
			sharedErr = err
			cancel(sharedErr)
		}
	}

	wg := sync.WaitGroup{}
	resultsChan := make(chan []docSim, concurrency)
	chunkSize := queryChunkSize(numDocs)
	for i := range concurrency {
		wg.Add(1)
		go func(workerIndex int) {
			defer wg.Done()
			localMaxDocs := newMaxDocSims(n)
			workerStart, workerEnd := workerRange(numDocs, concurrency, workerIndex)
			for start := workerStart; start < workerEnd; start += chunkSize {
				if ctx.Err() != nil {
					break
				}
				end := min(start+chunkSize, workerEnd)
				for _, doc := range docs[start:end] {
					if ctx.Err() != nil {
						break
					}

					embedding, err := c.ensureDocumentEmbeddingLoaded(doc, false)
					if err != nil {
						setSharedErr(fmt.Errorf("couldn't load embedding for document '%s': %w", doc.ID, err))
						break
					}
					if len(embedding) != vectorDim {
						setSharedErr(fmt.Errorf("couldn't calculate similarity for document '%s': vectors must have the same length", doc.ID))
						break
					}

					sim := float32(0)
					if negativeFilterThreshold > 0 {
						nsim := float32(0)
						sim, nsim = dotProductPairScalar(queryVectors, negativeVector, embedding)
						if nsim > negativeFilterThreshold {
							continue
						}
					} else {
						sim = dotProductOptimized(queryVectors, embedding)
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

// getDocPath generates the path to the document file.
func (c *Collection) getDocPath(docID string) string {
	safeID := hash2hex(docID)
	docPath := filepath.Join(c.persistDirectory, safeID)
	docPath += ".gob"
	if c.compress {
		docPath += ".gz"
	}
	return docPath
}

func (c *Collection) getHNSWIndexPath() string {
	return filepath.Join(c.persistDirectory, hnswIndexFileName)
}

// persistMetadata persists the collection metadata to disk
func (c *Collection) persistMetadata() error {
	// Persist name and metadata
	metadataPath := filepath.Join(c.persistDirectory, metadataFileName)
	metadataPath += ".gob"
	if c.compress {
		metadataPath += ".gz"
	}
	pc := struct {
		Name     string
		Metadata map[string]string
	}{
		Name:     c.Name,
		Metadata: c.metadata,
	}
	err := persistToFile(metadataPath, pc, c.compress, "")
	if err != nil {
		return err
	}

	return nil
}

// cloneDocument creates a deep copy of the given Document, including its Metadata and Embedding slices.
func cloneDocument(doc *Document) *Document {
	docCopy := *doc
	docCopy.Metadata = maps.Clone(doc.Metadata)
	docCopy.Embedding = slices.Clone(doc.Embedding)
	return &docCopy
}

// makePartialDocument creates a copy of the given Document without its Metadata and Embedding slices.
func makePartialDocument(doc *Document) *Document {
	docCopy := *doc
	docCopy.Metadata = nil
	docCopy.Embedding = nil
	return &docCopy
}
