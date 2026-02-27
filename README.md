# chromem-go

[![Go Reference](https://pkg.go.dev/badge/github.com/TIANLI0/chromem-go.svg)](https://pkg.go.dev/github.com/TIANLI0/chromem-go)
[![Build status](https://github.com/TIANLI0/chromem-go/actions/workflows/go.yml/badge.svg)](https://github.com/TIANLI0/chromem-go/actions/workflows/go.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/TIANLI0/chromem-go)](https://goreportcard.com/report/github.com/TIANLI0/chromem-go)
[![GitHub Releases](https://img.shields.io/github/release/TIANLI0/chromem-go.svg)](https://github.com/TIANLI0/chromem-go/releases)

Embeddable vector database for Go with Chroma-like interface and zero third-party dependencies. In-memory with optional persistence.

This repository is a performance-focused fork of the original project at <https://github.com/philippgille/chromem-go>.

Because `chromem-go` is embeddable it enables you to add retrieval augmented generation (RAG) and similar embeddings-based features into your Go app *without having to run a separate database*. Like when using SQLite instead of PostgreSQL/MySQL/etc.

It's *not* a library to connect to Chroma and also not a reimplementation of it in Go. It's a database on its own.

The focus is not scale (millions of documents) or number of features, but simplicity and performance for the most common use cases. This fork adds post-`f63964a64bf64b261f665dd45f92cafadcb0b972` query-path optimizations and SIMD controls; see [Differences vs upstream (after `f63964a`)](#differences-vs-upstream-after-f63964a) and [Benchmarks](#benchmarks).

> ⚠️ The project is in beta, under heavy construction, and may introduce breaking changes in releases before `v1.0.0`. All changes are documented in the [`CHANGELOG`](./CHANGELOG.md).

## Contents

1. [Use cases](#use-cases)
2. [Interface](#interface)
3. [Differences vs upstream (after `f63964a`)](#differences-vs-upstream-after-f63964a)
4. [Features + Roadmap](#features)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Benchmarks](#benchmarks)
8. [Development](#development)
9. [Motivation](#motivation)
10. [Related projects](#related-projects)

## Use cases

With a vector database you can do various things:

- Retrieval augmented generation (RAG), question answering (Q&A)
- Text and code search
- Recommendation systems
- Classification
- Clustering

Let's look at the RAG use case in more detail:

### RAG

The knowledge of large language models (LLMs) - even the ones with 30 billion, 70 billion parameters and more - is limited. They don't know anything about what happened after their training ended, they don't know anything about data they were not trained with (like your company's intranet, Jira / bug tracker, wiki or other kinds of knowledge bases), and even the data they *do* know they often can't reproduce it *exactly*, but start to *hallucinate* instead.

Fine-tuning an LLM can help a bit, but it's more meant to improve the LLMs reasoning about specific topics, or reproduce the style of written text or code. Fine-tuning does *not* add knowledge *1:1* into the model. Details are lost or mixed up. And knowledge cutoff (about anything that happened after the fine-tuning) isn't solved either.

=> A vector database can act as the up-to-date, precise knowledge for LLMs:

1. You store relevant documents that you want the LLM to know in the database.
2. The database stores the *embeddings* alongside the documents, which you can either provide or can be created by specific "embedding models" like OpenAI's `text-embedding-3-small`.
   - `chromem-go` can do this for you and supports multiple embedding providers and models out-of-the-box.
3. Later, when you want to talk to the LLM, you first send the question to the vector DB to find *similar*/*related* content. This is called "nearest neighbor search".
4. In the question to the LLM, you provide this content alongside your question.
5. The LLM can take this up-to-date precise content into account when answering.

Check out the [example code](examples) to see it in action!

## Interface

Our original inspiration was the [Chroma](https://www.trychroma.com/) interface, whose core API is the following (taken from their [README](https://github.com/chroma-core/chroma/blob/0.4.21/README.md)):

<details><summary>Chroma core interface</summary>

```python
import chromadb
# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.Client()

# Create collection. get_collection, get_or_create_collection, delete_collection also available!
collection = client.create_collection("all-my-documents")

# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=["This is document1", "This is document2"], # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
    metadatas=[{"source": "notion"}, {"source": "google-docs"}], # filter on these!
    ids=["doc1", "doc2"], # unique for each doc
)

# Query/search 2 most similar results. You can also .get by id
results = collection.query(
    query_texts=["This is a query document"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)
```

</details>

Our Go library exposes the same interface:

<details><summary>chromem-go equivalent</summary>

```go
package main

import "github.com/TIANLI0/chromem-go"

func main() {
    // Set up chromem-go in-memory, for easy prototyping. Can add persistence easily!
    // We call it DB instead of client because there's no client-server separation. The DB is embedded.
    db := chromem.NewDB()

    // Create collection. GetCollection, GetOrCreateCollection, DeleteCollection also available!
    collection, _ := db.CreateCollection("all-my-documents", nil, nil)

    // Add docs to the collection. Update and delete will be added in the future.
    // Can be multi-threaded with AddConcurrently()!
    // We're showing the Chroma-like method here, but more Go-idiomatic methods are also available!
    _ = collection.Add(ctx,
        []string{"doc1", "doc2"}, // unique ID for each doc
        nil, // We handle embedding automatically. You can skip that and add your own embeddings as well.
        []map[string]string{{"source": "notion"}, {"source": "google-docs"}}, // Filter on these!
        []string{"This is document1", "This is document2"},
    )

    // Query/search 2 most similar results. You can also get by ID.
    results, _ := collection.Query(ctx,
        "This is a query document",
        2,
        map[string]string{"metadata_field": "is_equal_to_this"}, // optional filter
        map[string]string{"$contains": "search_string"},         // optional filter
    )
}
```

</details>

Initially `chromem-go` started with just the four core methods, but we added more over time. We intentionally don't want to cover 100% of Chroma's API surface though.  
We're providing some alternative methods that are more Go-idiomatic instead.

For the full interface see the Godoc: <https://pkg.go.dev/github.com/TIANLI0/chromem-go>

## Differences vs upstream (after `f63964a`)

Compared to the upstream baseline at commit `f63964a64bf64b261f665dd45f92cafadcb0b972`, this fork currently includes:

- Query execution internals reworked for lower latency:
  - chunk-based worker scheduling
  - per-worker top-k aggregation before final merge (less contention)
  - tuned concurrency heuristics based on document count and vector dimension
  - cached document snapshots to reduce lock contention under high query concurrency
  - pooled filtered-document slices to reduce query-time allocations
- Runtime tuning knobs for query behavior:
  - `CHROMEM_QUERY_SMALL_DOCS_THRESHOLD`
  - `CHROMEM_QUERY_SEQUENTIAL_DOCS_THRESHOLD`
  - `CHROMEM_QUERY_HIGH_DIM_THRESHOLD`
  - `CHROMEM_QUERY_HIGH_DIM_CONCURRENCY_DIVISOR`
  - `CHROMEM_QUERY_MAX_CONCURRENCY` (hard cap; `0` disables cap)
  - plus matching setter APIs (`SetQuery...`)
- Optional SIMD path for dot product (amd64 + `GOEXPERIMENT=simd`) with runtime threshold control:
  - env var: `CHROMEM_SIMD_MIN_LENGTH`
  - API: `SetSIMDMinLength()`
- Collection-level memory observability API:
  - `Collection.MemoryStats()`
- Reproducible benchmark workflow and matrix script:
  - `benchmark_matrix.ps1`

### Performance snapshot vs `f63964a`

Measured on this machine (`windows/amd64`, Intel i7-14700F), with:

```console
go test -run=^$ -bench "BenchmarkCollection_Query_NoContent_(1000|5000|25000|100000)$|BenchmarkDotProduct" -benchmem -benchtime=200ms -count=4 ./...
```

Average query latency (`ns/op`) improved versus `f63964a`:

- `1000` docs: `-56.33%`
- `5000` docs: `-68.95%`
- `25000` docs: `-42.94%`
- `100000` docs: `-38.61%`

At the same time, memory overhead increased for these scenarios (`B/op`, `allocs/op`), which is a deliberate speed/overhead trade-off in the current implementation.

Additional SIMD-only dot-product check on `HEAD` (`-cpu=1`, `CHROMEM_SIMD_MIN_LENGTH=0`) shows `optimized` path gains over no-SIMD build:

- `size=1024`: `-35.78%`
- `size=1536`: `-38.79%`
- `size=3072`: `-54.95%`

You can reproduce the same comparison by running the benchmark commands in [Development](#development), then comparing outputs with `benchstat` or equivalent summary tooling.

### High-concurrency snapshot (1GiB @ 1536 dims)

Measured on this machine (`windows/amd64`, Intel i7-14700F) with SIMD enabled and ~1GiB embeddings-only corpus:

```console
go test -run ^$ -bench "^BenchmarkCollection_Query_NoContent_1536_Approx1GiB_ParallelLatencyMatrix$" -benchmem -benchtime=1x -count=1
```

Observed range:

- workers=1: ~21.9 QPS, p50 ~45.5ms, p95 ~49.5ms
- workers=4: ~26.5 QPS, p50 ~145.9ms, p95 ~205.0ms
- workers=8: ~27.5 QPS, p50 ~254.8ms, p95 ~362.0ms
- workers=16: ~25.9 QPS, p95 ~1010ms
- workers=32: ~28.2 QPS, p95 ~1304ms

Interpretation: throughput plateaus around 4-8 workers while tail latency rises rapidly beyond that.

### Persistent mode comparison (1GiB @ 1536 dims)

Measured on this machine (`windows/amd64`, Intel i7-14700F):

```console
go test -run '^$' -bench '^BenchmarkCollection_Query_NoContent_1536_Approx1GiB_PersistentModes$' -benchmem -benchtime 1x -count 1 .
```

Observed query-time results (`nResults=10`):

- `default_preload`: `47.38ms/op`, `1.49MB/op`, `3,976 allocs/op`
- `lazy_payload`: `46.80ms/op`, `1.71MB/op`, `5,933 allocs/op`
- `stream_embeddings`: `1050.32ms/op`, `4.87GB/op`, `32,339,674 allocs/op`

Interpretation:

- `default_preload` and `lazy_payload` are close for embedding-only workloads (payload is tiny/empty here).
- `stream_embeddings` massively reduces resident embedding memory but trades off a lot of query throughput and increases allocations due to per-query disk reads and decode overhead.
- Recommended production default for balanced performance is still preload (or `lazy_payload` when content/metadata memory dominates).

Quick mode selection:

| Goal | Mode | Recommended options | Trade-off |
| --- | --- | --- | --- |
| Lowest query latency / highest throughput | `default_preload` | `PersistentDBOptions{Compress:false}` | Highest resident memory usage |
| Lower startup memory, similar query speed | `lazy_payload` | `PersistentDBOptions{Compress:false, LazyLoadPayload:true}` | First access to content/metadata may read from disk |
| Minimum resident memory | `stream_embeddings` | `PersistentDBOptions{Compress:false, LazyLoadPayload:true, StreamEmbeddingsOnQuery:true}` | Large query slowdown and much higher per-query allocations |

For most production workloads: start with `lazy_payload`, benchmark with your real data, and only use `stream_embeddings` when memory pressure is the top priority.

3-step decision flow:

1. If your dataset comfortably fits RAM, start with `default_preload`.
2. If startup memory is high but query latency still matters, switch to `lazy_payload`.
3. If RAM is still not enough, move to `stream_embeddings` and accept lower query throughput.

After choosing a mode, re-run the benchmark with your real filters/content mix and tune query concurrency (`CHROMEM_QUERY_MAX_CONCURRENCY`) for your latency target.

## Features

- [X] Zero dependencies on third party libraries
- [X] Embeddable (like SQLite, i.e. no client-server model, no separate DB to maintain)
- [X] Multithreaded processing (when adding and querying documents), making use of Go's native concurrency features
- [X] Experimental WebAssembly binding
- Embedding creators:
  - Hosted:
    - [X] [OpenAI](https://platform.openai.com/docs/guides/embeddings/embedding-models) (default)
    - [X] [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/embeddings)
    - [X] [GCP Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings)
    - [X] [Cohere](https://cohere.com/models/embed)
    - [X] [Mistral](https://docs.mistral.ai/platform/endpoints/#embedding-models)
    - [X] [Jina](https://jina.ai/embeddings)
    - [X] [mixedbread.ai](https://www.mixedbread.ai/)
  - Local:
    - [X] [Ollama](https://github.com/ollama/ollama)
    - [X] [LocalAI](https://github.com/mudler/LocalAI)
  - Bring your own (implement [`chromem.EmbeddingFunc`](https://pkg.go.dev/github.com/TIANLI0/chromem-go#EmbeddingFunc))
  - You can also pass existing embeddings when adding documents to a collection, instead of letting `chromem-go` create them
- Similarity search:
  - [X] Exhaustive nearest neighbor search using cosine similarity (sometimes also called exact search or brute-force search or FLAT index)
- Filters:
  - [X] Document filters: `$contains`, `$not_contains`
  - [X] Metadata filters: Exact matches
- Storage:
  - [X] In-memory
  - [X] Optional immediate persistence (writes one file for each added collection and document, encoded as [gob](https://go.dev/blog/gob), optionally gzip-compressed)
  - [X] Backups: Export and import of the entire DB to/from a single file (encoded as [gob](https://go.dev/blog/gob), optionally gzip-compressed and AES-GCM encrypted)
    - Includes methods for generic `io.Writer`/`io.Reader` so you can plug S3 buckets and other blob storage, see [examples/s3-export-import](examples/s3-export-import) for example code
- Observability:
  - [X] Collection memory stats (`Collection.MemoryStats()`)
- Data types:
  - [X] Documents (text)

### Roadmap

- Performance:
  - Further tune SIMD thresholds and defaults across CPU architectures
  - Add [roaring bitmaps](https://github.com/RoaringBitmap/roaring) to speed up full text filtering
- Embedding creators:
  - Add an `EmbeddingFunc` that downloads and shells out to [llamafile](https://github.com/Mozilla-Ocho/llamafile)
- Similarity search:
  - Approximate nearest neighbor search with index (ANN)
    - Hierarchical Navigable Small World (HNSW)
    - Inverted file flat (IVFFlat)
- Filters:
  - Operators (`$and`, `$or` etc.)
- Storage:
  - JSON as second encoding format
  - Write-ahead log (WAL) as second file format
  - Optional remote storage (S3, PostgreSQL, ...)
- Data types:
  - Images
  - Videos

## Installation

`go get github.com/TIANLI0/chromem-go@latest`

If you want the original upstream instead of this fork, use:

`go get github.com/philippgille/chromem-go@latest`

## Usage

See the Godoc for a reference: <https://pkg.go.dev/github.com/TIANLI0/chromem-go>

For full, working examples, using the vector database for retrieval augmented generation (RAG) and semantic search and using either OpenAI or locally running the embeddings model and LLM (in Ollama), see the [example code](examples).

### Quickstart

This is taken from the ["minimal" example](examples/minimal):

```go
package main

import (
 "context"
 "fmt"
 "runtime"

 "github.com/TIANLI0/chromem-go"
)

func main() {
  ctx := context.Background()

  db := chromem.NewDB()

  // Passing nil as embedding function leads to OpenAI being used and requires
  // "OPENAI_API_KEY" env var to be set. Other providers are supported as well.
  // For example pass `chromem.NewEmbeddingFuncOllama(...)` to use Ollama.
  c, err := db.CreateCollection("knowledge-base", nil, nil)
  if err != nil {
    panic(err)
  }

  err = c.AddDocuments(ctx, []chromem.Document{
    {
      ID:      "1",
      Content: "The sky is blue because of Rayleigh scattering.",
    },
    {
      ID:      "2",
      Content: "Leaves are green because chlorophyll absorbs red and blue light.",
    },
  }, runtime.NumCPU())
  if err != nil {
    panic(err)
  }

  res, err := c.Query(ctx, "Why is the sky blue?", 1, nil, nil)
  if err != nil {
    panic(err)
  }

  fmt.Printf("ID: %v\nSimilarity: %v\nContent: %v\n", res[0].ID, res[0].Similarity, res[0].Content)
}
```

Output:

```text
ID: 1
Similarity: 0.6833369
Content: The sky is blue because of Rayleigh scattering.
```

## Benchmarks

Benchmarked on 2024-03-17 with:

- Computer: Framework Laptop 13 (first generation, 2021)
- CPU: 11th Gen Intel Core i5-1135G7 (2020)
- Memory: 32 GB
- OS: Fedora Linux 39
  - Kernel: 6.7

```console
$ go test -benchmem -run=^$ -bench .
goos: linux
goarch: amd64
pkg: github.com/philippgille/chromem-go
cpu: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
BenchmarkCollection_Query_NoContent_100-8          13164      90276 ns/op     5176 B/op       95 allocs/op
BenchmarkCollection_Query_NoContent_1000-8          2142     520261 ns/op    13558 B/op      141 allocs/op
BenchmarkCollection_Query_NoContent_5000-8           561    2150354 ns/op    47096 B/op      173 allocs/op
BenchmarkCollection_Query_NoContent_25000-8          120    9890177 ns/op   211783 B/op      208 allocs/op
BenchmarkCollection_Query_NoContent_100000-8          30   39574238 ns/op   810370 B/op      232 allocs/op
BenchmarkCollection_Query_100-8                    13225      91058 ns/op     5177 B/op       95 allocs/op
BenchmarkCollection_Query_1000-8                    2226     519693 ns/op    13552 B/op      140 allocs/op
BenchmarkCollection_Query_5000-8                     550    2128121 ns/op    47108 B/op      173 allocs/op
BenchmarkCollection_Query_25000-8                    100   10063260 ns/op   211705 B/op      205 allocs/op
BenchmarkCollection_Query_100000-8                    30   39404005 ns/op   810295 B/op      229 allocs/op
PASS
ok   github.com/philippgille/chromem-go 28.402s
```

## Development

- Build: `go build ./...`
- Test: `go test -v -race -count 1 ./...`
- Benchmark:
  - `go test -benchmem -run=^$ -bench .` (add `> bench.out` or similar to write to a file)
  - With profiling: `go test -benchmem -run ^$ -cpuprofile cpu.out -bench .`
    - (profiles: `-cpuprofile`, `-memprofile`, `-blockprofile`, `-mutexprofile`)
- Compare benchmarks:
  1. Install `benchstat`: `go install golang.org/x/perf/cmd/benchstat@latest`
  2. Compare two benchmark results: `benchstat before.out after.out`

### Performance tuning (SIMD + concurrency)

The query path supports a SIMD-optimized dot product (Go `GOEXPERIMENT=simd`, AMD64) and adaptive multi-threaded scheduling.

- SIMD can be enabled for benchmarking via `GOEXPERIMENT=simd`.
- The runtime threshold for switching from scalar to SIMD dot product can be configured with env var `CHROMEM_SIMD_MIN_LENGTH` or programmatically with `chromem.SetSIMDMinLength()`.
- The default threshold is `1536`.
- Query concurrency can be tuned with:
  - `CHROMEM_QUERY_SMALL_DOCS_THRESHOLD`
  - `CHROMEM_QUERY_SEQUENTIAL_DOCS_THRESHOLD`
  - `CHROMEM_QUERY_HIGH_DIM_THRESHOLD`
  - `CHROMEM_QUERY_HIGH_DIM_CONCURRENCY_DIVISOR`
  - `CHROMEM_QUERY_MAX_CONCURRENCY` (`0` means no hard cap)
  - equivalent APIs: `SetQuerySmallDocsThreshold`, `SetQuerySequentialDocsThreshold`, `SetQueryHighDimThreshold`, `SetQueryHighDimConcurrencyDivisor`, `SetQueryMaxConcurrency`
- Persistent DB startup memory can be reduced with `NewPersistentDBWithOptions(..., PersistentDBOptions{LazyLoadPayload: true})`, which keeps embeddings in memory and loads content/metadata on demand.
- For an ultra-low-memory mode, set `StreamEmbeddingsOnQuery: true` to stream embeddings from disk during query instead of keeping them resident.

Based on benchmark runs on Intel i7-14700F:

- Dot product (`optimized`) is significantly faster for vectors `>= 1536` dimensions with the current defaults.
- End-to-end query performance also improves, but gains are smaller than raw dot-product gains because filtering, heap maintenance, scheduling, and memory bandwidth become dominant.
- A practical default is `CHROMEM_SIMD_MIN_LENGTH=1536` for balanced single-core and multi-core performance on this hardware.

For 1GiB / 1536-dim workloads, prefer running query concurrency around 4-8 workers for best throughput/latency tradeoff.

#### Recommended presets (copy/paste)

The following presets are good starting points for library users. Keep `CHROMEM_SIMD_MIN_LENGTH=1536` unless your benchmarks show a better value.

- Large persistent datasets with limited RAM:
  - use `NewPersistentDBWithOptions(path, chromem.PersistentDBOptions{Compress: false, LazyLoadPayload: true})`
  - pair with `CHROMEM_QUERY_MAX_CONCURRENCY=4..8` based on your latency budget

- Ultra-low-memory mode (accept lower query throughput):
  - use `NewPersistentDBWithOptions(path, chromem.PersistentDBOptions{Compress: false, LazyLoadPayload: true, StreamEmbeddingsOnQuery: true})`
  - useful when dataset size exceeds available RAM

- Low-latency API (stable p95/p99):
  - `CHROMEM_QUERY_MAX_CONCURRENCY=4`
  - `CHROMEM_QUERY_HIGH_DIM_CONCURRENCY_DIVISOR=2`
- Throughput-oriented batch/service:
  - `CHROMEM_QUERY_MAX_CONCURRENCY=8`
  - `CHROMEM_QUERY_HIGH_DIM_CONCURRENCY_DIVISOR=2`
- Conservative / unknown hardware:
  - `CHROMEM_QUERY_MAX_CONCURRENCY=0` (no hard cap)
  - `CHROMEM_QUERY_HIGH_DIM_CONCURRENCY_DIVISOR=2`

Programmatic equivalent:

```go
// Example: throughput-oriented profile.
chromem.SetQueryMaxConcurrency(8)
chromem.SetQueryHighDimConcurrencyDivisor(2)
```

Environment variables (no code changes in consuming app):

```bash
CHROMEM_SIMD_MIN_LENGTH=1536
CHROMEM_QUERY_MAX_CONCURRENCY=8
CHROMEM_QUERY_HIGH_DIM_CONCURRENCY_DIVISOR=2
```

#### Reproducible matrix benchmark

Use the included PowerShell script to benchmark baseline vs SIMD across CPU sets and thresholds:

```powershell
powershell -ExecutionPolicy Bypass -File .\benchmark_matrix.ps1
```

This runs:

- baseline (no SIMD)
- SIMD with thresholds `0`, `1024`, `1536`
- CPU sets `1` and `8`
- `benchstat` comparisons for each pair

Results are written to `bench-results/run-<timestamp>/compare-*.txt`.

If your benchmark outputs were created with an incompatible encoding, regenerate compare files with:

```powershell
powershell -ExecutionPolicy Bypass -File .\rebuild_compare.ps1 -RunDir .\bench-results\run-<timestamp>
```

## Motivation

In December 2023, when I wanted to play around with retrieval augmented generation (RAG) in a Go program, I looked for a vector database that could be embedded in the Go program, just like you would embed SQLite in order to not require any separate DB setup and maintenance. I was surprised when I didn't find any, given the abundance of embedded key-value stores in the Go ecosystem.

At the time most of the popular vector databases like Pinecone, Qdrant, Milvus, Chroma, Weaviate and others were not embeddable at all or only in Python or JavaScript/TypeScript.

Then I found [@eliben](https://github.com/eliben)'s [blog post](https://eli.thegreenplace.net/2023/retrieval-augmented-generation-in-go/) and [example code](https://github.com/eliben/code-for-blog/tree/eda87b87dad9ed8bd45d1c8d6395efba3741ed39/2023/go-rag-openai) which showed that with very little Go code you could create a very basic PoC of a vector database.

That's when I decided to build my own vector database, embeddable in Go, inspired by the ChromaDB interface. ChromaDB stood out for being embeddable (in Python), and by showing its core API in 4 commands on their README and on the landing page of their website.

## Related projects

- Shoutout to [@eliben](https://github.com/eliben) whose [blog post](https://eli.thegreenplace.net/2023/retrieval-augmented-generation-in-go/) and [example code](https://github.com/eliben/code-for-blog/tree/eda87b87dad9ed8bd45d1c8d6395efba3741ed39/2023/go-rag-openai) inspired me to start this project!
- [Chroma](https://github.com/chroma-core/chroma): Looking at Pinecone, Qdrant, Milvus, Weaviate and others, Chroma stood out by showing its core API in 4 commands on their README and on the landing page of their website. It was also putting the most emphasis on its embeddability (in Python).
- The big, full-fledged client-server-based vector databases for maximum scale and performance:
  - [Pinecone](https://www.pinecone.io/): Closed source
  - [Qdrant](https://github.com/qdrant/qdrant): Written in Rust, not embeddable in Go
  - [Milvus](https://github.com/milvus-io/milvus): Written in Go and C++, but not embeddable as of December 2023
  - [Weaviate](https://github.com/weaviate/weaviate): Written in Go, but not embeddable in Go as of March 2024 (only in Python and JavaScript/TypeScript and that's experimental)
- Some non-specialized SQL, NoSQL and Key-Value databases added support for storing vectors and (some of them) querying based on similarity:
  - [pgvector](https://github.com/pgvector/pgvector) extension for [PostgreSQL](https://www.postgresql.org/): Client-server model
  - [Redis](https://github.com/redis/redis) ([1](https://redis.io/docs/interact/search-and-query/query/vector-search/), [2](https://redis.io/docs/interact/search-and-query/advanced-concepts/vectors/)): Client-server model
  - [sqlite-vss](https://github.com/asg017/sqlite-vss) extension for [SQLite](https://www.sqlite.org/): Embedded, but the [Go bindings](https://github.com/asg017/sqlite-vss/tree/8fc44301843029a13a474d1f292378485e1fdd62/bindings/go) require CGO. There's a [CGO-free Go library](https://gitlab.com/cznic/sqlite) for SQLite, but then it's without the vector search extension.
  - [DuckDB](https://github.com/duckdb/duckdb) has a function to calculate cosine similarity ([1](https://duckdb.org/docs/sql/functions/nested)): Embedded, but the Go bindings use CGO
  - [MongoDB](https://github.com/mongodb/mongo)'s cloud platform offers a vector search product ([1](https://www.mongodb.com/products/platform/atlas-vector-search)): Client-server model
- Some libraries for vector similarity search:
  - [Faiss](https://github.com/facebookresearch/faiss): Written in C++; 3rd party Go bindings use CGO
  - [Annoy](https://github.com/spotify/annoy): Written in C++; Go bindings use CGO ([1](https://github.com/spotify/annoy/blob/2be37c9e015544be2cf60c431f0cccc076151a2d/README_GO.rst))
  - [USearch](https://github.com/unum-cloud/usearch): Written in C++; Go bindings use CGO
- Some orchestration libraries, inspired by the Python library [LangChain](https://github.com/langchain-ai/langchain), but with no or only rudimentary embedded vector DB:
  - [LangChain Go](https://github.com/tmc/langchaingo)
  - [LinGoose](https://github.com/henomis/lingoose)
  - [GoLC](https://github.com/hupe1980/golc)
