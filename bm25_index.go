package chromem

import (
	"cmp"
	"math"
	"slices"
	"strings"
	"unicode"
)

const (
	bm25K1 = 1.2
	bm25B  = 0.75
)

type bm25Posting struct {
	docIndex int
	tf       int
}

type bm25Index struct {
	docs       []*Document
	postings   map[string][]bm25Posting
	docLengths []int
	avgDocLen  float64
}

func newBM25Index() *bm25Index {
	return &bm25Index{}
}

func (idx *bm25Index) Build(docs []*Document) {
	idx.docs = docs
	idx.postings = make(map[string][]bm25Posting, len(docs)*2)
	idx.docLengths = make([]int, len(docs))
	idx.avgDocLen = 0

	if len(docs) == 0 {
		return
	}

	totalLen := 0
	for docIndex, doc := range docs {
		tokens := tokenizeBM25(doc.Content)
		docLen := len(tokens)
		idx.docLengths[docIndex] = docLen
		totalLen += docLen
		if docLen == 0 {
			continue
		}

		tfByTerm := make(map[string]int, docLen)
		for _, token := range tokens {
			tfByTerm[token]++
		}
		for term, tf := range tfByTerm {
			idx.postings[term] = append(idx.postings[term], bm25Posting{docIndex: docIndex, tf: tf})
		}
	}

	idx.avgDocLen = float64(totalLen) / float64(len(docs))
}

func (idx *bm25Index) Search(query string, k int) []hnswNeighbor {
	if idx == nil || k <= 0 || len(idx.docs) == 0 {
		return nil
	}

	terms := tokenizeBM25(query)
	if len(terms) == 0 {
		return nil
	}

	queryTF := make(map[string]int, len(terms))
	for _, term := range terms {
		queryTF[term]++
	}

	totalDocs := float64(len(idx.docs))
	scores := make(map[int]float64, min(len(idx.docs), k*8))

	for term, qtf := range queryTF {
		postings := idx.postings[term]
		if len(postings) == 0 {
			continue
		}

		df := float64(len(postings))
		idf := math.Log(1 + (totalDocs-df+0.5)/(df+0.5))
		qBoost := float64(qtf)
		for _, posting := range postings {
			docLen := float64(idx.docLengths[posting.docIndex])
			tf := float64(posting.tf)
			norm := tf + bm25K1*(1-bm25B+bm25B*docLen/max(idx.avgDocLen, 1e-9))
			if norm <= 0 {
				continue
			}
			scores[posting.docIndex] += idf * ((tf * (bm25K1 + 1)) / norm) * qBoost
		}
	}

	if len(scores) == 0 {
		return nil
	}

	type scoredDoc struct {
		docIndex int
		score    float64
	}
	ranked := make([]scoredDoc, 0, len(scores))
	for docIndex, score := range scores {
		ranked = append(ranked, scoredDoc{docIndex: docIndex, score: score})
	}
	slices.SortFunc(ranked, func(a, b scoredDoc) int {
		return cmp.Compare(b.score, a.score)
	})

	if len(ranked) > k {
		ranked = ranked[:k]
	}

	maxScore := ranked[0].score
	if maxScore <= 0 {
		maxScore = 1
	}

	out := make([]hnswNeighbor, 0, len(ranked))
	for _, item := range ranked {
		normalized := float32(item.score / maxScore)
		out = append(out, hnswNeighbor{doc: idx.docs[item.docIndex], similarity: normalized})
	}

	return out
}

func tokenizeBM25(content string) []string {
	if content == "" {
		return nil
	}

	normalized := strings.ToLower(content)
	tokens := make([]string, 0, len(normalized)/6)
	start := -1
	for i, r := range normalized {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			if start < 0 {
				start = i
			}
			continue
		}
		if start >= 0 {
			tokens = append(tokens, normalized[start:i])
			start = -1
		}
	}
	if start >= 0 {
		tokens = append(tokens, normalized[start:])
	}

	return tokens
}
