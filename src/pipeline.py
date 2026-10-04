"""Production RAG: child retrieval, parent context, real generation and evaluation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks, extract_metadata
from src.llm import generate_answer
from config import RERANK_TOP_K

TIMINGS = {}


def build_pipeline():
    documents = load_documents()
    chunks, parents = [], {}
    start = time.perf_counter()
    for doc in documents:
        metadata = {**extract_metadata(doc['text']), **doc['metadata']}
        parent_chunks, children = chunk_hierarchical(doc['text'], metadata=metadata)
        parents.update({p.metadata['parent_id']: p.text for p in parent_chunks})
        chunks.extend({'text': c.text, 'metadata': {**c.metadata, 'parent_id': c.parent_id}} for c in children)
    TIMINGS['chunking_seconds'] = time.perf_counter()-start
    print(f'Đã chia {len(documents)} tài liệu thành {len(chunks)} đoạn con và {len(parents)} đoạn cha.', flush=True)
    start = time.perf_counter()
    enriched = enrich_chunks(chunks)
    indexed = [{'text': e.enriched_text, 'metadata': e.auto_metadata} for e in enriched]
    TIMINGS['enrichment_seconds'] = time.perf_counter()-start
    start = time.perf_counter()
    search = HybridSearch()
    search.index(indexed)
    # Cache public question vectors, not answers, to release BGE-M3 before CE.
    search.dense.prepare_queries([item['question'] for item in load_test_set()])
    search.parents = parents
    search.pipeline_info = {'num_documents': len(documents), 'num_children': len(chunks), 'num_parents': len(parents),
                            'enrichment_modes': sorted({e.auto_metadata['enrichment_mode'] for e in enriched})}
    TIMINGS['indexing_seconds'] = time.perf_counter()-start
    return search, CrossEncoderReranker()


def run_query(query, search, reranker):
    start = time.perf_counter()
    results = search.search(query)
    retrieved_ms = (time.perf_counter()-start)*1000
    documents = [{'text': r.text, 'score': r.score, 'metadata': r.metadata} for r in results]
    start = time.perf_counter()
    ranked = reranker.rerank(query, documents, top_k=RERANK_TOP_K)
    rerank_ms = (time.perf_counter()-start)*1000
    contexts, seen = [], set()
    for result in ranked:
        pid = result.metadata.get('parent_id')
        original = getattr(search, 'parents', {}).get(pid, result.metadata.get('original_text', result.text))
        if original not in seen:
            seen.add(original)
            contexts.append(original)
    start = time.perf_counter()
    answer = generate_answer(query, contexts)
    search.last_trace = {'retrieval_ms': retrieved_ms, 'rerank_ms': rerank_ms,
                         'generation_ms': (time.perf_counter()-start)*1000,
                         'sources': [r.metadata.get('source') for r in ranked],
                         'parent_ids': [r.metadata.get('parent_id') for r in ranked],
                         'context_count': len(contexts)}
    return answer, contexts


def evaluate_pipeline(search, reranker):
    test_set = load_test_set()
    questions, answers, contexts, truths, traces = [], [], [], [], []
    for index, item in enumerate(test_set):
        answer, evidence = run_query(item['question'], search, reranker)
        questions.append(item['question'])
        answers.append(answer)
        contexts.append(evidence)
        truths.append(item['ground_truth'])
        traces.append({'question': item['question'], **search.last_trace})
        print(f'Đã trả lời {index+1}/{len(test_set)} câu hỏi.', flush=True)
    start = time.perf_counter()
    results = evaluate_ragas(questions, answers, contexts, truths)
    results['pipeline'] = search.pipeline_info
    results['timings'] = {**TIMINGS, 'evaluation_seconds': time.perf_counter()-start}
    results['query_traces'] = traces
    save_report(results, failure_analysis(results['per_question'], bottom_n=5))
    return results


if __name__ == '__main__':
    search, reranker = build_pipeline()
    result = evaluate_pipeline(search, reranker)
    if not result.get('scores_valid'):
        raise SystemExit('Chưa có đủ điểm RAGAS hợp lệ; xem trạng thái trong báo cáo.')
