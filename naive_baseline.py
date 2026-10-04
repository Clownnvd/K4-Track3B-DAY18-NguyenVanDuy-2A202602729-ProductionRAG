"""Basic baseline: paragraph chunks, dense retrieval; same generator and judge."""
import time
from src.m1_chunking import load_documents, chunk_basic
from src.m2_search import DenseSearch
from src.m4_eval import load_test_set, evaluate_ragas, save_report
from src.llm import generate_answer
from config import NAIVE_COLLECTION


def main():
    chunks = []
    start = time.perf_counter()
    for doc in load_documents():
        chunks.extend({'text': c.text, 'metadata': c.metadata}
                      for c in chunk_basic(doc['text'], metadata=doc['metadata']))
    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)
    test_set = load_test_set()
    search.prepare_queries([item['question'] for item in test_set])
    build_seconds = time.perf_counter()-start
    questions, answers, contexts, truths, traces = [], [], [], [], []
    for index, item in enumerate(test_set):
        started = time.perf_counter()
        hits = search.search(item['question'], top_k=3, collection=NAIVE_COLLECTION)
        evidence = [r.text for r in hits]
        answer = generate_answer(item['question'], evidence)
        questions.append(item['question'])
        answers.append(answer)
        contexts.append(evidence)
        truths.append(item['ground_truth'])
        traces.append({'question': item['question'], 'elapsed_ms': (time.perf_counter()-started)*1000,
                       'sources': [r.metadata.get('source') for r in hits]})
        print(f'Bản cơ sở: đã trả lời {index+1}/{len(test_set)} câu hỏi.', flush=True)
    results = evaluate_ragas(questions, answers, contexts, truths)
    results['pipeline'] = {'strategy': 'paragraph+dense', 'num_chunks': len(chunks)}
    results['timings'] = {'build_seconds': build_seconds}
    results['query_traces'] = traces
    save_report(results, [], 'reports/naive_baseline_report.json')
    return results


if __name__ == '__main__':
    main()
