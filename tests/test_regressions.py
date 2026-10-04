"""Behavioral checks for provenance, hierarchy and honest evaluation status."""
from types import SimpleNamespace
import json
from src.m1_chunking import chunk_hierarchical, chunk_structure_aware
from src.m2_search import SearchResult, reciprocal_rank_fusion
from src.m3_rerank import RerankResult
from src.m4_eval import evaluate_ragas, save_report
from src import pipeline


def test_parent_ids_do_not_collide_between_documents():
    text = 'Một nội dung tài liệu có đủ thông tin và cần được bảo toàn. '*10
    first, children = chunk_hierarchical(text, 100, 35, {'source': 'first.md'})
    second, other = chunk_hierarchical(text, 100, 35, {'source': 'second.md'})
    assert {p.metadata['parent_id'] for p in first}.isdisjoint({p.metadata['parent_id'] for p in second})
    assert all(len(c.text) <= 35 for c in children+other)
    assert all(len(p.text) <= 100 for p in first+second)


def test_structure_does_not_split_code_fence_or_table():
    text = '# Section\n\n```python\n# comment is not a header\nprint(1)\n```\n\n| x | y |\n| --- | --- |\n| 1 | 2 |\n\n## Next\nBody'
    chunks = chunk_structure_aware(text)
    assert len(chunks) == 2
    assert '# comment is not a header' in chunks[0].text
    assert '| 1 | 2 |' in chunks[0].text


def test_rrf_duplicate_in_one_ranking_does_not_inflate_score():
    item = SearchResult('quoted', 2.0, {}, 'bm25')
    merged = reciprocal_rank_fusion([[item, item], [item]])
    assert len(merged) == 1 and abs(merged[0].score-2/61) < 1e-12


def test_query_returns_unique_original_parent_context(monkeypatch):
    child = SearchResult('enriched child', 0.2, {'source': 'source.md', 'parent_id': 'p'}, 'hybrid')
    search = SimpleNamespace(search=lambda _: [child, child], parents={'p': 'Original full parent with a verified condition.'})
    ranked = RerankResult(child.text, 0.2, 1.0, child.metadata, 0)
    reranker = SimpleNamespace(rerank=lambda *a, **kw: [ranked, ranked])
    seen = []
    monkeypatch.setattr(pipeline, 'generate_answer', lambda q, contexts: seen.append((q, contexts)) or 'answer')
    answer, contexts = pipeline.run_query('User question', search, reranker)
    assert contexts == [search.parents['p']] and seen == [('User question', contexts)]
    assert answer == 'answer'


def test_missing_judge_does_not_claim_valid_scores(monkeypatch):
    from src import m4_eval
    monkeypatch.setattr(m4_eval, 'ENABLE_LLM_EVAL', False)
    result = evaluate_ragas(['q'], ['actual answer'], [['context']], ['reference'])
    assert result['scores_valid'] is False and result['status'] == 'disabled'
    assert result['per_question'] == []


def test_report_never_writes_nonstandard_nan_json(tmp_path):
    path = tmp_path/'report.json'
    save_report({'faithfulness': float('nan'), 'per_question': [], 'scores_valid': False}, [], str(path))
    report = json.loads(path.read_text(encoding='utf-8'))
    assert report['aggregate']['faithfulness'] is None


def test_gemini_rate_limit_retries_without_parallel_calls(monkeypatch):
    from src import llm

    class RateLimitError(Exception):
        status_code = 429

    class Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RateLimitError('quota')
            return 'ok'

    completions = Completions()
    api = type('API', (), {'chat': type('Chat', (), {'completions': completions})()})()
    sleeps = []
    monkeypatch.setenv('GEMINI_MIN_INTERVAL_SECONDS', '0')
    monkeypatch.setattr(llm.time, 'sleep', sleeps.append)
    assert llm._request_with_rate_limit(api, model='judge') == 'ok'
    assert completions.calls == 2
    assert sleeps == [20]
