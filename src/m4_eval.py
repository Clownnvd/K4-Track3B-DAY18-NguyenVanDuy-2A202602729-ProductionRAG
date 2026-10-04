"""Module 4: genuine RAGAS scores; disabled/error results are explicitly labeled."""
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from config import TEST_SET_PATH, ENABLE_LLM_EVAL, GEMINI_API_KEY, EVAL_MODEL

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

METRICS = ('faithfulness', 'answer_relevancy', 'context_precision', 'context_recall')


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path=TEST_SET_PATH):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def evaluate_ragas(questions, answers, contexts, ground_truths):
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError('Danh sách câu hỏi, câu trả lời, ngữ cảnh và đáp án chuẩn phải cùng độ dài.')
    unavailable = {**{name: 0.0 for name in METRICS}, 'per_question': [], 'status': 'disabled',
                   'judge': EVAL_MODEL, 'scores_valid': False}
    if not questions:
        return {**unavailable, 'status': 'empty'}
    if not ENABLE_LLM_EVAL or not GEMINI_API_KEY:
        return {**unavailable, 'reason': 'Chưa bật đánh giá hoặc thiếu GEMINI_API_KEY; các số 0 không phải điểm đo.'}
    try:
        import numpy as np
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from src.judge import GeminiJudge
        judge = GeminiJudge()
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cpu'},
                                           encode_kwargs={'normalize_embeddings': True})
        dataset = Dataset.from_dict({'question': questions, 'answer': answers,
                                     'contexts': contexts, 'ground_truth': ground_truths})
        # Clone metric instances; RAGAS may attach clients during evaluation.
        import copy
        metrics = [copy.deepcopy(m) for m in (faithfulness, answer_relevancy, context_precision, context_recall)]
        # One generated question is sufficient for this 20-item lab set and keeps
        # the free-tier judge below its request budget. The metric itself remains
        # RAGAS AnswerRelevancy; only its sampling strictness is reduced from 3.
        for metric in metrics:
            if hasattr(metric, 'strictness'):
                metric.strictness = 1
        result = evaluate(dataset, metrics=metrics,
                          llm=judge, embeddings=LangchainEmbeddingsWrapper(embeddings),
                          run_config=RunConfig(max_workers=1, timeout=120, max_retries=2),
                          raise_exceptions=False)
        frame = result.to_pandas()
        items, missing = [], []
        for index, row in frame.iterrows():
            values = {}
            for name in METRICS:
                value = float(row.get(name, float('nan')))
                if not math.isfinite(value):
                    missing.append({'index': int(index), 'metric': name})
                values[name] = value
            items.append(EvalResult(str(row['question']), str(row['answer']), list(row['contexts']),
                                    str(row['ground_truth']), **values))
        aggregate = {name: float(np.nanmean([getattr(item, name) for item in items]))
                     if any(math.isfinite(getattr(item, name)) for item in items) else 0.0 for name in METRICS}
        return {**aggregate, 'per_question': items, 'status': 'complete' if not missing else 'partial',
                'scores_valid': not missing, 'missing_metrics': missing, 'judge': EVAL_MODEL,
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2', 'ragas_version': '0.1.x'}
    except Exception as exc:
        # Do not print transport errors containing URLs/credentials.
        return {**unavailable, 'status': 'error', 'error_type': type(exc).__name__,
                'reason': 'Đánh giá thật thất bại. Không thay bằng điểm giả hoặc điểm từ đáp án chuẩn.'}


def failure_analysis(eval_results, bottom_n=10):
    tree = {
        'faithfulness': ('Câu trả lời có nội dung không được tài liệu đỡ.', 'Siết hướng dẫn chỉ dùng bằng chứng và giữ trích dẫn.'),
        'context_recall': ('Thiếu đoạn chứa dữ kiện cần thiết.', 'Kiểm tra đoạn cha, thêm tìm kiếm từ khóa và dữ liệu còn thiếu.'),
        'context_precision': ('Đoạn không phù hợp đứng trước bằng chứng cần dùng.', 'Kiểm tra xếp hạng lại, phiên bản và bộ lọc siêu dữ liệu.'),
        'answer_relevancy': ('Câu trả lời chưa giải quyết đúng điều được hỏi.', 'Viết hướng dẫn trả lời trực tiếp và làm rõ câu hỏi nhập nhằng.'),
    }
    failures = []
    for index, item in enumerate(eval_results):
        values = {name: getattr(item, name) for name in METRICS}
        if not all(math.isfinite(value) for value in values.values()):
            failures.append({'question': item.question, 'worst_metric': 'evaluation_error', 'score': None,
                             'diagnosis': 'Một phép chấm không có kết quả hợp lệ.',
                             'suggested_fix': 'Kiểm tra quota, timeout và JSON trả về; chấm lại trước khi kết luận chất lượng.',
                             'error_tree': 'Có đủ điểm đo? -> Không -> kiểm tra tầng đánh giá', 'index': index})
            continue
        worst = min(values, key=values.get)
        diagnosis, fix = tree[worst]
        failures.append({'question': item.question, 'answer': item.answer, 'ground_truth': item.ground_truth,
                         'contexts': item.contexts, 'worst_metric': worst,
                         'score': sum(values.values())/4, 'metrics': values, 'diagnosis': diagnosis,
                         'suggested_fix': fix, 'index': index,
                         'error_tree': f'Câu trả lời đúng? -> Kiểm tra bằng chứng -> Chỉ số thấp nhất: {worst} -> {fix}'})
    failures.sort(key=lambda row: (row['score'] is not None, row['score'] if row['score'] is not None else 0, row['index']))
    return failures[:max(0, bottom_n)]


def save_report(results, failures, path='reports/ragas_report.json'):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite(item) for item in value]
        return value
    per_question = [asdict(item) for item in results.get('per_question', [])]
    report = {'aggregate': {key: value for key, value in results.items() if key != 'per_question'},
              'num_questions': len(per_question), 'per_question': per_question, 'failures': failures}
    target.write_text(json.dumps(finite(report), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Đã lưu báo cáo: {target}', flush=True)


if __name__ == '__main__':
    print(f'Có {len(load_test_set())} câu hỏi trong bộ kiểm thử.')
