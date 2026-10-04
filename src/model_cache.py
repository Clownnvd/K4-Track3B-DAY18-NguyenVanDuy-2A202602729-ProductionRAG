"""Share model loads and release them explicitly on small-memory machines."""
import gc
import os
from functools import lru_cache

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('HF_ENABLE_PARALLEL_LOADING', 'false')
os.environ.setdefault('HF_PARALLEL_LOADING_WORKERS', '1')


def configure_torch():
    import torch
    torch.set_num_threads(int(os.getenv('TORCH_NUM_THREADS', '2')))


@lru_cache(maxsize=2)
def sentence_model(name):
    from sentence_transformers import SentenceTransformer
    configure_torch()
    cache = os.getenv('BGE_CACHE_DIR') if name == 'BAAI/bge-m3' else None
    return SentenceTransformer(name, cache_folder=cache, token=False)


@lru_cache(maxsize=1)
def cross_model(name):
    from sentence_transformers import CrossEncoder
    configure_torch()
    return CrossEncoder(name, max_length=512)


def release_sentence_models():
    sentence_model.cache_clear()
    gc.collect()
