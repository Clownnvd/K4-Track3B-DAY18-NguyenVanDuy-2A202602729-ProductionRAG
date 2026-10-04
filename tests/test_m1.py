"""Tests for Module 1: Advanced Chunking."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m1_chunking import (chunk_basic, chunk_semantic, chunk_hierarchical,
                              chunk_structure_aware, compare_strategies, load_documents, Chunk)

TEXT = """# Nghỉ phép

## Nghỉ phép năm

Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm.
Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên.

## Nghỉ phép không lương

Nhân viên có thể xin nghỉ phép không lương tối đa 30 ngày mỗi năm.
Đơn xin nghỉ phải được Giám đốc bộ phận phê duyệt.

## Nghỉ ốm

Cần nộp giấy xác nhận y tế trong vòng 3 ngày làm việc."""


# --- Baseline (đã implement sẵn) ---

def test_basic_returns_chunks():
    assert len(chunk_basic(TEXT)) > 0

def test_basic_type():
    assert all(isinstance(c, Chunk) for c in chunk_basic(TEXT))


# --- Semantic Chunking ---

def test_semantic_returns_chunks():
    result = chunk_semantic(TEXT, threshold=0.5)
    assert len(result) > 0, "Semantic chunking should return chunks"

def test_semantic_type():
    assert all(isinstance(c, Chunk) for c in chunk_semantic(TEXT, 0.5))

def test_semantic_groups_by_topic():
    """Semantic should produce fewer chunks than basic (groups related sentences)."""
    basic = chunk_basic(TEXT, chunk_size=100)
    semantic = chunk_semantic(TEXT, threshold=0.5)
    assert len(semantic) <= len(basic) + 2  # Allow some tolerance


# --- Hierarchical Chunking ---

def test_hierarchical_returns_both():
    parents, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    assert len(parents) > 0, "Should return parents"
    assert len(children) > 0, "Should return children"

def test_hierarchical_children_have_parent_id():
    _, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    for c in children:
        assert c.parent_id is not None, "Each child must have parent_id"

def test_hierarchical_valid_parent_ids():
    parents, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    parent_ids = {p.metadata.get("parent_id") for p in parents}
    for c in children:
        assert c.parent_id in parent_ids, f"Child parent_id '{c.parent_id}' not in parents"

def test_hierarchical_children_smaller():
    parents, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    avg_p = sum(len(p.text) for p in parents) / max(len(parents), 1)
    avg_c = sum(len(c.text) for c in children) / max(len(children), 1)
    assert avg_c < avg_p, "Children should be smaller than parents"


# --- Structure-Aware Chunking ---

def test_structure_returns_chunks():
    result = chunk_structure_aware(TEXT)
    assert len(result) > 0, "Structure-aware should return chunks"

def test_structure_preserves_headers():
    result = chunk_structure_aware(TEXT)
    texts = " ".join(c.text for c in result)
    assert "Nghỉ phép năm" in texts, "Should preserve section headers"

def test_structure_has_section_metadata():
    result = chunk_structure_aware(TEXT)
    if result:
        assert any("section" in c.metadata for c in result), "Should have section in metadata"


# --- Compare ---

def test_compare_all_strategies():
    docs = load_documents()
    if docs:
        r = compare_strategies(docs)
        for key in ["basic", "semantic", "hierarchical", "structure"]:
            assert key in r, f"Missing strategy: {key}"
