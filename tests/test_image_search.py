import numpy as np
import pytest

from src.image_search import DEFAULT_TOP_K, ImageSearch
from src.summarizer import is_model_available  # real check used via seams


class FakeClip:
    """Encodes images from a per-call queue and text into a fixed query vector."""

    def __init__(self, image_vectors=None, query_vector=None):
        self.image_calls = 0
        self.seen_queries = []
        self._image_vectors = list(image_vectors or [])
        self._query_vector = np.asarray(query_vector, dtype=float)

    def encode(self, payload, convert_to_numpy=True):
        if isinstance(payload, str) or (
            isinstance(payload, list) and payload and isinstance(payload[0], str)
        ):
            self.seen_queries.extend(payload if isinstance(payload, list) else [payload])
            return self._query_vector
        self.image_calls += 1
        return np.asarray(self._image_vectors[self.image_calls - 1], dtype=float)


def make_search(image_vectors=None, query_vector=(1.0, 0.0), **kwargs):
    fake = FakeClip(image_vectors=image_vectors, query_vector=query_vector)
    instance = ImageSearch("clip", model_factory=lambda path, device: fake, **kwargs)
    return instance, fake


def make_png(path, color=(255, 0, 0)):
    from PIL import Image

    Image.new("RGB", (2, 2), color).save(path, format="PNG")


# --------------------------------------------------------------------- search


def test_search_ranks_by_cosine_similarity(tmp_path):
    img0 = tmp_path / "img0.png"; make_png(img0)
    img1 = tmp_path / "img1.png"; make_png(img1)
    img2 = tmp_path / "img2.png"; make_png(img2)
    # cos(img0, q)=0.6, cos(img1, q)=1.0, cos(img2, q)=0.0
    instance, _ = make_search(
        image_vectors=[[0.6, 0.8], [1.0, 0.0], [0.0, 1.0]], query_vector=[1.0, 0.0]
    )
    result = instance.search(tmp_path, "two dogs in snow")
    assert [path for _, path in result] == [
        str(img1.resolve()), str(img0.resolve()), str(img2.resolve())
    ]
    assert [score for score, _ in result] == pytest.approx([1.0, 0.6, 0.0])


def test_search_top_k_limits_results(tmp_path):
    for i in range(3):
        make_png(tmp_path / f"img{i}.png")
    instance, _ = make_search(image_vectors=[[1, 0], [0.9, 0.1], [0.5, 0.5]], top_k=2)
    result = instance.search(tmp_path, "q")
    assert len(result) == 2
    assert result[0][1].endswith("img0.png")
    assert result[0][0] == pytest.approx(1.0)
    assert result[0][0] >= result[1][0]  # descending scores


def test_search_top_k_clamped_to_image_count(tmp_path):
    for i in range(2):
        make_png(tmp_path / f"img{i}.png")
    instance, _ = make_search(image_vectors=[[1, 0], [0.9, 0.1]], top_k=1)
    result = instance.search(tmp_path, "q", top_k=99)
    assert len(result) == 2


def test_search_skips_unreadable_and_unsupported_files(tmp_path):
    make_png(tmp_path / "good.png")
    (tmp_path / "broken.png").write_bytes(b"not a png")
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")
    instance, fake = make_search(image_vectors=[[1.0, 0.0]])
    assert instance.search(tmp_path, "q") == [
        (pytest.approx(1.0), str((tmp_path / "good.png").resolve()))
    ]
    assert fake.image_calls == 1


def test_search_empty_folder_returns_empty_list(tmp_path):
    instance, _ = make_search()
    assert instance.search(tmp_path, "q") == []


def test_search_is_recursive(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    make_png(sub / "nested.png")
    instance, _ = make_search(image_vectors=[[1.0, 0.0]])
    assert instance.search(tmp_path, "q") == [
        (pytest.approx(1.0), str((sub / "nested.png").resolve()))
    ]


# ------------------------------------------------------------ input handling


def test_search_empty_query_raises():
    instance, _ = make_search()
    with pytest.raises(ValueError):
        instance.search("anywhere", "   ")


def test_search_missing_directory_raises():
    instance, _ = make_search()
    with pytest.raises(FileNotFoundError):
        instance.search("nope/missing", "q")


def test_search_file_as_directory_raises(tmp_path):
    target = tmp_path / "file.png"
    make_png(target)
    instance, _ = make_search()
    with pytest.raises(NotADirectoryError):
        instance.search(target, "q")


def test_search_accepts_extension_overrides(tmp_path):
    make_png(tmp_path / "photo.custom")  # real PNG bytes, custom extension
    make_png(tmp_path / "real.png")
    instance, _ = make_search(image_vectors=[[1, 0]], extensions=["custom"])
    assert instance.search(tmp_path, "q") == [
        (pytest.approx(1.0), str((tmp_path / "photo.custom").resolve()))
    ]


# -------------------------------------------------------------------- caching


def test_image_embeddings_cached_until_mtime_changes(tmp_path):
    make_png(tmp_path / "a.png")
    instance, fake = make_search(image_vectors=[[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])

    instance.search(tmp_path, "first")
    instance.search(tmp_path, "second")
    assert fake.image_calls == 1  # unchanged folder -> no re-encoding

    import os

    os.utime(tmp_path / "a.png", (1_000_000_000, 1_000_000_000))
    instance.search(tmp_path, "third")
    assert fake.image_calls == 2


def test_broken_image_not_cached(tmp_path):
    make_png(tmp_path / "a.png")
    instance, fake = make_search(image_vectors=[[1.0, 0.0]])
    broken = tmp_path / "b.png"
    broken.write_bytes(b"garbage")
    instance.search(tmp_path, "q1")
    broken.write_bytes(b"still garbage")
    instance.search(tmp_path, "q2")
    assert fake.image_calls == 1  # only the readable image was ever encoded


# ----------------------------------------------------------------- model seam


def test_missing_model_raises_friendly_error(monkeypatch):
    import src.image_search as image_search

    monkeypatch.setattr(image_search, "is_model_available", lambda path: False)
    instance = ImageSearch("missing-clip")
    with pytest.raises(FileNotFoundError) as exc:
        instance._encoder()
    assert "missing-clip" in str(exc.value)


def test_defaults_match_tool_spec():
    assert DEFAULT_TOP_K == 3