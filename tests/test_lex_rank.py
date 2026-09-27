import numpy as np
import pytest

from src.lex_rank import degree_centrality_scores


def test_degree_centrality_scores_returns_vector():
    scores = degree_centrality_scores(np.eye(3), threshold=0.5)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (3,)


def test_degree_centrality_scores_rejects_bad_threshold():
    with pytest.raises(ValueError):
        degree_centrality_scores(np.eye(3), threshold=1.5)
    # None is explicitly allowed.
    assert degree_centrality_scores(np.eye(3), threshold=None).shape == (3,)


def test_degree_centrality_scores_prefers_central_node():
    similarity = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    scores = degree_centrality_scores(similarity, threshold=0.5)
    assert int(np.argmax(scores)) == 1
