//! LexRank implementation (`src/lex_rank.py` port, from
//! https://github.com/crabcamp/lexrank/tree/dev), on `ndarray` instead of
//! numpy/scipy: connected components become union-find, softmax row-wise.

use ndarray::Array2;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LexRankError {
    #[error("'threshold' should be a floating-point number from the interval [0, 1) or None")]
    InvalidThreshold,
    #[error("'weights_matrix' should be square")]
    NotSquare,
}

/// Degree centrality scores of a similarity matrix (Python
/// `degree_centrality_scores`).
///
/// `threshold` in [0, 1) discretizes the weights; `None` uses raw weights.
pub fn degree_centrality_scores(
    similarity_matrix: &Array2<f64>,
    threshold: Option<f64>,
) -> Result<Vec<f64>, LexRankError> {
    if let Some(threshold) = threshold {
        if !(0.0..1.0).contains(&threshold) {
            return Err(LexRankError::InvalidThreshold);
        }
        let markov = create_markov_matrix_discrete(similarity_matrix, threshold)?;
        Ok(stationary_distribution(&markov, true, false))
    } else {
        let markov = create_markov_matrix(similarity_matrix)?;
        Ok(stationary_distribution(&markov, true, false))
    }
}

/// Power iteration for the stationary distribution (Python `_power_method`).
fn power_method(
    transition_matrix: &Array2<f64>,
    increase_power: bool,
    max_iter: usize,
) -> Vec<f64> {
    let n = transition_matrix.nrows();
    let mut eigenvector = vec![1.0; n];
    if n == 1 {
        return eigenvector;
    }
    let mut transition = transition_matrix.t().to_owned();
    let mut eigenvector_next = mat_vec(&transition, &eigenvector);
    for _ in 0..max_iter {
        eigenvector_next = mat_vec(&transition, &eigenvector);
        if all_close(&eigenvector_next, &eigenvector) {
            return eigenvector_next;
        }
        eigenvector = eigenvector_next.clone();
        if increase_power {
            transition = transition.dot(&transition);
        }
    }
    log::warn!("Maximum number of iterations for power method exceeded without convergence!");
    eigenvector_next
}

fn mat_vec(matrix: &Array2<f64>, vector: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; matrix.nrows()];
    for (i, row) in matrix.rows().into_iter().enumerate() {
        out[i] = row.iter().zip(vector).map(|(a, b)| a * b).sum();
    }
    out
}

fn all_close(a: &[f64], b: &[f64]) -> bool {
    const ATOL: f64 = 1e-8;
    const RTOL: f64 = 1e-5;
    a.iter()
        .zip(b)
        .all(|(x, y)| (x - y).abs() <= ATOL + RTOL * y.abs())
}

/// Connected components via union-find (Python `connected_nodes` via scipy).
fn connected_nodes(matrix: &Array2<f64>) -> Vec<Vec<usize>> {
    let n = matrix.nrows();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], i: usize) -> usize {
        let mut root = i;
        while parent[root] != root {
            root = parent[root];
        }
        let mut current = i;
        while parent[current] != root {
            let next = parent[current];
            parent[current] = root;
            current = next;
        }
        root
    }

    for i in 0..n {
        for j in (i + 1)..n {
            if matrix[[i, j]] != 0.0 || matrix[[j, i]] != 0.0 {
                let ri = find(&mut parent, i);
                let rj = find(&mut parent, j);
                if ri != rj {
                    parent[rj] = ri;
                }
            }
        }
    }

    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        groups.entry(find(&mut parent, i)).or_default().push(i);
    }
    let mut result: Vec<Vec<usize>> = groups.into_values().collect();
    result.sort();
    result
}

/// Row-normalize the weights into a Markov matrix
/// (Python `create_markov_matrix`).
pub fn create_markov_matrix(weights_matrix: &Array2<f64>) -> Result<Array2<f64>, LexRankError> {
    let (n1, n2) = (weights_matrix.nrows(), weights_matrix.ncols());
    if n1 != n2 {
        return Err(LexRankError::NotSquare);
    }
    let min = weights_matrix.iter().copied().fold(f64::INFINITY, f64::min);
    if min <= 0.0 {
        // normalize the probability distribution row-wise with softmax
        let mut out = weights_matrix.clone();
        for mut row in out.rows_mut() {
            softmax_row(&mut row);
        }
        Ok(out)
    } else {
        let mut out = weights_matrix.clone();
        for mut row in out.rows_mut() {
            let sum: f64 = row.sum();
            if sum > 0.0 {
                row.mapv_inplace(|v| v / sum);
            }
        }
        Ok(out)
    }
}

fn softmax_row(row: &mut ndarray::ArrayViewMut1<'_, f64>) {
    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = row.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    for (i, e) in exps.iter().enumerate() {
        row[i] = e / sum;
    }
}

/// Discretize weights at `threshold` then row-normalize
/// (Python `create_markov_matrix_discrete`).
pub fn create_markov_matrix_discrete(
    weights_matrix: &Array2<f64>,
    threshold: f64,
) -> Result<Array2<f64>, LexRankError> {
    let discrete = weights_matrix.mapv(|v| if v >= threshold { 1.0 } else { 0.0 });
    create_markov_matrix(&discrete)
}

/// Stationary distribution of the transition matrix, computed per connected
/// component with power iteration (Python `stationary_distribution`).
pub fn stationary_distribution(
    transition_matrix: &Array2<f64>,
    increase_power: bool,
    normalized: bool,
) -> Vec<f64> {
    let (n1, n2) = (transition_matrix.nrows(), transition_matrix.ncols());
    if n1 != n2 {
        // Python raises ValueError; keep the parity by panicking on misuse.
        panic!("'transition_matrix' should be square");
    }
    let mut distribution = vec![0.0; n1];
    for group in connected_nodes(transition_matrix) {
        let indices: Vec<usize> = group.clone();
        let mut sub = Array2::<f64>::zeros((indices.len(), indices.len()));
        for (i, &gi) in indices.iter().enumerate() {
            for (j, &gj) in indices.iter().enumerate() {
                sub[[i, j]] = transition_matrix[[gi, gj]];
            }
        }
        let eigenvector = power_method(&sub, increase_power, 10_000);
        for (position, &index) in indices.iter().enumerate() {
            distribution[index] = eigenvector[position];
        }
    }
    if normalized {
        for value in distribution.iter_mut() {
            *value /= n1 as f64;
        }
    }
    distribution
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn invalid_threshold() {
        let m = array![[1.0]];
        assert!(degree_centrality_scores(&m, Some(1.0)).is_err());
        assert!(degree_centrality_scores(&m, Some(-0.5)).is_err());
    }

    #[test]
    fn non_square_rejected() {
        let m = array![[1.0, 2.0]];
        assert!(create_markov_matrix(&m).is_err());
    }

    #[test]
    fn uniform_graph_uniform_scores() {
        // fully connected: every sentence equally central
        let similarity = array![[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]];
        let scores = degree_centrality_scores(&similarity, Some(0.5)).unwrap();
        for i in 1..scores.len() {
            assert!((scores[0] - scores[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn markov_matrix_row_stochastic() {
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let markov = create_markov_matrix(&weights).unwrap();
        assert!((markov[[0, 0]] + markov[[0, 1]] - 1.0).abs() < 1e-9);
        assert!((markov[[1, 0]] + markov[[1, 1]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn disconnected_components_supported() {
        // two disjoint pairs: scores within each component are uniform
        let similarity = array![
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ];
        let scores = degree_centrality_scores(&similarity, Some(0.5)).unwrap();
        assert!((scores[0] - scores[1]).abs() < 1e-9);
        assert!((scores[2] - scores[3]).abs() < 1e-9);
    }

    #[test]
    fn negative_weights_use_softmax() {
        let weights = array![[-1.0, 2.0], [3.0, -4.0]];
        let markov = create_markov_matrix(&weights).unwrap();
        assert!((markov[[0, 0]] + markov[[0, 1]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn centrality_follows_degree() {
        // sentence 1 is similar to both others; 2 and 3 only to 1
        let similarity = array![[1.0, 0.9, 0.9], [0.9, 1.0, 0.1], [0.9, 0.1, 1.0]];
        let scores = degree_centrality_scores(&similarity, Some(0.5)).unwrap();
        assert!(scores[0] > scores[1]);
        assert!(scores[0] > scores[2]);
    }
}
