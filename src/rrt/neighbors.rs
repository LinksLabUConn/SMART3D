use crate::rrt::state::RealVectorState;
use kiddo::float::{distance::SquaredEuclidean, kdtree::Axis, kdtree::KdTree};
use num_traits::Float;

/// A trait for a nearest neighbor data structure that supports nearest neighbors and radius queries.
/// Stores RealVectorStates and a usize index along with them.
pub trait NearestNeighbors<F: Float, const N: usize> {
    /// Constructs a new nearest neighbor data structure.
    /// The data structure is empty initially.
    fn new() -> Self;

    /// Adds a state to the data structure.
    ///
    /// Parameters:
    /// - `state`: The RealVectorState to add.
    /// - `item`: The index of the RealVectorState.
    fn add(&mut self, state: RealVectorState<F, N>, item: usize);

    /// Gets the nearest neighbor to the given RealVectorState.
    ///
    /// Parameters:
    /// - `state`: The RealVectorState to find the nearest neighbor to.
    ///
    /// Returns:
    /// The item/index of the nearest neighbor, if any.
    fn nearest_one(&self, state: &RealVectorState<F, N>) -> Option<usize> {
        let nearest_vec = self.nearest_k(state, 1);
        if nearest_vec.is_empty() {
            None
        } else {
            Some(nearest_vec[0])
        }
    }

    /// Gets the k nearest neighbors to the given RealVectorState.
    ///
    /// Parameters:
    /// - `state`: The RealVectorState to find the nearest neighbors to.
    /// - `k`: The number of neighbors to find.
    ///
    /// Returns:
    /// The items/indices of the k nearest neighbors.
    fn nearest_k(&self, state: &RealVectorState<F, N>, k: usize) -> Vec<usize>;

    /// Gets all RealVectorStates within a given radius of the given RealVectorState.
    ///
    /// Parameters:
    /// - `state`: The RealVectorState to find the neighbors of.
    /// - `radius`: The radius within which to find neighbors.
    ///
    /// Returns:
    /// The items/indices of the RealVectorStates within the radius.
    fn within_radius(&self, state: &RealVectorState<F, N>, radius: F) -> Vec<usize>;

    /// Gets all RealVectorStates within a given radius of the given RealVectorState in sorted order, nearest-first.
    ///
    /// Parameters:
    /// - `state`: The RealVectorState to find the neighbors of.
    /// - `radius`: The radius within which to find neighbors.
    ///
    /// Returns:
    /// The items/indices of the RealVectorStates within the radius.
    fn within_radius_sorted(&self, state: &RealVectorState<F, N>, radius: F) -> Vec<usize>;
}

/// A nearest neighbor data structure that uses a linear search to find the nearest neighbors.
/// This is useful for small datasets.
pub struct LinearNearestNeighbors<F: Float, const N: usize> {
    states: Vec<(RealVectorState<F, N>, usize)>,
}

impl<F: Float, const N: usize> NearestNeighbors<F, N> for LinearNearestNeighbors<F, N> {
    fn new() -> Self {
        Self { states: Vec::new() }
    }

    fn add(&mut self, state: RealVectorState<F, N>, item: usize) {
        self.states.push((state, item));
    }

    fn nearest_one(&self, state: &RealVectorState<F, N>) -> Option<usize> {
        let nearest = self.states.iter().min_by(|a, b| {
            state
                .euclidean_distance_squared(&a.0)
                .partial_cmp(&state.euclidean_distance_squared(&b.0))
                .unwrap()
        });
        nearest.map(|(_, i)| *i)
    }

    fn nearest_k(&self, state: &RealVectorState<F, N>, k: usize) -> Vec<usize> {
        let mut nearest = self
            .states
            .iter()
            .map(|(p, i)| (state.euclidean_distance_squared(&p), *i))
            .collect::<Vec<_>>();
        nearest.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        nearest.into_iter().take(k).map(|(_, i)| i).collect()
    }

    fn within_radius(&self, state: &RealVectorState<F, N>, radius: F) -> Vec<usize> {
        self.states
            .iter()
            .filter(|(p, _)| state.euclidean_distance_squared(&p) <= radius * radius)
            .map(|(_, i)| *i)
            .collect()
    }

    fn within_radius_sorted(&self, state: &RealVectorState<F, N>, radius: F) -> Vec<usize> {
        let mut within = self.within_radius(state, radius);
        within.sort_by(|&a, &b| {
            state
                .euclidean_distance_squared(&self.states[a].0)
                .partial_cmp(&state.euclidean_distance_squared(&self.states[b].0))
                .unwrap()
        });
        within
    }
}

pub struct KdTreeNearestNeighbors<F: Float + Axis, const N: usize> {
    kdtree: KdTree<F, usize, N, 32, u32>,
}

impl<F: Float + Axis, const N: usize> NearestNeighbors<F, N> for KdTreeNearestNeighbors<F, N> {
    fn new() -> Self {
        Self {
            kdtree: KdTree::new(),
        }
    }

    fn add(&mut self, state: RealVectorState<F, N>, item: usize) {
        self.kdtree.add(state.values(), item);
    }

    fn nearest_one(&self, state: &RealVectorState<F, N>) -> Option<usize> {
        let neighbor = self.kdtree.nearest_one::<SquaredEuclidean>(state.values());
        Some(neighbor.item)
    }

    fn nearest_k(&self, state: &RealVectorState<F, N>, k: usize) -> Vec<usize> {
        self.kdtree
            .nearest_n::<SquaredEuclidean>(state.values(), k)
            .iter()
            .map(|n| n.item)
            .collect()
    }

    fn within_radius(&self, state: &RealVectorState<F, N>, radius: F) -> Vec<usize> {
        self.kdtree
            .within_unsorted::<SquaredEuclidean>(state.values(), radius * radius)
            .iter()
            .map(|n| n.item)
            .collect()
    }

    fn within_radius_sorted(&self, state: &RealVectorState<F, N>, radius: F) -> Vec<usize> {
        self.kdtree
            .within::<SquaredEuclidean>(state.values(), radius * radius)
            .iter()
            .map(|n| n.item)
            .collect()
    }
}
