use crate::obstacles::StaticSphericalObstacle;
use num_traits::Float;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SphericalObstacleWithOhz<F: Float, const N: usize> {
    obstacle: StaticSphericalObstacle<F, N>,
    ohz: StaticSphericalObstacle<F, N>,
}

impl<F: Float, const N: usize> SphericalObstacleWithOhz<F, N> {
    pub fn new(
        obstacle: StaticSphericalObstacle<F, N>,
        ohz: StaticSphericalObstacle<F, N>,
    ) -> Self {
        Self { obstacle, ohz }
    }

    pub fn obstacle(&self) -> &StaticSphericalObstacle<F, N> {
        &self.obstacle
    }

    pub fn ohz(&self) -> &StaticSphericalObstacle<F, N> {
        &self.ohz
    }
}
