use num_traits::Float;
use smart_3d::obstacles::{DynamicSphericalObstacle, StaticSphericalObstacle};
use smart_3d::rrt::RealVectorState;
use smart_3d::smart::SphericalObstacleWithOhz;

pub fn make_spherical_obstacles_with_ohzs<F: Float, const N: usize>(
    spherical_obstacles: &Vec<Box<dyn DynamicSphericalObstacle<F, N>>>,
    obstacle_speed: F,
    ohz_time: F,
) -> Vec<SphericalObstacleWithOhz<F, N>> {
    let mut spherical_obstacles_with_ohz = Vec::new();

    // Iterate over the dynamic obstacles and create static obstacles with OHZ
    for dynamic_obstacle in spherical_obstacles {
        let obstacle = StaticSphericalObstacle::new(
            dynamic_obstacle.center().clone(),
            dynamic_obstacle.radius(),
        );

        let obstacle_with_ohz = StaticSphericalObstacle::new(
            dynamic_obstacle.center().clone(),
            dynamic_obstacle.radius() + ohz_time * obstacle_speed,
        );

        spherical_obstacles_with_ohz
            .push(SphericalObstacleWithOhz::new(obstacle, obstacle_with_ohz));
    }

    spherical_obstacles_with_ohz
}

pub fn holonomic_move_along_path<F: Float, const N: usize>(
    robot: &RealVectorState<F, N>,
    path: &Vec<RealVectorState<F, N>>,
    current_index: usize,
    distance: F,
) -> (RealVectorState<F, N>, usize) {
    let mut remaining_distance = distance;
    let mut current_index = current_index;
    let mut current_position = robot.clone();

    while current_index < path.len() {
        let current_segment = &path[current_index] - &current_position;
        let current_segment_length = current_segment.norm();

        if current_segment_length < remaining_distance {
            current_position = path[current_index].clone();
            remaining_distance = remaining_distance - current_segment_length;
            current_index += 1;
        } else {
            let direction = current_segment / current_segment_length;
            current_position = &current_position + &(direction * remaining_distance);
            break;
        }
    }

    (current_position, current_index)
}
