//! # Rapidly-exploring Random Tree (RRT) Example in 2 Dimensions
//!
//! ## Usage
//! Run the program with:
//! ```bash
//! cargo run --example rrt2d
//! ```

use macroquad::prelude::*;
use visualization_2d::Canvas2D;
mod visualization_2d;

use smart_3d::obstacles::{AnalyticValidityChecker, StaticSphericalObstacle};
use smart_3d::rrt::{
    termination::{MaxIterationsTermination, TerminationCondition},
    GoalBiasedUniformDistribution, KdTreeNearestNeighbors, RRTstar, RealVectorState,
};

const SCREEN_HEIGHT: i32 = 600;
const SCREEN_WIDTH: i32 = 600;

const MIN_X: i32 = 0;
const MAX_X: i32 = 30;
const MIN_Y: i32 = 0;
const MAX_Y: i32 = 30;

const OBSTACLE_COLOR: Color = BLACK;
const START_COLOR: Color = BLUE;
const GOAL_COLOR: Color = RED;
const BACKGROUND_COLOR: Color = WHITE;
const PATH_COLOR: Color = GREEN;
const PATH_THICKNESS: f32 = 2.0;
const TREE_COLOR: Color = BLACK;
const TREE_EDGE_THICKNESS: f32 = 1.0;
const TREE_NODE_RADIUS: f32 = 2.0;

fn window_conf() -> Conf {
    Conf {
        window_title: "RRT* in a 2D Environment".to_string(),
        window_width: SCREEN_HEIGHT,
        window_height: SCREEN_WIDTH,
        window_resizable: false,
        fullscreen: false,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    // Define the canvas for 2D visualization
    let canvas = Canvas2D::new(
        SCREEN_WIDTH as usize,
        SCREEN_HEIGHT as usize,
        MIN_X as f32,
        MIN_Y as f32,
        MAX_X as f32,
        MAX_Y as f32,
    );

    // Define the obstacles
    let circular_obstacles = vec![
        StaticSphericalObstacle::new(RealVectorState::new([20.0, 20.0]), 2.5),
        StaticSphericalObstacle::new(RealVectorState::new([20.0, 16.0]), 2.5),
        StaticSphericalObstacle::new(RealVectorState::new([10.0, 10.0]), 5.0),
        StaticSphericalObstacle::new(RealVectorState::new([15.0, 10.0]), 5.0),
        StaticSphericalObstacle::new(RealVectorState::new([20.0, 10.0]), 5.0),
        StaticSphericalObstacle::new(RealVectorState::new([10.0, 21.0]), 5.0),
    ];

    // We clone the spheres so that we have still have an unmoved copy that we can use for drawing.
    let validity_checker = AnalyticValidityChecker::new(circular_obstacles.clone());

    // Define the start and goal points.
    let start = RealVectorState::new([28.0, 28.0]);
    let goal = RealVectorState::new([2.0, 2.0]);
    let goal_tolerance = 0.5;

    // Use a uniform sampling distribution with 5% goal bias.
    let ranges = [(MIN_X as f32, MAX_X as f32), (MIN_Y as f32, MAX_Y as f32)];
    let goal_bias = 0.05;
    let result = GoalBiasedUniformDistribution::new(ranges, goal, goal_bias);
    if result.is_err() {
        println!(
            "Error creating sampling distribution: {}",
            result.err().unwrap()
        );
        return;
    }
    let sampling_distribution = result.unwrap();

    let steering_range = 1.0; // The maximum distance to steer towards the sample.
    let max_connection_radius = 2.0; // The maximum distance to connect during rewiring.
    let gamma = smart_3d::rrt::rrt_star::optimal_gamma(30.0 * 30.0, 2);

    // Create the RRT planner.
    let mut rrt_star = RRTstar::<f32, 2, KdTreeNearestNeighbors<_, 2>>::new(
        start,
        goal,
        goal_tolerance,
        Box::new(validity_checker),
        Box::new(sampling_distribution),
        steering_range,
        max_connection_radius,
        gamma,
    );

    let mut termination = MaxIterationsTermination::new(5000);

    loop {
        // Clear the screen
        clear_background(WHITE);

        // Draw the obstacles
        for circular_obs in &circular_obstacles {
            canvas.draw_circular_obstacle(circular_obs, OBSTACLE_COLOR);
        }

        // Draw the start and goal points.
        canvas.draw_circle(&start, 0.2, START_COLOR);
        canvas.draw_target(&goal, goal_tolerance, GOAL_COLOR, BACKGROUND_COLOR);

        if !termination.evaluate() {
            rrt_star.run_iterations(1);
        }

        // Draw each node and the edge to its parent.
        let nodes: &Vec<smart_3d::rrt::rrt_star::Node<f32, 2>> = rrt_star.get_tree();
        for node in nodes {
            let state = node.state();
            if let Some(parent_index) = node.parent() {
                let parent = &nodes[parent_index];
                let parent_state = parent.state();
                canvas.draw_edge(&state, &parent_state, TREE_EDGE_THICKNESS, TREE_COLOR);
            }
            canvas.draw_circle_radius_px(&state, TREE_NODE_RADIUS, TREE_COLOR);
        }

        // Draw the path if a solution was found.
        if let Some(path) = rrt_star.get_path() {
            canvas.draw_path(&path, PATH_THICKNESS, PATH_COLOR);
        }

        next_frame().await;
    }
}
