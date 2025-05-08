use super::scenario_utils::{holonomic_move_along_path, make_spherical_obstacles_with_ohzs};
use super::{ScenarioEndCondition, ScenarioFrame, ScenarioRecording, ScenarioResult};
use serde::{Deserialize, Serialize};
use smart_3d::obstacles::random_dynamic::RandomWaypointSphere;
use smart_3d::obstacles::{
    DynamicSphericalObstacle, StaticRectangularObstacle, StaticSphericalObstacle,
};
use smart_3d::rrt::KdTreeNearestNeighbors;
use smart_3d::rrt::{
    rrt_star::optimal_gamma,
    termination::{MaxIterationsTermination, MaxTimeTermination},
    validity_checker::AlwaysValid,
    RRTstar, RealVectorState, SamplingDistribution, UniformDistribution,
};
use smart_3d::smart::{SmartUpdateResult, SMART};
use std::time::Duration;
use std::time::Instant;

/// A 3D scenario with a robot and obstacles.
///
/// Important Notes for Obstacle Trajectories:
/// If `obstacle_trajectories` are provided, `obstacle_start_positions` must be `None`.
/// If `obstacle_trajectories` is `None` and `obstacle_start_positions` is None,
///     then obstacles are randomly placed.
///
#[derive(Clone, Serialize, Deserialize)]
pub struct Scenario3D {
    // Planning problem parameters
    pub start_state: RealVectorState<f32, 3>,
    pub start_tolerance: f32,
    pub goal_state: RealVectorState<f32, 3>,
    pub goal_tolerance: f32,
    pub ranges: [(f32, f32); 3],
    pub robot_speed: f32,

    // Obstacle parameters
    pub n_obstacles: usize,
    pub obstacle_radius: f32,
    pub obstacle_speed: f32,
    pub obstacle_start_positions: Option<Vec<RealVectorState<f32, 3>>>,
    pub obstacle_range: [(f32, f32); 3],
    pub obstacle_restricted_regions: Vec<StaticRectangularObstacle<f32, 3>>,
    pub ohz_time: f32,

    // SMART parameters
    pub lrz_time: f32,
    pub path_node_radius: f32,
    pub hot_node_neighborhood_radius: f32,
    pub lsr_initial_radius: f32,
    pub lsr_expansion_factor: f32,
    pub lsr_max_radius: f32,
    pub replan_when_path_is_safe_but_robot_inside_ohz: bool,
    pub ohz_shrinkage_margin: f32,

    // Initial plan parameters
    pub rrt_steering_range: f32,
    pub rrt_max_connection_radius: f32,
    pub rrt_max_iterations: usize,

    // Simulation parameters
    pub simulation_dt: f32,
    pub max_simulation_time: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Scenario3DRun {
    pub scenario: Scenario3D,
    pub result: ScenarioResult,
    pub recording: ScenarioRecording<f32, 3>,
}

impl Scenario3D {
    pub fn run(&self) -> Scenario3DRun {
        let sampling_distribution = UniformDistribution::new(self.ranges);

        let lrz_radius = self.robot_speed * self.lrz_time;

        let static_validity_checker = Box::new(AlwaysValid::new());

        let gamma = optimal_gamma(
            (self.ranges[0].1 - self.ranges[0].0)
                * (self.ranges[1].1 - self.ranges[1].0)
                * (self.ranges[2].1 - self.ranges[2].0),
            3,
        );

        let mut rrt_star = RRTstar::new(
            self.goal_state.clone(),
            self.start_state.clone(),
            self.start_tolerance,
            static_validity_checker,
            Box::new(sampling_distribution),
            self.rrt_steering_range,
            self.rrt_max_connection_radius,
            gamma,
        );

        let mut termination = MaxIterationsTermination::new(self.rrt_max_iterations);
        rrt_star.plan_until(&mut termination);

        let (nodes, nearest_neighbors, static_validity_checker, sampling_distribution) =
            rrt_star.deconstruct_into_components();

        let mut smart = SMART::new_from_initial_rrt_star_tree(
            &nodes,
            nearest_neighbors,
            self.start_state.clone(),
            self.goal_state.clone(),
            self.goal_tolerance,
            static_validity_checker,
            sampling_distribution,
            lrz_radius,
            self.lsr_initial_radius,
            self.lsr_expansion_factor,
            self.lsr_max_radius,
            self.hot_node_neighborhood_radius,
            self.path_node_radius,
            self.replan_when_path_is_safe_but_robot_inside_ohz,
            self.ohz_shrinkage_margin,
        );

        smart.initial_plan_add_node_at_start();
        smart.initial_plan_build_path();

        self.run_simulation(smart)
    }

    fn run_simulation(
        &self,
        mut smart: SMART<f32, 3, KdTreeNearestNeighbors<f32, 3>>,
    ) -> Scenario3DRun {
        let mut obstacles = self.generate_obstacles();

        let mut scenario_result = ScenarioResult::new();
        let mut scenario_recording = ScenarioRecording::new(vec![], self.simulation_dt);

        if smart.get_path().is_none() {
            scenario_result.end_condition = ScenarioEndCondition::InitialPlanFailed;
            return Scenario3DRun {
                scenario: self.clone(),
                result: scenario_result,
                recording: scenario_recording,
            };
        }

        let initial_path: Vec<RealVectorState<f32, 3>> = smart.get_path().unwrap();

        let dt = self.simulation_dt;
        let mut robot = self.start_state.clone();
        let mut current_path = initial_path.clone();
        let mut current_path_index = 1;

        // Record the initial frame
        let scenario_frame = ScenarioFrame {
            robot: robot.clone(),
            obstacles: make_spherical_obstacles_with_ohzs(
                &obstacles,
                self.obstacle_speed,
                self.ohz_time,
            ),
            nodes: Some(smart.get_nodes().clone()),
            path: Some(current_path.clone()),
            replanning_trigger: None,
            planning_result: None,
        };
        scenario_recording.add_frame(scenario_frame);

        loop {
            // Update robot position
            let travel_distance = self.robot_speed * dt;
            let (new_robot, new_index) = holonomic_move_along_path(
                &robot,
                &current_path,
                current_path_index,
                travel_distance,
            );
            robot = new_robot;
            current_path_index = new_index;
            scenario_result.travel_distance += travel_distance;
            scenario_result.travel_time += dt;

            // Update obstacle positions
            for obstacle in &mut obstacles {
                obstacle.update(dt);
            }

            let update_start = Instant::now();
            let termination_time = {
                if cfg!(debug_assertions) {
                    dt * 20.0
                } else {
                    dt
                }
            };

            let dynamic_obstacles_with_ohz =
                make_spherical_obstacles_with_ohzs(&obstacles, self.obstacle_speed, self.ohz_time);

            let mut termination =
                MaxTimeTermination::new(Duration::from_secs_f32(termination_time));

            let result = smart.update(
                robot.clone(),
                current_path_index,
                &dynamic_obstacles_with_ohz,
                Some(&mut termination),
            );
            let replanning_time = update_start.elapsed();

            match result.clone() {
                SmartUpdateResult::NoReplanningNeeded => {
                    scenario_result.updates_without_replanning += 1;
                }
                SmartUpdateResult::ReplanningSuccessful => {
                    scenario_result.replanning_times.push(replanning_time);
                    current_path = smart.get_path().unwrap();
                    current_path_index = 0;
                }
                SmartUpdateResult::ReplanningFailed => {
                    scenario_result.failed_replanning_attempts += 1;
                    scenario_result.end_condition = ScenarioEndCondition::ReplanningFailed;
                }
                SmartUpdateResult::ReplanningError(msg) => {
                    scenario_result.failed_replanning_attempts += 1;
                    scenario_result.end_condition = ScenarioEndCondition::ReplanningError;
                    current_path = smart.get_path().unwrap();
                    println!("Error: {}", msg);
                    println!("\x1b[31mREPLANNING ERROR! 1\x1b[0m");
                    println!("\x1b[31mREPLANNING ERROR! 2\x1b[0m");
                    println!("\x1b[31mREPLANNING ERROR! 3\x1b[0m");
                }
                SmartUpdateResult::InCollision => {
                    scenario_result.end_condition = ScenarioEndCondition::RobotCollidedWithObstacle;
                }
            }

            // record the latest frame
            let scenario_frame = ScenarioFrame {
                robot: robot.clone(),
                obstacles: dynamic_obstacles_with_ohz,
                nodes: Some(smart.get_nodes().clone()),
                path: Some(current_path.clone()),
                replanning_trigger: smart.replanning_triggered_by().clone(),
                planning_result: Some(result.clone()),
            };
            scenario_recording.add_frame(scenario_frame);

            if result.is_failure() {
                break;
            }

            if robot.euclidean_distance(&self.goal_state) <= self.goal_tolerance {
                scenario_result.end_condition = ScenarioEndCondition::RobotReachedGoal;
                break;
            }

            if scenario_result.travel_time >= self.max_simulation_time {
                scenario_result.end_condition = ScenarioEndCondition::OutOfTime;
                break;
            }
        }

        Scenario3DRun {
            scenario: self.clone(),
            result: scenario_result,
            recording: scenario_recording,
        }
    }

    pub fn generate_obstacles(&self) -> Vec<Box<dyn DynamicSphericalObstacle<f32, 3>>> {
        let obstacles: Vec<Box<dyn DynamicSphericalObstacle<f32, 3>>>;

        let obstacle_starts: Vec<RealVectorState<f32, 3>>;
        if let Some(starts) = &self.obstacle_start_positions {
            assert!(
                starts.len() == self.n_obstacles,
                "Number of obstacle start positions must match number of obstacles"
            );
            obstacle_starts = starts.clone();
        } else {
            let mut obstacle_sampling_distribution = UniformDistribution::new(self.obstacle_range);
            obstacle_starts = (0..self.n_obstacles)
                .map(|_| obstacle_sampling_distribution.sample())
                .collect();
        }

        obstacles = obstacle_starts
            .iter()
            .map(|start| {
                let circle = StaticSphericalObstacle::new(start.clone(), self.obstacle_radius);
                let obstacle = RandomWaypointSphere::new(
                    circle,
                    self.obstacle_speed,
                    Box::new(UniformDistribution::new(self.obstacle_range)),
                    self.obstacle_restricted_regions.clone(),
                );
                Box::new(obstacle) as Box<dyn DynamicSphericalObstacle<f32, 3>>
            })
            .collect();

        obstacles
    }
}
