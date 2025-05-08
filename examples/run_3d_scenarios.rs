use bincode;
use clap::Parser;
use json;
use smart_3d::obstacles::StaticRectangularObstacle;
use smart_3d::rrt::RealVectorState;

mod scenarios;
use scenarios::scenario3d::Scenario3D;
use scenarios::ScenarioEndCondition;

use std::fs;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(version, about = "Run SMART 3D Scenarios", long_about = None)]
struct CliArgs {
    /// Directory to save the output files
    #[arg(short, long, default_value = "output/3d_scenarios")]
    output_dir: String,
}

fn main() {
    let args: CliArgs = CliArgs::parse();

    let default_scenario = Scenario3D {
        start_state: RealVectorState::new([2.0, 2.0, 2.0]),
        start_tolerance: 0.5,
        goal_state: RealVectorState::new([30.0, 30.0, 30.0]),
        goal_tolerance: 0.5,
        ranges: [(0.0, 32.0), (0.0, 32.0), (0.0, 32.0)],
        robot_speed: 4.0,
        n_obstacles: 15,
        obstacle_radius: 1.0,
        obstacle_speed: 4.0,
        obstacle_start_positions: None,
        obstacle_range: [(0.0, 32.0), (0.0, 32.0), (0.0, 32.0)],
        obstacle_restricted_regions: vec![
            StaticRectangularObstacle::new(
                RealVectorState::new([0.0, 0.0, 0.0]),
                RealVectorState::new([4.0, 4.0, 4.0]),
            ),
            StaticRectangularObstacle::new(
                RealVectorState::new([28.0, 28.0, 28.0]),
                RealVectorState::new([32.0, 32.0, 32.0]),
            ),
        ],
        ohz_time: 0.4,

        lrz_time: 1.0,
        lsr_initial_radius: 1.0,
        lsr_expansion_factor: 1.5,
        lsr_max_radius: 10.0,
        path_node_radius: 1.7,
        hot_node_neighborhood_radius: 1.7,
        replan_when_path_is_safe_but_robot_inside_ohz: false,
        ohz_shrinkage_margin: 0.01,

        rrt_steering_range: 1.0,
        rrt_max_connection_radius: 1.7,
        rrt_max_iterations: 20000,

        simulation_dt: 0.1,
        max_simulation_time: 300.0,
    };

    let num_obstacles_values = vec![25, 50, 75, 100];
    let obstacle_speeds = vec![1.0, 2.0, 3.0, 4.0];
    let trials_per_scenario = 100;
    let mut scenarios_json: json::JsonValue = json::array![];

    let output_dir: &Path = Path::new(&args.output_dir);
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).unwrap();
    }
    let runs_dir: &Path = &output_dir.join("runs");
    if !runs_dir.exists() {
        fs::create_dir_all(runs_dir).unwrap();
    }

    for &num_obstacles in &num_obstacles_values {
        for &obstacle_speed in &obstacle_speeds {
            let mut scenario = default_scenario.clone();
            scenario.n_obstacles = num_obstacles;
            scenario.obstacle_speed = obstacle_speed;

            let mut trials_json = json::array![];
            for trial_number in 0..trials_per_scenario {
                let mut scenario_run = scenario.run();
                while scenario_run.result.end_condition == ScenarioEndCondition::InitialPlanFailed {
                    println!(
                        "Scenario {:02} {:02} trial {:02} initial plan failed, retrying...",
                        num_obstacles, obstacle_speed, trial_number
                    );
                    scenario_run = scenario.run();
                }
                let scenario_run_bin_file_name = format!(
                    "scenario_run_{:02}_{:02}_{:02}.bin",
                    num_obstacles, obstacle_speed, trial_number
                );
                let scenario_run_bin_file = runs_dir.join(scenario_run_bin_file_name);

                let scenario_run_json_file_name = format!(
                    "scenario_run_{:02}_{:02}_{:02}.json",
                    num_obstacles, obstacle_speed, trial_number
                );
                let scenario_run_json_file = runs_dir.join(scenario_run_json_file_name);

                let replanning_times: Vec<f32> = scenario_run
                    .result
                    .replanning_times
                    .iter()
                    .map(|&time| time.as_secs_f32())
                    .collect();
                let trial_json = json::object! {
                    "scenario_run_bin_file" => scenario_run_bin_file.clone().into_os_string().into_string().unwrap(),
                    "scenario_run_json_file" => scenario_run_json_file.clone().into_os_string().into_string().unwrap(),
                    "success" => scenario_run.result.end_condition == ScenarioEndCondition::RobotReachedGoal,
                    "end_condition" => format!("{:?}", scenario_run.result.end_condition),
                    "replanning_times" => replanning_times.clone(),
                    "travel_distance" => scenario_run.result.travel_distance,
                    "travel_time" => scenario_run.result.travel_time,
                };
                trials_json
                    .push(trial_json)
                    .expect("Failed to push trial json.");

                // before we serialize the scenario run, set the nodes to None to save space
                scenario_run.recording.remove_nodes();
                let encoded: Vec<u8> = bincode::serialize(&scenario_run).unwrap();
                fs::write(scenario_run_bin_file, encoded).unwrap();

                let scenario_run_json = json::object! {
                    "recording" =>  scenario_run.recording.to_json(),
                    "success" => scenario_run.result.end_condition == ScenarioEndCondition::RobotReachedGoal,
                    "end_condition" => format!("{:?}", scenario_run.result.end_condition),
                    "replanning_times" => replanning_times,
                    "travel_distance" => scenario_run.result.travel_distance,
                    "travel_time" => scenario_run.result.travel_time,
                    "start_state" => json::array![scenario.start_state[0], scenario.start_state[1], scenario.start_state[2]],
                    "goal_state" => json::array![scenario.goal_state[0], scenario.goal_state[1], scenario.goal_state[2]],
                    "goal_tolerance" => scenario.goal_tolerance,
                    "bounds" => json::object! {
                        "min" => json::array![scenario.ranges[0].0, scenario.ranges[1].0, scenario.ranges[2].0],
                        "max" => json::array![scenario.ranges[0].1, scenario.ranges[1].1, scenario.ranges[2].1],
                    },
                };

                fs::write(scenario_run_json_file, scenario_run_json.to_string()).unwrap();

                println!(
                    "Scenario {:02} {:02} trial {:02} end: {:?}",
                    num_obstacles, obstacle_speed, trial_number, scenario_run.result.end_condition
                );

                if scenario_run.result.end_condition == ScenarioEndCondition::ReplanningError {
                    println!("Error in replanning. Aborting remaining trials.");
                    break;
                }
            }

            let scenario_json = json::object! {
                "num_obstacles" => num_obstacles,
                "obstacle_speed" => obstacle_speed,
                "trials" => trials_json
            };
            scenarios_json
                .push(scenario_json)
                .expect("Failed to push scenario json.");
        }
    }

    let output_json = json::object! { "scenarios" => scenarios_json };

    fs::write(
        output_dir.join("3d_scenarios.json"),
        output_json.to_string(),
    )
    .unwrap();
}
