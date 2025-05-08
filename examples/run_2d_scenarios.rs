use bincode;
use clap::Parser;
use json::object;
use smart_3d::rrt::RealVectorState;
use std::fs;
use std::path::Path;

mod scenarios;
use scenarios::scenario2d::Scenario2D;
use scenarios::ScenarioEndCondition;

#[derive(Parser, Debug)]
#[command(version, about = "Run SMART 2D Scenarios", long_about = None)]
struct CliArgs {
    /// Directory to save the output files
    #[arg(short, long, default_value = "output/2d_scenarios")]
    output_dir: String,

    /// Whether to save the tree in the scenario run bin file
    /// Saving tree is useful for visualiztion/debugging, but it increases the file size significantly.
    #[arg(short, long)]
    save_tree: bool,
}

fn gen_obstacle_starts(n_obstacles: usize) -> Vec<RealVectorState<f32, 2>> {
    let y_positions = vec![6.0, 11.0, 16.0, 21.0, 26.0];

    if n_obstacles % y_positions.len() != 0 {
        panic!(
            "This function only supports numbers of obstacles that are multiples of {}",
            y_positions.len()
        );
    }

    let x_positions = match n_obstacles {
        5 => vec![16.0],
        10 => vec![10.0, 22.0],
        15 => vec![6.0, 16.0, 26.0],
        20 => vec![6.0, 12.66, 19.33, 26.0],
        _ => panic!("This function only supports 5, 10, 15, or 20 obstacles."),
    };

    let mut starts = Vec::new();
    for &y in &y_positions {
        for &x in &x_positions {
            starts.push(RealVectorState::new([x, y]));
        }
    }

    assert!(starts.len() == n_obstacles);

    starts
}

fn main() {
    let args: CliArgs = CliArgs::parse();

    let static_rectangles = vec![];

    let default_scenario = Scenario2D {
        start_state: RealVectorState::new([2.0, 2.0]),
        start_tolerance: 1.0,
        goal_state: RealVectorState::new([30.01, 30.01]),
        goal_tolerance: 1.0,
        ranges: [(0.0, 32.0), (0.0, 32.0)],
        robot_speed: 4.0,
        static_rectangles,
        n_obstacles: 15,
        obstacle_radius: 1.0,
        obstacle_speed: 4.0,
        obstacle_start_positions: None,
        obstacle_trajectory_max_distance: 10.0,
        use_legacy_smart2d_obstacle: true,
        obstacle_range: [(0.0, 32.0), (0.0, 32.0)],
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
        rrt_max_iterations: 2500,

        simulation_dt: 0.1,
        max_simulation_time: 300.0,
    };

    let num_obstacles_values = vec![5, 10, 15, 20];
    let obstacle_speeds = vec![1.0, 2.0, 3.0, 4.0];
    let trials_per_scenario = 100;
    let mut scenarios_json = json::array![];

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
            scenario.obstacle_start_positions = Some(gen_obstacle_starts(num_obstacles));
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

                let replanning_times: Vec<f32> = scenario_run
                    .result
                    .replanning_times
                    .iter()
                    .map(|&time| time.as_secs_f32())
                    .collect();
                let trial_json = object! {
                    "scenario_run_bin_file" => scenario_run_bin_file.clone().into_os_string().into_string().unwrap(),
                    "success" => scenario_run.result.end_condition == ScenarioEndCondition::RobotReachedGoal,
                    "replanning_times" => replanning_times,
                    "travel_distance" => scenario_run.result.travel_distance,
                    "travel_time" => scenario_run.result.travel_time,
                };
                trials_json
                    .push(trial_json)
                    .expect("Failed to push trial json.");

                if !args.save_tree {
                    // Remove the tree from the scenario run to save space
                    scenario_run.recording.remove_nodes();
                }
                let encoded: Vec<u8> = bincode::serialize(&scenario_run).unwrap();
                fs::write(scenario_run_bin_file, encoded).unwrap();

                println!(
                    "Scenario {:02} {:02} trial {:02} end: {:?}",
                    num_obstacles, obstacle_speed, trial_number, scenario_run.result.end_condition
                );

                if scenario_run.result.end_condition == ScenarioEndCondition::ReplanningError {
                    println!("Error in replanning. Aborting remaining trials.");
                    break;
                }
            }

            let scenario_json = object! {
                "num_obstacles" => num_obstacles,
                "obstacle_speed" => obstacle_speed,
                "trials" => trials_json
            };
            scenarios_json
                .push(scenario_json)
                .expect("Failed to push scenario json.");
        }
    }

    let output_json = object! { "scenarios" => scenarios_json };

    fs::write(
        output_dir.join("2d_scenarios.json"),
        output_json.to_string(),
    )
    .unwrap();
}
