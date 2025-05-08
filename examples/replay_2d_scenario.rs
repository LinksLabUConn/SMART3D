use bincode;
use clap::Parser;
use macroquad::prelude::*;
use smart_3d::obstacles::SphericalObstacle;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use image::{ImageBuffer, Rgba};
use tempfile::TempDir;

mod scenarios;
use scenarios::scenario2d::{Scenario2D, Scenario2DRun};
use scenarios::ScenarioFrame;

use visualization_2d::Canvas2D;
mod visualization_2d;

const METERS_TO_PIXELS: f32 = 20.0;

/// The window configuration is determined by reading the scenario file
/// early (using std::env::args) to compute its dimensions.
/// It looks for "--filepath" or "-f" and then reads the file, deserializes it,
/// and multiplies the scenario’s dimensions by METERS_TO_PIXELS.
fn window_conf() -> Conf {
    // Collect the command-line arguments.
    let args: Vec<String> = std::env::args().collect();
    // Look for either "--filepath" or "-f" in the arguments.
    let mut filepath: Option<String> = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--filepath" || arg == "-f" {
            filepath = iter.next().cloned();
            break;
        }
    }
    if let Some(fp) = filepath {
        match std::fs::read(&fp) {
            Ok(bytes) => match bincode::deserialize::<Scenario2DRun>(&bytes) {
                Ok(scenario_run) => {
                    let window_width = scenario_run.scenario.ranges[0].1 as f32 * METERS_TO_PIXELS;
                    let window_height = scenario_run.scenario.ranges[1].1 as f32 * METERS_TO_PIXELS;
                    println!(
                        "Window dimensions determined from scenario: {} x {}",
                        window_width, window_height
                    );
                    Conf {
                        window_title: "SMART 2D Scenario Replay".to_string(),
                        window_width: window_width as i32,
                        window_height: window_height as i32,
                        window_resizable: false,
                        fullscreen: false,
                        ..Default::default()
                    }
                }
                Err(e) => {
                    eprintln!("Failed to deserialize scenario file '{}': {:?}", fp, e);
                    std::process::exit(1);
                }
            },
            Err(e) => {
                eprintln!("Failed to read scenario file '{}': {:?}", fp, e);
                std::process::exit(1);
            }
        }
    } else {
        // If no scenario file is provided, fall back to a default size.
        println!("No scenario file provided in window_conf(). Using default window size 800x600.");
        Conf {
            window_title: "SMART 2D Scenario Replay".to_string(),
            window_width: 800,
            window_height: 600,
            window_resizable: false,
            fullscreen: false,
            ..Default::default()
        }
    }
}

/// Command-line arguments for the scenario player.
/// (These are parsed later in main using Clap.)
#[derive(Parser, Debug)]
#[command(version, about = "SMART 2D Scenario Replay", long_about = None)]
struct Cli {
    /// File path to the scenario file (e.g., "output/scenario_run.bin")
    #[arg(short, long)]
    filepath: String,

    /// Do not display the tree in the visualization.
    #[arg(short, long)]
    hide_tree: bool,

    /// Do not display the LRZ in the visualization.
    #[arg(short, long)]
    hide_lrz: bool,

    /// Playback speed multiplier (default: 1.0)
    #[arg(short, long, default_value_t = 1.0)]
    speed: f32,

    /// Optional output video file path. When provided (e.g. `--output-video output.mp4`),
    /// the scenario will be recorded as a video.
    #[arg(long)]
    output_video: Option<String>,
}

const ROBOT_COLOR: Color = BLUE;
const BACKGROUND_COLOR: Color = WHITE;
const GOAL_COLOR: Color = RED;
const LRZ_COLOR: Color = GREEN;
const OBSTACLE_COLOR: Color = GRAY;
const STATIC_OBSTACLE_COLOR: Color = BLACK;
const OHZ_COLOR: Color = RED;
const TREE_COLOR: Color = LIGHTGRAY;
const TREE_NODE_RADIUS: f32 = 2.0;
const TREE_EDGE_THICKNESS: f32 = 1.0;
const PATH_COLOR: Color = GREEN;
const PATH_COLOR_AFTER_REPLAN: Color = BLUE;
const PATH_THICKNESS: f32 = 2.0;

/// Draws one scenario frame.
fn draw_frame(
    canvas: &Canvas2D<f32>,
    frame: &ScenarioFrame<f32, 2>,
    scenario: &Scenario2D,
    show_tree: bool,
    show_lrz: bool,
) {
    clear_background(BACKGROUND_COLOR);

    // Draw static rectangle (in black)
    let static_rectangles = &scenario.static_rectangles;
    for rectangle in static_rectangles.iter() {
        canvas.draw_rectangular_obstacle(rectangle, STATIC_OBSTACLE_COLOR);
    }

    // Draw robot.
    canvas.draw_circle_radius_px(&frame.robot, 5.0, ROBOT_COLOR);

    // Draw LRZ.
    if show_lrz {
        let lrz_radius = scenario.robot_speed * scenario.lrz_time;
        canvas.draw_circle_outline(&frame.robot, lrz_radius, 3.0, LRZ_COLOR);
    }

    // Draw goal.
    canvas.draw_target_radius_px(&scenario.goal_state, 15.0, GOAL_COLOR, BACKGROUND_COLOR);

    // Draw the tree.
    if let Some(nodes) = &frame.nodes {
        if show_tree {
            for node in nodes {
                let state = node.state();
                if let Some(parent_index) = node.parent() {
                    let parent = &nodes[parent_index];
                    let parent_state = parent.state();
                    canvas.draw_edge(&state, &parent_state, TREE_EDGE_THICKNESS, TREE_COLOR);
                }
                canvas.draw_circle_radius_px(&state, TREE_NODE_RADIUS, TREE_COLOR);
            }
        }
    }

    // Draw the solution path if available.
    if let Some(path) = &frame.path {
        let path_color = if frame
            .planning_result
            .as_ref()
            .map_or(false, |pr| pr.replanned())
        {
            PATH_COLOR_AFTER_REPLAN
        } else {
            PATH_COLOR
        };

        canvas.draw_path(path, PATH_THICKNESS, path_color);
    }

    // Draw obstacles.
    for obstacle in frame.obstacles.iter() {
        canvas.draw_circular_obstacle(obstacle.obstacle(), OBSTACLE_COLOR);
        canvas.draw_circle_outline(
            obstacle.ohz().center(),
            obstacle.ohz().radius(),
            2.0,
            OHZ_COLOR,
        );
    }
}

/// Captures the current screen image and saves it as a PNG file in `save_path`
/// with a filename based on the frame number.
async fn save_frame(save_path: &Path, frame_number: usize) {
    // Capture the rendered screen.
    let screen_image = get_screen_data();
    let width = screen_image.width as u32;
    let height = screen_image.height as u32;

    // Convert Macroquad's image into an ImageBuffer.
    let buffer = ImageBuffer::<Rgba<u8>, _>::from_raw(
        width,
        height,
        screen_image.bytes.to_vec(), // Raw pixel data.
    )
    .expect("Failed to create image buffer");

    // Build the output filename.
    let filename = format!("{}/frame_{:03}.png", save_path.display(), frame_number);
    buffer.save(&filename).expect("Failed to save image");
}

/// Plays the scenario. If `output_dir` is provided, each frame is saved as an image
/// in that directory.
async fn play_scenario(
    scenario_run: &Scenario2DRun,
    show_tree: bool,
    show_lrz: bool,
    playback_speed: f32,
    output_dir: Option<&Path>,
) {
    // Resize the window based on the scenario dimensions.
    let window_width = scenario_run.scenario.ranges[0].1 as f32 * METERS_TO_PIXELS;
    let window_height = scenario_run.scenario.ranges[1].1 as f32 * METERS_TO_PIXELS;
    request_new_screen_size(window_width, window_height);

    let canvas = Canvas2D::new(
        window_width as usize,
        window_height as usize,
        scenario_run.scenario.ranges[0].0,
        scenario_run.scenario.ranges[1].0,
        scenario_run.scenario.ranges[0].1,
        scenario_run.scenario.ranges[1].1,
    );

    let dt = scenario_run.scenario.simulation_dt;
    let mut frame_counter = 0;

    for frame in scenario_run.recording.frames().iter() {
        let frame_start = Instant::now();

        draw_frame(&canvas, frame, &scenario_run.scenario, show_tree, show_lrz);

        // Maintain playback timing.
        let target_duration = Duration::from_secs_f32(dt / playback_speed);
        let elapsed = frame_start.elapsed();
        if elapsed < target_duration {
            std::thread::sleep(target_duration - elapsed);
        }

        // Save the frame if video recording is enabled.
        if let Some(dir) = output_dir {
            save_frame(dir, frame_counter).await;
            frame_counter += 1;
        }

        next_frame().await;
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    // Parse the command-line arguments with Clap.
    let cli = Cli::parse();

    // Verify that the scenario file exists.
    let path = Path::new(&cli.filepath);
    if !path.exists() {
        eprintln!("File not found: {}", cli.filepath);
        std::process::exit(1);
    }

    // Read and deserialize the scenario file.
    let encoded = std::fs::read(path).expect("Failed to read the scenario file");
    let scenario_run: Scenario2DRun =
        bincode::deserialize(&encoded).expect("Failed to deserialize scenario");

    println!("{}", scenario_run.result);

    // If the user requested a video output, create a temporary directory to store frames.
    let temp_dir_opt: Option<TempDir> = cli
        .output_video
        .as_ref()
        .map(|_| TempDir::new().expect("Failed to create temporary directory for video frames"));
    let output_dir: Option<&Path> = temp_dir_opt.as_ref().map(|temp_dir| temp_dir.path());

    // Play the scenario. If video recording is enabled, frames are saved to `output_dir`.
    play_scenario(
        &scenario_run,
        !cli.hide_tree,
        !cli.hide_lrz,
        cli.speed,
        output_dir,
    )
    .await;

    // If an output video file was specified, use ffmpeg to stitch the frames together.
    if let Some(video_path) = cli.output_video {
        // Compute the frame rate. Since the simulation runs at dt seconds per frame,
        // the effective FPS is (playback_speed / dt).
        let fps = cli.speed / scenario_run.scenario.simulation_dt;
        // Build the input pattern. (The frames were saved as frame_000.png, frame_001.png, etc.)
        let temp_dir = temp_dir_opt.expect("Temporary directory missing despite video request");
        let input_pattern = format!("{}/frame_%03d.png", temp_dir.path().display());
        println!("Running ffmpeg on frames in: {}", temp_dir.path().display());

        let status = Command::new("ffmpeg")
            .args(&[
                "-y", // Overwrite output file if it exists.
                "-framerate",
                &format!("{}", fps),
                "-i",
                &input_pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                &video_path,
            ])
            .status()
            .expect("Failed to execute ffmpeg command. Is it installed?");

        if !status.success() {
            eprintln!("ffmpeg failed with status: {:?}", status);
        } else {
            println!("Video created successfully: {}", video_path);
        }
    }
}
