# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyvista",
#   "numpy",
#   "imageio",
#   "imageio[ffmpeg]"
# ]
# ///

import argparse
import json
import time
import numpy as np
import pyvista as pv
from pathlib import Path 
import math

BACKGROUND_COLOR = "white"
GOAL_COLOR = "red"
ROBOT_COLOR = "blue"
OBSTACLE_COLOR = "gray"
OBSTACLE_OPACITY = 0.5
PATH_COLOR = "green"
REPLANNED_PATH_UNSAFE_COLOR = "blue"
REPLANNED_PATH_ROBOT_OHZ_COLOR = "blue"
BOUNDS_COLOR = "gray"

def generate_sphere_mesh(center, radius, resolution=20):
    """
    Create a sphere mesh using PyVista.
    """
    sphere = pv.Sphere(radius=radius, center=center,
                       theta_resolution=resolution,
                       phi_resolution=resolution)
    return sphere

def get_camera_position(view, center, diag):
    """
    Return a camera configuration (position, focal point, viewup) based on the view name.
    The distance from the center is kept the same as the default view.
    For horizontal views the elevation is the same (~26.565°), while top/bottom
    place the camera directly above or below the center.
    """
    # Compute the default offset used in the original script.
    # Default: position = center + (0, -1.5*diag, 0.75*diag)
    default_offset = np.array([0, -1.5, 0.75]) * diag
    # Its magnitude (r) is:
    r = np.linalg.norm(default_offset)  # approximately 1.6771 * diag
    # The default elevation angle is:
    default_elev = np.arcsin(0.75 / 1.6771)  # about 26.565° in radians

    view = view.lower()
    if view in ["default", "back"]:
        azimuth = np.deg2rad(270)  # same as default (-90°)
        elev = default_elev
    elif view == "front":
        azimuth = np.deg2rad(90)
        elev = default_elev
    elif view == "left":
        azimuth = np.deg2rad(180)
        elev = default_elev
    elif view == "right":
        azimuth = np.deg2rad(0)
        elev = default_elev
    elif view in ["top", "topdown"]:
        # For top view, use an elevation of 90°.
        elev = np.deg2rad(90)
        azimuth = 0  # azimuth is irrelevant when looking straight down
    elif view in ["bottom", "bottomup"]:
        elev = np.deg2rad(-90)
        azimuth = 0
    else:
        raise ValueError(f"Unknown view: {view}")
    
    # For top/bottom, horizontal components are zero.
    if abs(elev - np.deg2rad(90)) < 1e-6 or abs(elev + np.deg2rad(90)) < 1e-6:
        offset = np.array([0, 0, r * np.sign(np.sin(elev))])
    else:
        # Compute offset using spherical coordinates (r, azimuth, elevation).
        offset_x = r * np.cos(elev) * np.cos(azimuth)
        offset_y = r * np.cos(elev) * np.sin(azimuth)
        offset_z = r * np.sin(elev)
        offset = np.array([offset_x, offset_y, offset_z])
    # Camera position is the center plus the offset.
    position = [center[i] + offset[i] for i in range(3)]
    # Use z-up for most views; for top/bottom this might be arbitrary.
    
    # Use a different viewup for top/bottom views to avoid degeneracy.
    if view in ["top", "topdown", "bottom", "bottomup"]:
        viewup = (0, 1, 0)
    else:
        viewup = (0, 0, 1)
        
    return (tuple(position), tuple(center), viewup)   

def main():
    parser = argparse.ArgumentParser(
        description="Animate scenario JSON with PyVista (real-time interactive view)."
    )
    parser.add_argument("json_file", help="Path to the scenario JSON file.")
    parser.add_argument("--video", type=str, default=None,
                        help="Path of MP4 to record the animation")
    parser.add_argument("--views", nargs="*", default=["default"],
                        help=("Additional camera views to display. "
                              "Available options: top, bottom, left, right, front, back. "
                              "The default view is always included."))
    parser.add_argument("--width", type=int, default=1024,
                        help="Width of each plot window in pixels (default: 1024)")
    parser.add_argument("--height", type=int, default=768,
                        help="Height of each plot window in pixels (default: 768)")
    args = parser.parse_args()

    # Load JSON data.
    with open(args.json_file, "r") as f:
        data = json.load(f)

    # Check for the required keys.
    required_keys = ["recording", "goal_state", "goal_tolerance", "bounds", 
                     "replanning_times", "travel_distance", "travel_time", "success"]
    for key in required_keys:
        if key not in data:
            print(f"Invalid JSON: Missing '{key}' key.")
            return

    # Print out the run statistics.
    num_replannings = len(data["replanning_times"])
    travel_time = data["travel_time"]
    travel_distance = data["travel_distance"]
    success = data["success"]
    end_condition = data.get("end_condition", "Unknown")
    print("Run Statistics:")
    print(f"  Number of replannings: {num_replannings}")
    print(f"  Travel time: {travel_time}")
    print(f"  Travel distance: {travel_distance}")
    print(f"  Success: {success}")
    print(f"  End condition: {end_condition}")

    recording = data["recording"]
    dt = recording["dt"]  # time (in seconds) between frames
    goal_state = data["goal_state"]
    goal_tolerance = data["goal_tolerance"]
    bounds = data["bounds"]
    min_bounds = bounds["min"]
    max_bounds = bounds["max"]

    # Compute the center and diagonal of the bounds.
    center = [ (min_bounds[i] + max_bounds[i]) / 2 for i in range(3) ]
    diag = np.linalg.norm(np.array(max_bounds) - np.array(min_bounds))
    ROBOT_RADIUS = 0.5  # constant for the robot sphere radius

    # Build the list of camera setups from the views.
    camera_setups = []
    for view in args.views:
        view = view.lower()
        try:
            cam = get_camera_position(view, center, diag)
            camera_setups.append((view, cam))
        except ValueError as e:
            print(f"Warning: {e}. Skipping this view.")

    # Determine subplot grid dimensions.
    total_views = len(camera_setups)
    ncols = math.ceil(math.sqrt(total_views))
    nrows = math.ceil(total_views / ncols)
    width = args.width * ncols
    height = args.height * nrows
    
    # Create a multi-panel Plotter.
    plotter = pv.Plotter(shape=(nrows, ncols), window_size=(width, height))
    plotter.set_background("white")

    # For each subplot, add static elements (bounds, goal) and set the corresponding camera.
    for idx, (view_name, cam) in enumerate(camera_setups):
        row, col = idx // ncols, idx % ncols
        plotter.subplot(row, col)
        # Add a wireframe box showing the bounds.
        bounds_box = pv.Box(bounds=(min_bounds[0], max_bounds[0],
                                    min_bounds[1], max_bounds[1],
                                    min_bounds[2], max_bounds[2]))
        plotter.add_mesh(bounds_box, style='wireframe', color=BOUNDS_COLOR, opacity=0.5,
                         name=f"BoundsBox_{idx}")
        # Add goal sphere.
        goal_sphere = generate_sphere_mesh(goal_state, goal_tolerance, resolution=20)
        plotter.add_mesh(goal_sphere, color=GOAL_COLOR, opacity=0.8, name=f"Goal_{idx}")

        # Set the subplot's camera position.
        plotter.camera_position = cam

    # Optional video recording.
    if args.video:
        video_path = Path(args.video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.open_movie(str(video_path), quality=5)
        print(f"Recording video to {video_path}")

    # Open the interactive window in non-blocking mode.
    plotter.show(interactive_update=True)

    # Animation loop: iterate over each frame.
    for frame in recording["frames"]:
        # For each subplot update dynamic actors (robot, obstacles, path).
        for idx, (_, cam) in enumerate(camera_setups):
            row, col = idx // ncols, idx % ncols
            plotter.subplot(row, col)
            # Remove previous dynamic actors from this subplot.
            dynamic_keys = [name for name in list(plotter.actors.keys())
                            if name.startswith(f"Robot_{idx}") or
                               name.startswith(f"Obstacle_{idx}_") or
                               name.startswith(f"Path_{idx}")]
            for key in dynamic_keys:
                plotter.remove_actor(key)

            # Add the robot as a red sphere.
            robot_pos = frame["robot"]
            robot_sphere = generate_sphere_mesh(robot_pos, ROBOT_RADIUS, resolution=20)
            plotter.add_mesh(robot_sphere, color=ROBOT_COLOR, opacity=1.0,
                             name=f"Robot_{idx}", reset_camera=False)

            # Add obstacles as blue spheres.
            for j, obs in enumerate(frame["obstacles"]):
                center_obs = obs["center"]
                radius = obs["radius"]
                obs_sphere = generate_sphere_mesh(center_obs, radius, resolution=20)
                plotter.add_mesh(obs_sphere, color=OBSTACLE_COLOR, opacity=OBSTACLE_OPACITY,
                                 name=f"Obstacle_{idx}_{j}", reset_camera=False)

            # Add the path if available.
            if frame.get("path") is not None and len(frame["path"]) > 0:
                path_points = np.array(frame["path"])
                n_points = path_points.shape[0]
                # Create connectivity for a single polyline.
                cells = np.hstack(([n_points], np.arange(n_points)))
                polyline = pv.PolyData(path_points, lines=cells)

                if frame.get("replanning_trigger") is None:
                    path_color = PATH_COLOR
                elif frame["replanning_trigger"] == "PathUnsafe":
                    path_color = REPLANNED_PATH_UNSAFE_COLOR
                elif frame["replanning_trigger"] == "RobotInOHZ":
                    path_color = REPLANNED_PATH_ROBOT_OHZ_COLOR

                plotter.add_mesh(polyline, color=path_color, line_width=3,
                                 name=f"Path_{idx}")

            # Reapply the fixed camera for this subplot.
            plotter.camera_position = cam

        # Render the updated scene across all subplots.
        plotter.render()
        if args.video:
            plotter.write_frame()
        time.sleep(dt)

    if args.video:
        plotter.close()  # Finalize the video file.
        print("Video recording finished.")
    else:
        input("Animation finished. Press Enter to exit...")

if __name__ == "__main__":
    main()