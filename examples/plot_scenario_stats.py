# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "numpy",
#   "matplotlib",
# ]
# ///

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv

def load_scenario_data(filename):
    file_path = Path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find '{filename}'. Make sure the file exists.")
    with file_path.open("r") as f:
        data = json.load(f)
    return data["scenarios"]

def compute_success_rates(scenarios):
    success_rates = {}
    for scenario in scenarios:
        num_obstacles = scenario["num_obstacles"]
        obstacle_speed = scenario["obstacle_speed"]
        key = (num_obstacles, obstacle_speed)
        # Compute mean success (True==1, False==0) over all trials.
        successes = [trial["success"] for trial in scenario["trials"]]
        if len(successes) == 0 or np.sum(successes) == 0:
            success_rate = 0.0
        else:
            success_rate = np.mean(successes)
        success_rates[key] = success_rate
    return success_rates

def compute_replanning_metrics(scenarios):
    """
    For each scenario (keyed by (num_obstacles, obstacle_speed)),
    compute two metrics over the trials:
      - replanning_counts: number of replanning events per trial (only for successful trials)
      - avg_replanning_times: average replanning time per trial (for all trials, including unsuccessful ones)
    Returns two dictionaries mapping the scenario key to a list (one value per trial).
    """
    replanning_counts = {}
    avg_replanning_times = {}
    
    for scenario in scenarios:
        num_obstacles = scenario["num_obstacles"]
        obstacle_speed = scenario["obstacle_speed"]
        key = (num_obstacles, obstacle_speed)
        if key not in replanning_counts:
            replanning_counts[key] = []
            avg_replanning_times[key] = []
        
        for trial in scenario["trials"]:
            times = trial["replanning_times"]
            count = len(times)
            if count > 0:
                avg_time = np.mean(times) * 1000 # Convert to milliseconds
                avg_replanning_times[key].append(avg_time)

            # Only include replanning count if the trial was successful.
            if trial["success"]:
                replanning_counts[key].append(count)
    
    return replanning_counts, avg_replanning_times

def compute_all_replanning_times(scenarios):
    """
    For each scenario (keyed by (num_obstacles, obstacle_speed)), collect all replanning times from all trials.
    """
    all_replanning_times = {}
    for scenario in scenarios:
        key = (scenario["num_obstacles"], scenario["obstacle_speed"])
        all_replanning_times.setdefault(key, [])
        for trial in scenario["trials"]:
            # extend with every individual replanning event
            all_replanning_times[key].extend(trial["replanning_times"])
    for key in all_replanning_times:
        # Convert to milliseconds
        all_replanning_times[key] = [t * 1000 for t in all_replanning_times[key]]
    return all_replanning_times

def compute_travel_metrics(scenarios):
    """
    For each scenario (keyed by (num_obstacles, obstacle_speed)),
    compute two travel metrics over the trials:
      - travel_distances: distance traveled per trial (only for successful trials).
      - travel_times: travel time per trial (only for successful trials).
    Returns two dictionaries mapping the scenario key to a list (one value per trial).
    """
    travel_distances = {}
    travel_times = {}
    
    for scenario in scenarios:
        num_obstacles = scenario["num_obstacles"]
        obstacle_speed = scenario["obstacle_speed"]
        key = (num_obstacles, obstacle_speed)
        if key not in travel_distances:
            travel_distances[key] = []
            travel_times[key] = []
            
        for trial in scenario["trials"]:
            if trial["success"]:
                travel_distances[key].append(trial["travel_distance"])
                travel_times[key].append(trial["travel_time"])
    
    return travel_distances, travel_times

def plot_bar(x, y, xlabel, ylabel, title, filename, even_space=False, save_pdf=False, ylim = None, title_font = None, label_font = None, tick_font = None):
    """
    Plot a bar chart and save it to a file. Also save raw data to a CSV file.

    Arguments:
    - x: x-axis values (list of numbers)
    - y: y-axis values (list of numbers)
    - xlabel: x-axis label
    - ylabel: y-axis label
    - title: plot title
    - even_space: whether to use even spacing for x-axis labels (regardless of their numerical vals) (default: True)
    - filename: output file path (PNG image and CSV file)
    - save_pdf: whether to also save the plot as a PDF (default: False)
    - ylim: y-axis limits, min max tuple (default: None, auto-scaled)
    """
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))

    if even_space:
        x_labels = [str(i) for i in x]
        x_positions = list(range(len(x)))
    else:
        x_labels = x
        x_positions = x

    plt.bar(x_positions, y, color="skyblue", alpha=0.7)
    plt.xlabel(xlabel, fontsize = label_font)
    plt.ylabel(ylabel, fontsize = label_font)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.title(title, fontsize = title_font)
    plt.xticks(ticks=x_positions, labels=x_labels, fontsize=tick_font if tick_font else None)
    plt.yticks(fontsize=tick_font if tick_font else None)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(filename)
    if save_pdf:
        pdf_filename = filename.with_suffix('.pdf')
        plt.savefig(pdf_filename, format='pdf')
    plt.close()

    # Export bar data
    csv_file = filename.with_suffix('.csv')
    with csv_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([xlabel, ylabel])
        for xi, yi in zip(x, y):
            writer.writerow([xi, yi])

def plot_box(x_labels, data, xlabel, ylabel, title, filename, save_pdf=False, title_font = None, label_font = None, tick_font = None):
    """
    Plot a box plot and save it to a file. Also save median data to a CSV file.

    Arguments:
    - x_labels: list of labels for each box
    - data: list of lists (each inner list holds the metric values for one x category)
    - xlabel: x-axis label
    - ylabel: y-axis label
    - title: plot title
    - filename: output file path (PNG image and CSV file)
    - save_pdf: whether to also save the plot as a PDF (default: False)
    """
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, patch_artist=True)
    plt.xlabel(xlabel, fontsize = label_font)
    plt.ylabel(ylabel, fontsize = label_font)
    plt.title(title, fontsize = title_font)
    plt.xticks(range(1, len(x_labels) + 1), x_labels, fontsize=tick_font if tick_font else None)
    plt.yticks(fontsize=tick_font if tick_font else None)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    if save_pdf:
        pdf_filename = filename.with_suffix('.pdf')
        plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    plt.close()

    # Compute medians and export
    medians = [float(np.median(d)) for d in data]
    csv_file = filename.with_suffix('.csv')
    with csv_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([xlabel, f"Median {ylabel}"])
        for label, m in zip(x_labels, medians):
            writer.writerow([label, m])

def make_title(main_title, subtitle, seperate_lines=False):
    """
    Create a title for the plot with a main title and a subtitle.
    """
    if seperate_lines:
        return f"{main_title}\n{subtitle}"
    else:
        return f"{main_title} {subtitle}"

def main():

    parser = argparse.ArgumentParser(description="Plot scenario statistics from JSON data.")
    parser.add_argument("scenario_json", type=str, help="Scenario JSON file.")
    parser.add_argument("--pdf", action="store_true", help="Save plots as PDF files in addition to png.")
    parser.add_argument("--large-font", action="store_true", help="Use larger font sizes for labels and title.")
    parser.add_argument("--larger-font", action="store_true", help="Use even larger font sizes for labels and title.")
    args = parser.parse_args()
    json_file = Path(args.scenario_json)
    scenarios = load_scenario_data(json_file)
    save_pdf = args.pdf
    
    if args.larger_font:
        title_font = 23
        label_font = 22
        tick_font = 18
        split_title = True
    elif args.large_font:
        title_font = 20
        label_font = 18
        tick_font = 14
        split_title = True
    else:
        title_font = None
        label_font = None
        tick_font = None
        split_title = False

    # Scenario directory
    scenario_dir = json_file.parent
    plots_dir = scenario_dir / "plots"
    
    # ------------------------
    # Plot Success Rates (Bar Plot)
    # ------------------------
    success_rates = compute_success_rates(scenarios)
    num_obstacles_set = sorted(set(k[0] for k in success_rates.keys()))
    obstacle_speeds_set = sorted(set(k[1] for k in success_rates.keys()))
    
    # For each obstacle speed: success rate vs number of obstacles.
    for obstacle_speed in obstacle_speeds_set:
        x_vals = []
        y_vals = []
        for num_obstacles in num_obstacles_set:
            key = (num_obstacles, obstacle_speed)
            if key in success_rates:
                x_vals.append(num_obstacles)
                y_vals.append(success_rates[key])
        output_file = plots_dir / f"success_rate/vs_num_obstacles/speed_{obstacle_speed}.png"
        plot_bar(
            x_vals, y_vals,
            "Number of Obstacles", "Success Rate",
            make_title("Success Rate vs Number of Obstacles", f"(Obstacle Speed {obstacle_speed} m/s)", split_title),
            output_file,
            even_space=True,
            save_pdf=save_pdf,
            ylim=(0, 1),
            title_font=title_font, label_font=label_font, tick_font=tick_font
        )
    
    # For each number of obstacles: success rate vs obstacle speed.
    for num_obstacles in num_obstacles_set:
        x_vals = []
        y_vals = []
        for obstacle_speed in obstacle_speeds_set:
            key = (num_obstacles, obstacle_speed)
            if key in success_rates:
                x_vals.append(obstacle_speed)
                y_vals.append(success_rates[key])
        output_file = plots_dir / f"success_rate/vs_obstacle_speed/num_obstacles_{num_obstacles}.png"
        plot_bar(
            x_vals, y_vals,
            "Obstacle Speed (m/s)", "Success Rate",
            make_title("Success Rate vs Obstacle Speed", f"({num_obstacles} Obstacles)", split_title),
            output_file,
            even_space=True,
            save_pdf=save_pdf,
            ylim=(0, 1),
            title_font=title_font, label_font=label_font, tick_font=tick_font
        )
    
    # ------------------------
    # Compute Replanning Metrics
    # ------------------------
    replanning_counts, avg_replanning_times = compute_replanning_metrics(scenarios)
    
    # ------------------------
    # Plot Replanning Counts (Box Plot)
    # ------------------------
    # For each obstacle speed: replanning counts vs number of obstacles.
    for obstacle_speed in obstacle_speeds_set:
        x_labels = []
        box_data = []
        for num_obstacles in num_obstacles_set:
            key = (num_obstacles, obstacle_speed)
            if key in replanning_counts:
                x_labels.append(str(num_obstacles))
                box_data.append(replanning_counts[key])
        output_file = plots_dir / f"replannings/vs_num_obstacles/speed_{obstacle_speed}.png"
        plot_box(
            x_labels, box_data,
            "Number of Obstacles", "Number of Replannings",
            make_title("Replannings vs Number of Obstacles", f"(Obstacle Speed {obstacle_speed} m/s)", split_title),
            output_file,
            save_pdf=save_pdf,
            title_font=title_font, label_font=label_font, tick_font=tick_font
        )
    
    # For each number of obstacles: replanning counts vs obstacle speed.
    for num_obstacles in num_obstacles_set:
        x_labels = []
        box_data = []
        for obstacle_speed in obstacle_speeds_set:
            key = (num_obstacles, obstacle_speed)
            if key in replanning_counts:
                x_labels.append(str(obstacle_speed))
                box_data.append(replanning_counts[key])
        output_file = plots_dir / f"replannings/vs_obstacle_speed/num_obstacles_{num_obstacles}.png"
        plot_box(
            x_labels, box_data,
            "Obstacle Speed (m/s)", "Number of Replannings",
            make_title("Replannings vs Obstacle Speed", f"({num_obstacles} Obstacles)", split_title),
            output_file,
            save_pdf=save_pdf,
            title_font=title_font, label_font=label_font, tick_font=tick_font
        )

    # ------------------------
    # Plot Average Replanning Time (Box Plot)
    # ------------------------
    # For each obstacle speed: average replanning time vs number of obstacles.
    for obstacle_speed in obstacle_speeds_set:
        x_labels = []
        box_data = []
        for num_obstacles in num_obstacles_set:
            key = (num_obstacles, obstacle_speed)
            if key in avg_replanning_times:
                if len(avg_replanning_times[key]) == 0:
                    print(f"Warning: No average replanning times for key {key}.")
                else:
                    x_labels.append(str(num_obstacles))
                    box_data.append(avg_replanning_times[key])
        output_file = plots_dir / f"avg_replanning_time/vs_num_obstacles/speed_{obstacle_speed}.png"
        plot_box(
            x_labels, box_data,
            "Number of Obstacles", "Average Replanning Time (ms)",
            make_title("Avg Replanning Time vs Number of Obstacles", f"(Obstacle Speed {obstacle_speed} m/s)", split_title),
            output_file,
            save_pdf=save_pdf,
            title_font=title_font, label_font=label_font, tick_font=tick_font
        )
    
    # For each number of obstacles: average replanning time vs obstacle speed.
    for num_obstacles in num_obstacles_set:
        x_labels = []
        box_data = []
        for obstacle_speed in obstacle_speeds_set:
            key = (num_obstacles, obstacle_speed)
            if key in avg_replanning_times:
                if len(avg_replanning_times[key]) == 0:
                    print(f"Warning: No average replanning times for key {key}.")
                else:
                    x_labels.append(str(obstacle_speed))
                    box_data.append(avg_replanning_times[key])
        output_file = plots_dir / f"avg_replanning_time/vs_obstacle_speed/num_obstacles_{num_obstacles}.png"
        plot_box(
            x_labels, box_data,
            "Obstacle Speed (m/s)", "Average Replanning Time (ms)",
            make_title("Avg Replanning Time vs Obstacle Speed", f"({num_obstacles} Obstacles)", split_title),
            output_file,
            save_pdf=save_pdf,
            title_font=title_font, label_font=label_font, tick_font=tick_font
        )
    
    # ------------------------
    # Plot All Replanning Times (Box Plot)
    # ------------------------ 
    all_replanning_times = compute_all_replanning_times(scenarios)
    for speed in obstacle_speeds_set:
        labels, data = [], []
        for num in num_obstacles_set:
            key = (num, speed)
            if key in all_replanning_times:
                if len(all_replanning_times[key]) == 0:
                    print(f"Warning: No replanning times for key {key}.")
                else:
                    labels.append(str(num))
                    data.append(all_replanning_times[key])
        plot_box(
            labels, data,
            "Number of Obstacles", "Replanning Event Time (ms)",
            make_title("All Replanning Times vs # Obstacles", f"(Obstacle Speed {speed} m/s)", split_title),
            plots_dir / f"replanning_times/vs_num_obstacles/speed_{speed}.png",
            save_pdf=save_pdf,
            title_font=title_font, label_font=label_font, tick_font=tick_font
        )
    for num in num_obstacles_set:
        labels, data = [], []
        for speed in obstacle_speeds_set:
            key = (num, speed)
            if key in all_replanning_times:
                if len(all_replanning_times[key]) == 0:
                    print(f"Warning: No replanning times for key {key}.")
                else:
                    labels.append(str(speed))
                    data.append(all_replanning_times[key])
        plot_box(
            labels, data,
            "Obstacle Speed (m/s)", "Replanning Event Time (ms)",
            make_title("All Replanning Times vs Speed", f"({num} Obstacles)", split_title),
            plots_dir / f"replanning_times/vs_obstacle_speed/num_obstacles_{num}.png",
            save_pdf=save_pdf,
            title_font=title_font, label_font=label_font, tick_font=tick_font
        )

    # ------------------------
    # Compute Travel Metrics (only for successful runs)
    # ------------------------
    travel_distances, travel_times = compute_travel_metrics(scenarios)
    
    # ------------------------
    # Plot Travel Distance (Box Plot)
    # ------------------------
    # For each obstacle speed: travel distance vs number of obstacles.
    for obstacle_speed in obstacle_speeds_set:
        x_labels = []
        box_data = []
        for num_obstacles in num_obstacles_set:
            key = (num_obstacles, obstacle_speed)
            if key in travel_distances:
                x_labels.append(str(num_obstacles))
                box_data.append(travel_distances[key])
        output_file = plots_dir / f"travel_distance/vs_num_obstacles/speed_{obstacle_speed}.png"
        plot_box(
            x_labels, box_data,
            "Number of Obstacles", "Travel Distance (m)",
            make_title("Travel Distance vs Number of Obstacles", f"(Obstacle Speed {obstacle_speed} m/s)", split_title),
            output_file,
            save_pdf=save_pdf,
            title_font=title_font, label_font=label_font, tick_font=tick_font
        )
    
    # For each number of obstacles: travel distance vs obstacle speed.
    for num_obstacles in num_obstacles_set:
        x_labels = []
        box_data = []
        for obstacle_speed in obstacle_speeds_set:
            key = (num_obstacles, obstacle_speed)
            if key in travel_distances:
                x_labels.append(str(obstacle_speed))
                box_data.append(travel_distances[key])
        output_file = plots_dir / f"travel_distance/vs_obstacle_speed/num_obstacles_{num_obstacles}.png"
        plot_box(
            x_labels, box_data,
            "Obstacle Speed (m/s)", "Travel Distance (m)",
            make_title(f"Travel Distance vs Obstacle Speed", f"({num_obstacles} Obstacles)", split_title),
            output_file,
            save_pdf=save_pdf,
            title_font=title_font, label_font=label_font, tick_font=tick_font
        )
    
    # ------------------------
    # Plot Travel Time (Box Plot)
    # ------------------------
    # For each obstacle speed: travel time vs number of obstacles.
    for obstacle_speed in obstacle_speeds_set:
        x_labels = []
        box_data = []
        for num_obstacles in num_obstacles_set:
            key = (num_obstacles, obstacle_speed)
            if key in travel_times:
                x_labels.append(str(num_obstacles))
                box_data.append(travel_times[key])
        output_file = plots_dir / f"travel_time/vs_num_obstacles/speed_{obstacle_speed}.png"
        plot_box(
            x_labels, box_data,
            "Number of Obstacles", "Travel Time (s)",
            make_title("Travel Time vs Number of Obstacles", f"(Obstacle Speed {obstacle_speed} m/s)", split_title),
            output_file,
            save_pdf=save_pdf,
            title_font=title_font, label_font=label_font, tick_font=tick_font
        )
    
    # For each number of obstacles: travel time vs obstacle speed.
    for num_obstacles in num_obstacles_set:
        x_labels = []
        box_data = []
        for obstacle_speed in obstacle_speeds_set:
            key = (num_obstacles, obstacle_speed)
            if key in travel_times:
                x_labels.append(str(obstacle_speed))
                box_data.append(travel_times[key])
        output_file = plots_dir / f"travel_time/vs_obstacle_speed/num_obstacles_{num_obstacles}.png"
        plot_box(
            x_labels, box_data,
            "Obstacle Speed (m/s)", "Travel Time (s)",
            make_title("Travel Time vs Obstacle Speed", f"({num_obstacles} Obstacles)", split_title),
            output_file,
            save_pdf=save_pdf,
            title_font=title_font, label_font=label_font, tick_font=tick_font
        )

if __name__ == "__main__":
    main()