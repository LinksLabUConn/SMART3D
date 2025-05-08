use super::json_util::{real_vector_to_json_array, rectangle_to_json};
use json::JsonValue;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use smart_3d::obstacles::{SphericalObstacle, StaticRectangularObstacle};
use smart_3d::rrt::state::RealVectorState;
use smart_3d::smart::SmartUpdateResult;
use smart_3d::smart::{obstacle::SphericalObstacleWithOhz, Node, ReplanningTriggerCondition};

#[derive(Clone, Serialize, Deserialize)]
pub struct ScenarioFrame<F: Float, const N: usize> {
    pub robot: RealVectorState<F, N>,
    pub obstacles: Vec<SphericalObstacleWithOhz<F, N>>,
    pub nodes: Option<Vec<Node<F, N>>>,
    pub path: Option<Vec<RealVectorState<F, N>>>,
    pub replanning_trigger: Option<ReplanningTriggerCondition>,
    pub planning_result: Option<SmartUpdateResult>,
}

impl<F: Float, const N: usize> ScenarioFrame<F, N> {
    pub fn to_json(&self) -> JsonValue {
        let mut json_object = json::object! {
            robot: real_vector_to_json_array(&self.robot),
            obstacles: JsonValue::new_array(),
            path: JsonValue::Null,
            planning_result: JsonValue::Null,
        };

        for obstacle in &self.obstacles {
            let obstacle_json = json::object! {
                center: real_vector_to_json_array(obstacle.obstacle().center()),
                radius: JsonValue::Number(obstacle.obstacle().radius().to_f64().unwrap().into()),
                ohz: json::object! {
                    center: real_vector_to_json_array(&obstacle.ohz().center()),
                    radius: JsonValue::Number(obstacle.ohz().radius().to_f64().unwrap().into()),
                },
            };
            json_object["obstacles"]
                .push(obstacle_json)
                .expect("Failed to push obstacle to JSON array");
        }

        if self.path.is_some() {
            let mut path_json = JsonValue::new_array();
            for path_point in self.path.as_ref().unwrap() {
                path_json
                    .push(real_vector_to_json_array(path_point))
                    .expect("Failed to push path point to JSON array");
            }
            json_object["path"] = path_json;
        }

        if self.replanning_trigger.is_some() {
            json_object["replanning_trigger"] =
                json::JsonValue::String(self.replanning_trigger.clone().unwrap().to_string());
        } else {
            json_object["replanning_trigger"] = JsonValue::Null;
        }

        if self.planning_result.is_some() {
            json_object["planning_result"] =
                json::JsonValue::String(self.planning_result.clone().unwrap().to_string());
        } else {
            json_object["planning_result"] = JsonValue::Null;
        }
        json_object
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ScenarioRecording<F: Float, const N: usize> {
    frames: Vec<ScenarioFrame<F, N>>,
    dt: F, // Time step
    static_rectangles: Vec<StaticRectangularObstacle<F, N>>,
}

impl<F: Float, const N: usize> ScenarioRecording<F, N> {
    pub fn new(static_rectangles: Vec<StaticRectangularObstacle<F, N>>, dt: F) -> Self {
        Self {
            static_rectangles,
            frames: Vec::new(),
            dt,
        }
    }

    pub fn frames(&self) -> &Vec<ScenarioFrame<F, N>> {
        &self.frames
    }

    pub fn dt(&self) -> F {
        self.dt
    }

    pub fn add_frame(&mut self, frame: ScenarioFrame<F, N>) {
        self.frames.push(frame);
    }

    pub fn to_json(&self) -> JsonValue {
        let mut static_rectangles_json_array = JsonValue::new_array();
        for rectangle in &self.static_rectangles {
            static_rectangles_json_array
                .push(rectangle_to_json(rectangle))
                .expect("Failed to push static rectangle to JSON array");
        }

        let mut frames_json_array = JsonValue::new_array();
        for frame in &self.frames {
            frames_json_array
                .push(frame.to_json())
                .expect("Failed to push frame to JSON array");
        }
        let json_object = json::object! {
            static_rectangles: static_rectangles_json_array,
            frames: frames_json_array,
            dt: JsonValue::Number(self.dt.to_f64().unwrap().into()),
        };
        json_object
    }

    /// Removes the nodes from all frames to save space.
    pub fn remove_nodes(&mut self) {
        for frame in &mut self.frames {
            frame.nodes = None;
        }
    }
}
