use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Debug)]
pub enum ScenarioEndCondition {
    RobotReachedGoal,
    ReplanningFailed,
    ReplanningError,
    RobotCollidedWithObstacle,
    OutOfTime,
    InitialPlanFailed,
    InProgress,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    pub end_condition: ScenarioEndCondition,
    pub replanning_times: Vec<Duration>,
    pub failed_replanning_attempts: usize,
    pub updates_without_replanning: usize,
    pub travel_distance: f32,
    pub travel_time: f32,
}

impl ScenarioResult {
    pub fn new() -> Self {
        Self {
            end_condition: ScenarioEndCondition::InProgress,
            replanning_times: Vec::new(),
            failed_replanning_attempts: 0,
            updates_without_replanning: 0,
            travel_distance: 0.0,
            travel_time: 0.0,
        }
    }
}

impl std::fmt::Display for ScenarioResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "End Condition: {:?}\nReplanning Times: {:?}\nFailed Replanning Attempts: {}\nUpdates Without Replanning: {}\nTravel Distance: {}\nTravel Time: {}",
            self.end_condition,
            self.replanning_times,
            self.failed_replanning_attempts,
            self.updates_without_replanning,
            self.travel_distance,
            self.travel_time
        )
    }
}
