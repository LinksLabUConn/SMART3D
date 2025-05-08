use json::JsonValue;
use num_traits::Float;
use smart_3d::obstacles::{RectangularObstacle, StaticRectangularObstacle};
use smart_3d::rrt::state::RealVectorState;

pub fn real_vector_to_json_array<F: Float, const N: usize>(
    real_vector: &RealVectorState<F, N>,
) -> JsonValue {
    let mut array = JsonValue::new_array();
    for i in 0..N {
        let dimension_value: f64 = real_vector[i].to_f64().expect("Failed to convert to f64");
        array
            .push(JsonValue::Number(dimension_value.into()))
            .expect("Failed to push real vector component to JSON array");
    }
    array
}

pub fn rectangle_to_json<F: Float, const N: usize>(
    rectangle: &StaticRectangularObstacle<F, N>,
) -> JsonValue {
    let json_object = json::object! {
        min_corner: real_vector_to_json_array(&rectangle.min_corner()),
        max_corner: real_vector_to_json_array(&rectangle.max_corner()),
    };
    json_object
}
