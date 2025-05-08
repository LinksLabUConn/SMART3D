//! Module to visualize 2D RRT trees and paths using macroquad.
//! This module is not a standalone example, but is used by the other 2D examples.

use macroquad::{color, prelude::*};
use num_traits::Float;
use smart_3d::obstacles::{RectangularObstacle, SphericalObstacle};
use smart_3d::rrt::RealVectorState;

pub struct StatePixelConverter<F: Float> {
    screen_width: usize,
    screen_height: usize,
    min_x: F,
    min_y: F,
    max_x: F,
    max_y: F,
}

impl<F: Float> StatePixelConverter<F> {
    pub fn new(
        screen_width: usize,
        screen_height: usize,
        min_x: F,
        min_y: F,
        max_x: F,
        max_y: F,
    ) -> Self {
        Self {
            screen_width,
            screen_height,
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    #[inline]
    fn state_to_pixel(&self, state: &RealVectorState<F, 2>) -> (i32, i32) {
        let w = F::from(self.screen_width).unwrap();
        let h = F::from(self.screen_height).unwrap();

        let x = ((state[0] - self.min_x) / (self.max_x - self.min_x) * w).round();
        let y = ((state[1] - self.min_y) / (self.max_y - self.min_y) * h).round();

        (x.to_i32().unwrap(), y.to_i32().unwrap())
    }

    /// Converts a radius in state‐space units to pixels (assuming uniform scaling on x axis).
    #[inline]
    fn radius_to_pixel(&self, radius: F) -> f32 {
        let span_x = (self.max_x - self.min_x).to_f32().unwrap();
        radius.to_f32().unwrap() * (self.screen_width as f32 / span_x)
    }
}

#[allow(dead_code)]
pub struct Canvas2D<F: Float> {
    converter: StatePixelConverter<F>,
}

#[allow(dead_code)]
impl<F: Float> Canvas2D<F> {
    /// Create a new 2D window‐space converter
    pub fn new(
        screen_width: usize,
        screen_height: usize,
        min_x: F,
        min_y: F,
        max_x: F,
        max_y: F,
    ) -> Self {
        Self {
            converter: StatePixelConverter::new(
                screen_width,
                screen_height,
                min_x,
                min_y,
                max_x,
                max_y,
            ),
        }
    }

    /// Draw a single RRT edge
    pub fn draw_edge(
        &self,
        start: &RealVectorState<F, 2>,
        end: &RealVectorState<F, 2>,
        thickness: f32,
        color: color::Color,
    ) {
        let (x0, y0) = self.converter.state_to_pixel(start);
        let (x1, y1) = self.converter.state_to_pixel(end);
        draw_line(x0 as f32, y0 as f32, x1 as f32, y1 as f32, thickness, color);
    }

    /// Draw a poly‐line path
    pub fn draw_path(&self, path: &[RealVectorState<F, 2>], thickness: f32, color: color::Color) {
        if path.len() < 2 {
            return;
        }
        for pair in path.windows(2) {
            self.draw_edge(&pair[0], &pair[1], thickness, color);
        }
    }

    /// Draw a filled circular obstacle
    pub fn draw_circular_obstacle(&self, obs: &dyn SphericalObstacle<F, 2>, color: color::Color) {
        let (cx, cy) = self.converter.state_to_pixel(obs.center());
        let r = self.converter.radius_to_pixel(obs.radius());
        draw_circle(cx as f32, cy as f32, r, color);
    }

    /// Draw a filled rectangular obstacle
    pub fn draw_rectangular_obstacle(
        &self,
        obs: &dyn RectangularObstacle<F, 2>,
        color: color::Color,
    ) {
        let (min_x, min_y) = self.converter.state_to_pixel(obs.min_corner());
        let (max_x, max_y) = self.converter.state_to_pixel(obs.max_corner());
        draw_rectangle(
            min_x as f32,
            min_y as f32,
            (max_x - min_x) as f32,
            (max_y - min_y) as f32,
            color,
        );
    }

    /// Draw the outline of a circle
    pub fn draw_circle_outline_radius_px(
        &self,
        center: &RealVectorState<F, 2>,
        radius: f32,
        thickness: f32,
        color: color::Color,
    ) {
        let (cx, cy) = self.converter.state_to_pixel(center);
        draw_circle_lines(cx as f32, cy as f32, radius, thickness, color);
    }

    /// Draw the outline of a circle
    pub fn draw_circle_outline(
        &self,
        center: &RealVectorState<F, 2>,
        radius: F,
        thickness: f32,
        color: color::Color,
    ) {
        let (cx, cy) = self.converter.state_to_pixel(center);
        let radius = self.converter.radius_to_pixel(radius);
        draw_circle_lines(cx as f32, cy as f32, radius, thickness, color);
    }

    pub fn draw_circle_radius_px(
        &self,
        center: &RealVectorState<F, 2>,
        radius: f32,
        color: color::Color,
    ) {
        let (cx, cy) = self.converter.state_to_pixel(center);
        draw_circle(cx as f32, cy as f32, radius, color);
    }

    /// Draw a filled circle at an arbitrary center/radius
    pub fn draw_circle(&self, center: &RealVectorState<F, 2>, radius: F, color: color::Color) {
        let (cx, cy) = self.converter.state_to_pixel(center);
        let r = self.converter.radius_to_pixel(radius);
        draw_circle(cx as f32, cy as f32, r, color);
    }

    pub fn draw_target_radius_px(
        &self,
        goal: &RealVectorState<F, 2>,
        target_radius_px: f32,
        color: color::Color,
        background_color: color::Color,
    ) {
        let (goal_x, goal_y) = self.converter.state_to_pixel(goal);
        draw_circle(goal_x as f32, goal_y as f32, target_radius_px, color);
        draw_circle(
            goal_x as f32,
            goal_y as f32,
            target_radius_px * (2.0 / 3.0),
            background_color,
        );
        draw_circle(goal_x as f32, goal_y as f32, target_radius_px / 3.0, color);
    }

    pub fn draw_target(
        &self,
        goal: &RealVectorState<F, 2>,
        target_radius: F,
        color: color::Color,
        background_color: color::Color,
    ) {
        let target_radius_px = self.converter.radius_to_pixel(target_radius);
        self.draw_target_radius_px(goal, target_radius_px, color, background_color);
    }
}
