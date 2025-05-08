use std::collections::HashMap;

use crate::obstacles::{SphericalObstacle, StaticSphericalObstacle};
use crate::rrt::rrt_star;
use crate::rrt::{
    NearestNeighbors, RealVectorState, SamplingDistribution, TerminationCondition, ValidityChecker,
};
use crate::smart::{
    node::convert_rrt_star_nodes_to_smart_nodes, Node, ReplanningTriggerCondition,
    SmartUpdateResult, SphericalObstacleWithOhz,
};
use crate::util::ordered_float::OrderedFloat;
use num_traits::Float;

/// Self-Morphing Adaptive Replanning Tree (SMART) algorithm.
pub struct SMART<F: Float, const N: usize, NN> {
    // Problem definition
    goal: RealVectorState<F, N>, // The goal state
    goal_tolerance: F,           // The goal tolerance
    static_validity_checker: Box<dyn ValidityChecker<F, N>>, // The static validity checker

    // SMART planning parameters
    sampling_distribution: Box<dyn SamplingDistribution<F, N>>,
    lrz_radius: F,
    lsr_initial_radius: F, // The initial radius of the LSR (Local Search Region)
    lsr_expansion_factor: F, // The factor by which the LSR expands
    lsr_max_radius: F,     // The maximum radius of the LSR
    hot_node_neighborhood_radius: F,
    path_node_radius: F, // When we look for a path, we search for starting nodes within this radius
    replan_when_path_is_safe_but_robot_inside_ohz: bool, // Whether to replan when the path is safe but the robot is inside an OHZ

    // OHZ margin is a special parameter for when the robot is inside an OHZ.
    // When the robot is inside an OHZ, the OHZ is shrunk to the distance between the robot and obstacle.
    // However, for planning purposes, we need to further shrink the OHZ by a small margin such that the robot is not inside the OHZ.
    // This is done to avoid the robot being inside the OHZ. If the robot is inside the OHZ, all validity checks for all edges to/from the robot will fail.
    // This parameter is used regardless of the value of `replan_when_path_is_safe_but_robot_inside_ohz`.
    ohz_shrinkage_margin: F,

    // Keep track of current phase
    initial_planning_phase: bool, // Whether we are in the initial planning phase

    // SMART tree state
    nodes: Vec<Node<F, N>>, // The nodes in the tree
    nearest_neighbors: NN,

    // SMART inputs (provided by user during each update)
    cpr_obstacles: Vec<StaticSphericalObstacle<F, N>>, // The obstacles with which intersect the LRZ (derived from dynamic obstacles).
    // We use a vector of static obstacles, but these are actually one snapshot of the dynamic obstacles at the given timestep.
    max_cpr_radius: F, // The maximum radius for which dynamic obstacle collision checking is needed (derived from cpr_obstacles).
    // max_cpr_radius is the maximum distance between the current state and the furthest point of any CPR obstacle.
    current_state: RealVectorState<F, N>, // The current state of the robot
    robot_inside_ohzs: Vec<StaticSphericalObstacle<F, N>>, // The obstacles with which the robot is inside the OHZ (derived from dynamic obstacles).

    // SMART path (updated during each update)
    current_path: Option<Vec<usize>>,

    // Temporarily variables created and consumed during each update
    tree_id_counter: i32,
    disjoint_tree_roots: Vec<(usize, Option<usize>)>, // (node_id, old_parent_id)
    pruned_nodes: Vec<(usize, Option<usize>)>,        // (node_id, old_parent_id)
    optimization_start_nodes: Vec<usize>,
    replanning_triggered_by: Option<ReplanningTriggerCondition>,
}

#[derive(Clone)]
struct HotNode<F> {
    node_id: usize,
    feasible_neighbors_of_different_tree: Vec<usize>,
    utility: F,
}

impl<F: Float + std::fmt::Debug, const N: usize, NN: NearestNeighbors<F, N>> SMART<F, N, NN> {
    const MAIN_TREE_ID: i32 = 0; // The ID of the main tree
    const PRUNED_TREE_ID: i32 = -1; // The ID of a pruned node

    /// Creates a new SMART planner from an RRT star tree.
    pub fn new_from_initial_rrt_star_tree(
        rrt_star_nodes: &Vec<rrt_star::Node<F, N>>,
        rrt_star_nearest_neighbors: NN,
        start: RealVectorState<F, N>,
        goal: RealVectorState<F, N>,
        goal_tolerance: F,
        static_validity_checker: Box<dyn ValidityChecker<F, N>>,
        sampling_distribution: Box<dyn SamplingDistribution<F, N>>,
        lrz_radius: F,
        lsr_initial_radius: F,
        lsr_expansion_factor: F,
        lsr_max_radius: F,
        hot_node_neighborhood_radius: F,
        path_node_radius: F,
        replan_when_path_is_safe_but_robot_inside_ohz: bool,
        ohz_shrinkage_margin: F,
    ) -> Self {
        // Create the SMART planner
        Self {
            goal,
            goal_tolerance,
            static_validity_checker,
            sampling_distribution,
            lrz_radius,
            lsr_initial_radius,
            lsr_expansion_factor,
            lsr_max_radius,
            hot_node_neighborhood_radius,
            path_node_radius,
            replan_when_path_is_safe_but_robot_inside_ohz,
            ohz_shrinkage_margin,
            initial_planning_phase: true,
            nodes: convert_rrt_star_nodes_to_smart_nodes(rrt_star_nodes),
            nearest_neighbors: rrt_star_nearest_neighbors,
            cpr_obstacles: Vec::new(),
            max_cpr_radius: F::zero(),
            current_state: start.clone(),
            robot_inside_ohzs: Vec::new(),
            current_path: None,
            tree_id_counter: 0,
            disjoint_tree_roots: Vec::new(),
            pruned_nodes: Vec::new(),
            optimization_start_nodes: Vec::new(),
            replanning_triggered_by: None,
        }
    }

    /// Optimize the full tree from the initial plan.
    pub fn initial_plan_optimize_full_tree(&mut self) {
        assert!(
            self.initial_planning_phase,
            "Function only valid for initial planning"
        );
        self.rewire_cascade(0);
    }

    pub fn initial_plan_add_node_at_start(&mut self) {
        assert!(
            self.initial_planning_phase,
            "Function only valid for initial planning"
        );
        let new_node_state = self.current_state.clone();
        self.add_new_node_at_state_and_connect(new_node_state);
        self.tree_optimization();
    }

    pub fn initial_plan_build_path(&mut self) {
        assert!(
            self.initial_planning_phase,
            "Function only valid for initial planning"
        );
        assert!(self.debug_check_all_nodes_in_main_tree());
        assert!(self.debug_check_tree_consistency());
        assert!(self.debug_check_costs());

        self.current_path = self.build_planned_path();
        self.initial_planning_phase = false; // Set the initial planning phase to false
    }

    /// Update the SMART planner and replan if necessary.
    ///
    /// Parameters:
    /// - `current_state`: The current state.
    /// - `path_index`: The index of the next state in the current path.
    /// - `dynamic_obstacles`: The dynamic obstacles with OHZ.
    /// - `termination`: The termination condition for the update.
    ///                  The termination condition only applies to the time taken for pruning, reconnection and sampling phases.
    ///                  Tree optimization and path building will be performed after the termination condition is met.
    pub fn update<TC>(
        &mut self,
        current_state: RealVectorState<F, N>,
        path_index: usize,
        dynamic_obstacles: &Vec<SphericalObstacleWithOhz<F, N>>,
        mut termination_condition: Option<&mut TC>,
    ) -> SmartUpdateResult
    where
        TC: TerminationCondition,
    {
        assert!(
            !self.initial_planning_phase,
            "Function only valid for non-initial planning"
        );

        // Update the current state
        self.current_state = current_state;

        // Check if current state is in collision with static obstacles
        if !self
            .static_validity_checker
            .is_state_valid(&self.current_state)
        {
            return SmartUpdateResult::InCollision;
        }

        // Check if current state is in collision with dynamic obstacles or in OHZ of any of them
        for obs in dynamic_obstacles {
            if obs.obstacle().contains(&self.current_state) {
                return SmartUpdateResult::InCollision;
            }
        }

        // Build the CPR from the dynamic obstacles
        self.build_cpr(dynamic_obstacles);

        // Check if the current path is safe
        let (path_safe, unsafe_node_id) = self.is_path_safe_within_lrz(path_index);
        let hotnode_search_center;
        if !path_safe {
            // The path is not safe, so we need to replan
            self.replanning_triggered_by = Some(ReplanningTriggerCondition::PathUnsafe);
            let unsafe_node_id =
                unsafe_node_id.expect("Path is not safe, but no unsafe node was returned");
            // center of the LSR is the position of the first pruned node on the path
            hotnode_search_center = Some(self.node(unsafe_node_id).state().clone());
        } else if self.replan_when_path_is_safe_but_robot_inside_ohz
            && !self.robot_inside_ohzs.is_empty()
        {
            // The robot is inside an OHZ, so we need to replan
            self.replanning_triggered_by = Some(ReplanningTriggerCondition::RobotInOHZ);
            hotnode_search_center = Some(self.current_state.clone());
        } else {
            // The path is safe and the robot is not inside any OHZ, so we can continue
            return SmartUpdateResult::NoReplanningNeeded;
        }

        self.replanning_triggered_by = Some(ReplanningTriggerCondition::PathUnsafe);

        // Clear the previous replanning results
        self.tree_id_counter = Self::MAIN_TREE_ID;
        assert!(self.disjoint_tree_roots.is_empty());
        assert!(self.pruned_nodes.is_empty());
        assert!(self.optimization_start_nodes.is_empty());
        debug_assert!(self.debug_check_all_nodes_in_main_tree());
        debug_assert!(self.debug_check_tree_consistency());

        self.prune_tree(); // Prune the tree based on the CPR

        debug_assert!(self.debug_check_tree_consistency()); // Check that the tree is consistent after pruning
        debug_assert!(self.debug_check_pruning()); // Check that the pruning was done correctly
        debug_assert!(self.debug_check_costs()); // Check that the costs are correct after pruning

        // Begin replanning with SMART
        self.current_path = self.build_planned_path(); // Try to build a new planned path

        while self.current_path.is_none() {
            // Check if the termination condition is met
            if let Some(termination_condition) = termination_condition.as_mut() {
                if termination_condition.evaluate() {
                    self.tree_optimization(); // Optimize the tree before exiting
                    self.current_path = self.build_planned_path(); // Try to build a new planned path after optimization
                    self.form_single_tree(); // Form a single tree
                    debug_assert!(self.debug_check_tree_consistency()); // Check that the tree is consistent before exiting
                    debug_assert!(self.debug_check_costs()); // Check that the costs are correct before exiting
                    return SmartUpdateResult::ReplanningFailed;
                }
            }

            let mut lsr_radius = self.lsr_initial_radius;
            let mut best_hot_node = None;

            while best_hot_node.is_none() && lsr_radius <= self.lsr_max_radius {
                // Search for a hot node in the LSR
                best_hot_node = self.hot_node_search(
                    &hotnode_search_center.unwrap(),
                    lsr_radius,
                    self.lsr_initial_radius,
                );
                lsr_radius = lsr_radius * self.lsr_expansion_factor;
            }

            if best_hot_node.is_some() {
                let best_hot_node = best_hot_node.unwrap();
                self.reconnect_best_hot_node(&best_hot_node);
            } else {
                // sampling tree repair will sample new nodes to add to the tree
                self.sampling_tree_repair_iteration();
            }

            debug_assert!(self.debug_check_costs());
            debug_assert!(self.debug_check_non_pruned_nodes_are_safe());
            debug_assert!(self.debug_check_tree_consistency());

            self.current_path = self.build_planned_path(); // Search for a new path within the tree
        }

        self.tree_optimization(); // Optimize the tree after replanning
        debug_assert!(self.debug_check_tree_consistency()); // Check that the tree is consistent after optimization

        self.current_path = self.build_planned_path(); // Try to build a new planned path after optimization

        self.form_single_tree(); // Form a single tree
        debug_assert!(self.debug_check_all_nodes_in_main_tree());
        debug_assert!(self.debug_check_tree_consistency());

        SmartUpdateResult::ReplanningSuccessful
    }

    pub fn get_path(&self) -> Option<Vec<RealVectorState<F, N>>> {
        let path_nodes = self.current_path.clone();
        if path_nodes.is_none() {
            return None;
        }
        let path_nodes = path_nodes.unwrap();
        let mut path = Vec::new();
        for &node_id in path_nodes.iter() {
            path.push(self.node(node_id).state().clone());
        }
        Some(path)
    }

    pub fn replanning_triggered_by(&self) -> Option<ReplanningTriggerCondition> {
        self.replanning_triggered_by.clone()
    }

    pub fn get_nodes(&self) -> &Vec<Node<F, N>> {
        &self.nodes
    }

    // Private functions
    // ----------------

    /// Builds the Critical Pruning Zone (CPR). Collisions with dynamic obstacles are checked only in this region.
    /// This function sets the `cpr_obstacles` and `max_cpr_radius` fields.
    /// `cpr_obstacles` contains the obstacles whose OHZ's interesect the LRZ.
    /// `max_cpr_radius` is the maximum distance between the current state and the furthest point of any CPR obstacle.
    fn build_cpr(&mut self, dynamic_obstacles: &Vec<SphericalObstacleWithOhz<F, N>>) {
        self.cpr_obstacles = Vec::new();
        self.robot_inside_ohzs = Vec::new();
        self.max_cpr_radius = self.lrz_radius; // At the very least, the nodes within LRZ must be checked for collision

        for obs in dynamic_obstacles {
            // Distance between obstacle center and current state (lrz center)
            let dist = obs.ohz().center().euclidean_distance(&self.current_state);

            // Check if the obstacles is within the LRZ
            if dist < self.lrz_radius + obs.ohz().radius() {
                // Add the obstacle to the CPR
                if obs.ohz().contains(&self.current_state) {
                    // If the robot is in the OHZ, we decrease the OHZ radius to the distance between the robot and the obstacle center
                    self.cpr_obstacles.push(StaticSphericalObstacle::new(
                        obs.ohz().center().clone(),
                        (dist - F::from(self.ohz_shrinkage_margin).unwrap()).max(F::zero()),
                    ));
                    self.robot_inside_ohzs.push(obs.ohz().clone());
                } else {
                    // If the robot is not in the OHZ, we consider the obstacle's OHZ
                    self.cpr_obstacles.push(obs.ohz().clone());
                }

                // The radius from the current state that has to be checked for collision with this obstacle
                let obstacle_coll_check_radius = self.cpr_obstacles.last().unwrap().radius() + dist;
                // Check if the radius is greater than the current max_cpr_radius
                if obstacle_coll_check_radius > self.max_cpr_radius {
                    self.max_cpr_radius = obstacle_coll_check_radius;
                }
            }
        }
    }

    /// Checks if the current path is safe, considering only the obstacles in the CPR and nodes in LRZ.
    /// - Checks the edge between the current state and the first node in the path.
    /// - Checks the nodes and edges in the path that are within the LRZ.
    ///
    /// If the path is not safe, also returns the id of the first unsafe node on the path.
    fn is_path_safe_within_lrz(&self, current_path_index: usize) -> (bool, Option<usize>) {
        if self.current_path.is_none() {
            return (false, None);
        }

        // check edge between current state and first node in the path
        let first_node_id = self.current_path.as_ref().unwrap()[current_path_index];
        if !self.cpr_validity_check_edge(&self.current_state, self.node(first_node_id).state()) {
            return (false, Some(first_node_id));
        }

        // Create an lrz "obstacle" object just for easy containment checking
        let lrz = StaticSphericalObstacle::new(self.current_state.clone(), self.lrz_radius);

        for &node_id in self.current_path.as_ref().unwrap()[current_path_index + 1..].iter() {
            if lrz.contains(self.node(node_id).state()) {
                // Check if the node's state is valid
                if !self.cpr_validity_check_state(self.node(node_id).state()) {
                    return (false, Some(node_id));
                }
            }
            if let Some(parent_id) = self.node(node_id).parent {
                // Check if the edge is in the lrz
                if lrz.contains(self.node(parent_id).state())
                    && !self.cpr_validity_check_edge(
                        self.node(parent_id).state(),
                        self.node(node_id).state(),
                    )
                {
                    return (false, Some(node_id));
                }
            }
        }
        (true, None)
    }

    /// Prunes nodes and edges that are in the CPR.
    fn prune_tree(&mut self) {
        let cpr_obstacles = self.cpr_obstacles.clone();

        let mut pruned_nodes = Vec::new();
        // Prune all nodes within the CPR obstacles
        for obstacle in &cpr_obstacles {
            // Get all nodes within the obstacle's radius
            let nodes_within_radius: Vec<usize> = self
                .nearest_neighbors
                .within_radius(obstacle.center(), obstacle.radius());

            for node_id in nodes_within_radius {
                if self.node(node_id).tree_id == Self::PRUNED_TREE_ID {
                    continue; // The node is already pruned
                }

                debug_assert!(!self.cpr_validity_check_state(self.node(node_id).state()),
                    "Node is within obstacle radius but validity check failed. This should not happen. 
                    Node ID: {}, Obstacle center: {:?}, Obstacle radius: {:?}, Node state: {:?}, Distance: {:?}",
                    node_id, obstacle.center(), obstacle.radius(), self.node(node_id).state(), self.node(node_id).state().euclidean_distance(obstacle.center()),
                );

                // Mark the node as pruned
                self.node_mut(node_id).tree_id = Self::PRUNED_TREE_ID;

                // Store the old parent ID
                let old_parent_id = self.node(node_id).parent;

                // Remove the parent-child relationship
                self.set_node_parent(node_id, None);

                // Set the cumulative cost to infinity
                self.node_mut(node_id).cumulative_cost = F::infinity();

                // Add the node to the pruned nodes list
                pruned_nodes.push((node_id, old_parent_id));
            }
        }

        let mut disjoint_tree_roots = Vec::new();

        // Make the children of the pruned nodes disjoint trees
        for &(node_id, _) in &pruned_nodes {
            // Get the children of the pruned node
            let children = self.node(node_id).children.clone();
            for child_id in children {
                // If the child was pruned, skip it
                if self.node(child_id).tree_id == Self::PRUNED_TREE_ID {
                    continue;
                }

                // Give the child a new tree id
                self.tree_id_counter += 1;
                let new_tree_id = self.tree_id_counter;

                // Update the parent (this also removes the child from the pruned node's children)
                let old_parent_id = self.node(child_id).parent;
                debug_assert!(old_parent_id.is_some() && old_parent_id.unwrap() == node_id);
                self.set_node_parent(child_id, None);

                // Set the cumulative cost of the child to infinity and update the tree id
                let child = self.node_mut(child_id);
                child.cumulative_cost = F::infinity();
                child.tree_id = new_tree_id;

                // propogate info to children
                self.propagate_info_to_children(child_id);

                // Add the child to the disjoint tree roots
                disjoint_tree_roots.push((child_id, old_parent_id));
            }
        }

        // Get all nodes within the obstacle's radius
        let nodes_within_radius: Vec<usize> = self.nearest_neighbors.within_radius(
            &self.current_state,
            self.max_cpr_radius + self.hot_node_neighborhood_radius,
        );

        for node_id in nodes_within_radius {
            if self.node(node_id).tree_id == Self::PRUNED_TREE_ID {
                continue; // The node is already pruned
            }

            // Check if the edge from the parent is valid
            if let Some(parent_id) = self.node(node_id).parent {
                if !self.cpr_validity_check_edge(
                    self.node(parent_id).state(),
                    self.node(node_id).state(),
                ) {
                    // Break the edge
                    self.set_node_parent(node_id, None);
                    self.tree_id_counter += 1;
                    self.node_mut(node_id).tree_id = self.tree_id_counter;
                    self.node_mut(node_id).cumulative_cost = F::infinity();

                    // Propagate info to children
                    self.propagate_info_to_children(node_id);

                    // Add the node to the disjoint tree roots list
                    disjoint_tree_roots.push((node_id, Some(parent_id)));
                }
            }
        }

        // Update the pruned nodes and disjoint tree roots list
        self.pruned_nodes = pruned_nodes;
        self.disjoint_tree_roots = disjoint_tree_roots;
    }

    fn propagate_info_to_children(&mut self, node_id: usize) {
        // Create a stack to manage traversal
        let mut stack = vec![node_id];

        // While there are nodes to process
        while let Some(current_id) = stack.pop() {
            // Get current node's data
            let cumulative_cost = self.node(current_id).cumulative_cost;
            let tree_id = self.node(current_id).tree_id;

            // we should never be propogating pruned nodes
            debug_assert!(tree_id != Self::PRUNED_TREE_ID);

            // Get a copy of the children of the current node
            let children: Vec<usize> = self.node(current_id).children.clone();

            // Update all children and push them onto the stack
            for &child_id in children.iter() {
                self.node_mut(child_id).cumulative_cost =
                    cumulative_cost + self.node(child_id).edge_cost;
                self.node_mut(child_id).tree_id = tree_id;

                // Add child to the stack for processing
                stack.push(child_id);
            }
        }
    }

    /// Backtracks up the tree to find a path to the goal and returns the path as a vector of node IDs.
    /// The caller is responsible for storing the return value in `self.current_path`.
    fn build_planned_path(&mut self) -> Option<Vec<usize>> {
        // get all nodes within the path node radius of current state
        let path_nodes = self
            .nearest_neighbors
            .within_radius(&self.current_state, self.path_node_radius);

        // filter nodes that are not in the main tree
        let path_nodes: Vec<usize> = path_nodes
            .iter()
            .filter(|&&node_id| self.node(node_id).tree_id == Self::MAIN_TREE_ID)
            .copied()
            .collect();

        // filter out nodes where the edge to the node from the current state is not valid
        let mut path_nodes: Vec<usize> = path_nodes
            .iter()
            .filter(|&&node_id| {
                self.cpr_validity_check_edge(&self.current_state, self.node(node_id).state())
                    && self
                        .static_validity_checker
                        .is_edge_valid(&self.current_state, self.node(node_id).state())
            })
            .copied()
            .collect();

        if path_nodes.is_empty() {
            return None; // No nodes are within the path node radius
        }

        if !self.robot_inside_ohzs.is_empty() {
            // The robot is inside an OHZ
            // in this case, we prioritize the path nodes that are not in the OHZ
            let path_nodes_outside_ohz: Vec<usize> = path_nodes
                .iter()
                .filter(|&&node_id| {
                    !self
                        .robot_inside_ohzs
                        .iter()
                        .any(|ohz| ohz.contains(self.node(node_id).state()))
                })
                .copied()
                .collect();
            if !path_nodes_outside_ohz.is_empty() {
                path_nodes = path_nodes_outside_ohz; // Use the nodes outside the OHZ
            }
        }

        // find the path node with the lowest distance from robot + cumulative cost
        let best_node_id = path_nodes
            .into_iter()
            .min_by_key(|&node_id| {
                let node = self.node(node_id);
                OrderedFloat::from(
                    self.current_state.euclidean_distance(node.state()) + node.cumulative_cost,
                )
            })
            .unwrap();

        // Backtrack up the tree to find the path to the root
        let mut current_node_id = best_node_id;
        let mut path = Vec::new();
        while let Some(parent_id) = self.node(current_node_id).parent {
            path.push(current_node_id);
            current_node_id = parent_id;
        }
        path.push(current_node_id);

        debug_assert!(
            self.node(current_node_id)
                .state()
                .euclidean_distance(&self.goal)
                < self.goal_tolerance
        );

        Some(path)
    }

    /// Finds the best hot node within the Local Search Region (LSR) radius.
    /// Only considers nodes outside the already searched LSR radius.
    ///
    /// A hot node
    ///     - has neighboring nodes that belong to different trees
    ///     - has feasible edges to these neighboring nodes
    ///
    /// Parameters
    /// - `search_center`: The center of the local search region (the position of the closest pruned node).
    /// - `lsr_size`: The size of the Local Search Region (LSR).
    /// - `lsr_already_searched`: The size of the LSR that is already searched.
    ///  
    fn hot_node_search(
        &mut self,
        search_center: &RealVectorState<F, N>,
        lsr_radius: F,
        lsr_already_searched: F,
    ) -> Option<HotNode<F>> {
        // Get all nodes within the LSR radius in sorted order, nearest first
        let nodes_within_lsr: Vec<usize> = self
            .nearest_neighbors
            .within_radius_sorted(search_center, lsr_radius);

        // Find the first node that is outside the already searched LSR radius
        let mut start_index = 0;
        for (i, node_id) in nodes_within_lsr.iter().enumerate() {
            if self.node(*node_id).cumulative_cost > lsr_already_searched {
                start_index = i;
                break;
            }
        }

        // Slice the nodes to only include those outside the already searched LSR radius
        let nodes_within_lsr = &nodes_within_lsr[start_index..];

        let mut best_hot_node: Option<HotNode<F>> = None;
        for &node_id in nodes_within_lsr {
            // If the node is pruned, skip it
            if self.node(node_id).tree_id == Self::PRUNED_TREE_ID {
                continue;
            }

            // Check if the node is a hot node
            let node = self.node(node_id);
            let feasible_neighbors = self.get_feasible_neighbors_of_different_tree(node);

            if feasible_neighbors.is_empty() {
                continue; // No feasible neighbors of different tree. It is not a hot node.
            }

            let utility = self.utility(node);

            if best_hot_node.is_none() || utility < best_hot_node.as_ref().unwrap().utility {
                // Create a new hot node with the current node ID and its feasible neighbors
                best_hot_node = Some(HotNode {
                    node_id,
                    feasible_neighbors_of_different_tree: feasible_neighbors,
                    utility,
                });
            }
        }

        best_hot_node
    }

    fn get_feasible_neighbors_of_different_tree(&self, node: &Node<F, N>) -> Vec<usize> {
        let neighbors = self
            .nearest_neighbors
            .within_radius(node.state(), self.hot_node_neighborhood_radius);
        let mut feasible_neighbors = Vec::new();
        for &neighbor_id in neighbors.iter() {
            if self.node(neighbor_id).tree_id != node.tree_id
                && self.cpr_validity_check_edge(node.state(), self.node(neighbor_id).state())
                && self
                    .static_validity_checker
                    .is_edge_valid(node.state(), self.node(neighbor_id).state())
            {
                feasible_neighbors.push(neighbor_id);
            }
        }
        feasible_neighbors
    }

    fn utility(&self, node: &Node<F, N>) -> F {
        let state = node.state();
        let distance_to_robot = self.current_state.euclidean_distance(&state);

        let distance_to_goal;
        if node.tree_id == Self::MAIN_TREE_ID {
            distance_to_goal = node.cumulative_cost;
        } else {
            distance_to_goal = state.euclidean_distance(&self.goal);
        }

        let utility = F::one() / (distance_to_robot + distance_to_goal);
        utility
    }

    fn reconnect_best_hot_node(&mut self, hot_node: &HotNode<F>) {
        // Reconnect the best hot node to the tree
        let node_id = hot_node.node_id;
        let feasible_neighbors = &hot_node.feasible_neighbors_of_different_tree;

        for &neighbor_id in feasible_neighbors.iter() {
            debug_assert!(self.node(node_id).tree_id != Self::PRUNED_TREE_ID);
            debug_assert!(self.node(neighbor_id).tree_id != Self::PRUNED_TREE_ID);

            if self.node(node_id).tree_id == self.node(neighbor_id).tree_id {
                // Skip if they are in same tree, which is possible after a reconnection
                continue;
            }

            self.reconnect_nodes(node_id, neighbor_id);
        }
    }

    /// Reconnects two nodes of different trees.
    /// If any node is the MAIN_TREE, it is guaranteed to be the parent of the other node.
    /// When nodes are reconnected, the child node will be added to set of optimization start nodes.
    fn reconnect_nodes(&mut self, node_a_id: usize, node_b_id: usize) {
        let node_a_tree_id = self.node(node_a_id).tree_id;
        let node_b_tree_id = self.node(node_b_id).tree_id;

        debug_assert!(node_a_tree_id != Self::PRUNED_TREE_ID);
        debug_assert!(node_b_tree_id != Self::PRUNED_TREE_ID);
        debug_assert!(self.debug_check_tree_consistency());
        assert!(node_a_tree_id != node_b_tree_id);

        if node_a_tree_id < node_b_tree_id {
            // The node with the lower tree ID is the parent
            // This guarantees that the MAIN_TREE is always the parent of the other node
            // This also means new sampled nodes will always be added to an existing tree

            // Make node_b the root of its own subtree
            self.make_root_of_subtree(node_b_id);
            debug_assert!(self.debug_check_tree_consistency());

            // Set node_a as the parent of node_b
            self.set_node_parent(node_b_id, Some(node_a_id));

            // check the edge for validity
            debug_assert!(self.cpr_validity_check_edge(
                self.node(node_a_id).state(),
                self.node(node_b_id).state()
            ));

            let edge_cost = self
                .node(node_a_id)
                .state()
                .euclidean_distance(self.node(node_b_id).state());

            let cumulative_cost = self.node(node_a_id).cumulative_cost + edge_cost;

            self.node_mut(node_b_id).tree_id = node_a_tree_id;
            self.node_mut(node_b_id).edge_cost = edge_cost;
            self.node_mut(node_b_id).cumulative_cost = cumulative_cost;

            self.propagate_info_to_children(node_b_id);
            debug_assert!(self.debug_check_tree_consistency());
            self.add_optimization_start_node(node_b_id);
        } else {
            // Make node_a the root of its own subtree
            self.make_root_of_subtree(node_a_id);

            // Set node_b as the parent of node_a
            self.set_node_parent(node_a_id, Some(node_b_id));

            // check the motion for validity
            debug_assert!(self.cpr_validity_check_edge(
                self.node(node_b_id).state(),
                self.node(node_a_id).state()
            ));

            let edge_cost = self
                .node(node_b_id)
                .state()
                .euclidean_distance(self.node(node_a_id).state());

            let cumulative_cost = self.node(node_b_id).cumulative_cost + edge_cost;

            self.node_mut(node_a_id).cumulative_cost = cumulative_cost;
            self.node_mut(node_a_id).tree_id = node_b_tree_id;
            self.node_mut(node_a_id).edge_cost = edge_cost;
            self.node_mut(node_a_id).cumulative_cost = cumulative_cost;
            self.propagate_info_to_children(node_a_id);
            debug_assert!(self.debug_check_tree_consistency());
            self.add_optimization_start_node(node_a_id);
        }

        debug_assert!(self.node(node_a_id).tree_id == self.node(node_b_id).tree_id);
    }

    /// Makes a node the root of its own subtree (all other nodes become its successors).
    /// Can only be used on disjoint subtrees, not on the main tree.
    fn make_root_of_subtree(&mut self, node_id: usize) {
        debug_assert!(self.node(node_id).tree_id != Self::PRUNED_TREE_ID);
        debug_assert!(self.node(node_id).tree_id != Self::MAIN_TREE_ID);

        let mut previous_node_id = None;
        let mut current_node_id = Some(node_id);

        while current_node_id.is_some() {
            let current: usize = current_node_id.unwrap();
            let old_parent_id = self.node(current).parent;

            // Set the parent of the current node to the previous node
            self.set_node_parent(current, previous_node_id);
            if let Some(new_parent) = previous_node_id {
                // Update the edge_cost
                let edge_cost = self
                    .node(new_parent)
                    .state()
                    .euclidean_distance(self.node(current).state());

                debug_assert!(self.node(new_parent).tree_id == self.node(current).tree_id);
                self.node_mut(current).edge_cost = edge_cost;
            }

            // The old parent of the current node becomes the new current node
            previous_node_id = current_node_id;
            current_node_id = old_parent_id;

            // We don't need to update cost as it should be infinite for non-main tree nodes
            debug_assert!(self.node(previous_node_id.unwrap()).cumulative_cost == F::infinity());
        }
    }

    /// In sampling tree repair, we sample a new node and try to connect it to all of its neighbors
    fn sampling_tree_repair_iteration(&mut self) {
        loop {
            let new_node_state = self.sample_free();
            if self.add_new_node_at_state_and_connect(new_node_state) {
                // If the node was added successfully, we can stop sampling
                break;
            }
        }
    }

    /// Samples a new state that is free (considering all static obstacles and dynamic obstacles only within LRZ)
    fn sample_free(&mut self) -> RealVectorState<F, N> {
        let mut state = self.sampling_distribution.sample();
        while !self.cpr_validity_check_state(&state)
            || !self.static_validity_checker.is_state_valid(&state)
        {
            state = self.sampling_distribution.sample();
        }
        state
    }

    /// Adds a new node at the specified state and connects it to an existing tree.
    /// Returns true if the node was added successfully, false otherwise.
    fn add_new_node_at_state_and_connect(&mut self, state: RealVectorState<F, N>) -> bool {
        self.tree_id_counter += 1;
        let new_node = Node::new(
            state.clone(),
            None,
            F::infinity(),
            F::infinity(),
            self.tree_id_counter,
        );

        // get feasible neighbors of different trees
        let neighbors = self.get_feasible_neighbors_of_different_tree(&new_node);
        if neighbors.is_empty() {
            // If there are no feasible neighbors, we can skip adding the node
            return false;
        }

        // add the node to the node list and nearest neighbors data structure
        let new_node_id = self.add_node(new_node);

        let hot_node = HotNode {
            node_id: new_node_id,
            feasible_neighbors_of_different_tree: neighbors.clone(),
            utility: F::zero(), // Utility doesn't matter here
        };

        // Connect the new node to its neighbors
        self.reconnect_best_hot_node(&hot_node);
        true
    }

    /// Adds a node to the tree and the nearest neighbors data structure.
    fn add_node(&mut self, node: Node<F, N>) -> usize {
        let index = self.nodes.len();
        self.nearest_neighbors.add(node.state().clone(), index);
        self.nodes.push(node);
        index
    }

    fn add_optimization_start_node(&mut self, node_id: usize) {
        if self.node(node_id).tree_id != Self::MAIN_TREE_ID {
            return;
        }

        // Check if a parent of the node is already in the optimization start nodes
        let mut parent_id = node_id;
        while let Some(parent) = self.node(parent_id).parent {
            if self.optimization_start_nodes.contains(&parent) {
                return;
            }
            parent_id = parent;
        }

        self.optimization_start_nodes.push(node_id);
    }

    fn tree_optimization(&mut self) {
        // Take ownership of the optimization_start_nodes, leaving it empty
        let optimization_start_nodes = std::mem::take(&mut self.optimization_start_nodes);

        // Sort the nodes by their cumulative cost
        let mut sorted_nodes = optimization_start_nodes;
        sorted_nodes.sort_by(|&a, &b| {
            self.node(a)
                .cumulative_cost
                .partial_cmp(&self.node(b).cumulative_cost)
                .unwrap()
        });

        // Iterate over the sorted nodes and perform the rewiring
        for node_id in sorted_nodes {
            self.rewire_cascade(node_id);
        }

        // `self.optimization_start_nodes` remains empty after this method
        debug_assert!(self.optimization_start_nodes.is_empty());
    }

    fn rewire_cascade(&mut self, node_id: usize) {
        // Although this is not the most technically correct way to implement a rewiring cascade,
        // this is the same way as it was done in the original SMART-2D implementation.
        // Original implementation: https://github.com/ZongyuanShen/SMART/blob/main/SMART/SMART.cpp#L1225
        // This uses a depth-first search, but a queue based approach would be more correct.
        self.rewire_to_best_parent(node_id);
        let children = self.node(node_id).children.clone();
        for child_id in children {
            self.rewire_cascade(child_id);
        }
    }

    /// Rewires the node to the best parent within the hot nod rewiring radius.
    fn rewire_to_best_parent(&mut self, node_id: usize) {
        let neighbors = self.nearest_neighbors.within_radius(
            self.node(node_id).state(),
            self.hot_node_neighborhood_radius,
        );

        let mut best_parent = self.node(node_id).parent;
        let mut best_cost = self.node(node_id).cumulative_cost;
        let mut best_edge_cost = self.node(node_id).edge_cost;

        for &neighbor_id in neighbors.iter() {
            if Some(neighbor_id) == best_parent || neighbor_id == node_id {
                continue;
            }
            let neighbor = self.node(neighbor_id);

            // Check if the neighbor is in the same tree or if it is prunedl

            let edge_cost = self
                .node(neighbor_id)
                .state()
                .euclidean_distance(self.node(node_id).state());
            let cost = neighbor.cumulative_cost + edge_cost;
            if cost < best_cost
                && self.cpr_validity_check_edge(neighbor.state(), self.node(node_id).state())
                && self
                    .static_validity_checker
                    .is_edge_valid(neighbor.state(), self.node(node_id).state())
            {
                best_parent = Some(neighbor_id);
                best_cost = cost;
                best_edge_cost = edge_cost;
            }
        }

        self.set_node_parent(node_id, best_parent);
        self.node_mut(node_id).cumulative_cost = best_cost;
        self.node_mut(node_id).edge_cost = best_edge_cost;
        self.propagate_info_to_children(node_id);
    }

    fn form_single_tree(&mut self) {
        // Take ownership of the pruned nodes, leaving it empty
        let pruned = std::mem::take(&mut self.pruned_nodes);

        // Iterate over the pruned nodes and add them back to the tree
        for &(node_id, old_parent_id) in pruned.iter() {
            // The pruned nodes should have no parent and a pruned tree id
            assert!(self.node(node_id).tree_id == Self::PRUNED_TREE_ID);
            assert!(self.node(node_id).parent.is_none());

            // Set the node back to its old parent
            self.set_node_parent(node_id, old_parent_id);

            if old_parent_id.is_none() {
                // The node was the root of the main tree
                assert!(node_id == 0);
                assert!(self.node(node_id).parent.is_none());

                self.node_mut(node_id).tree_id = Self::MAIN_TREE_ID;
                self.node_mut(node_id).cumulative_cost = F::zero();
                self.node_mut(node_id).edge_cost = F::zero();
            } else {
                let old_parent_id = old_parent_id.unwrap();

                if self.node(old_parent_id).tree_id == Self::PRUNED_TREE_ID {
                    // the parent is also pruned
                    // this node will be handled by propogation from the parent
                    assert!(pruned.iter().any(|&(n, _)| n == old_parent_id));
                    continue;
                }

                let edge_cost = self
                    .node(old_parent_id)
                    .state()
                    .euclidean_distance(self.node(node_id).state());

                // We need to fix the cumulative cost
                self.node_mut(node_id).cumulative_cost =
                    self.node(old_parent_id).cumulative_cost + edge_cost;
                self.node_mut(node_id).edge_cost = edge_cost;

                // The tree id of the node should be the same as its parent
                self.node_mut(node_id).tree_id = self.node(old_parent_id).tree_id;
            }

            // Propagate the information to the children (if any)
            self.propagate_info_to_children(node_id);
        }

        assert!(self.debug_check_tree_consistency());
        assert!(self.debug_check_costs());

        // Take ownership of the disjoint tree roots, leaving it empty
        let disjoint_tree_roots = std::mem::take(&mut self.disjoint_tree_roots);

        // Iterate over the disjoint tree roots and reconnect them to the main tree
        for &(node_id, old_parent_id) in disjoint_tree_roots.iter() {
            // A disjoint tree root should always have an old parent
            debug_assert!(old_parent_id.is_some());
            let old_parent_id = old_parent_id.unwrap();
            debug_assert!(self.node(old_parent_id).tree_id != Self::PRUNED_TREE_ID);

            // If the node is already in the main tree, skip it
            if self.node(node_id).tree_id == Self::MAIN_TREE_ID {
                continue;
            }

            // If this already has the same tree id as the old parent, don't reconnect
            if self.node(old_parent_id).tree_id == self.node(node_id).tree_id {
                continue;
            }

            // Make the node the root of its own subtree
            self.make_root_of_subtree(node_id);

            // Set the node back to its old parent
            self.set_node_parent(node_id, Some(old_parent_id));

            // Set the edge and cumulative cost
            let edge_cost = self
                .node(old_parent_id)
                .state()
                .euclidean_distance(self.node(node_id).state());
            self.node_mut(node_id).edge_cost = edge_cost;
            self.node_mut(node_id).cumulative_cost =
                self.node(old_parent_id).cumulative_cost + self.node(node_id).edge_cost;

            // The tree id of the node should be the same as its parent
            self.node_mut(node_id).tree_id = self.node(old_parent_id).tree_id;

            // Propagate the information to the children
            self.propagate_info_to_children(node_id);
        }

        // pruned nodes and disjoint tree roots should be empty after this method
        debug_assert!(self.pruned_nodes.is_empty());
        debug_assert!(self.disjoint_tree_roots.is_empty());
    }

    /// Check if a state is a valid, considering the CPR obstacles.
    fn cpr_validity_check_state(&self, state: &RealVectorState<F, N>) -> bool {
        if state.euclidean_distance(&self.current_state) > self.max_cpr_radius {
            return true; // The state is outside the CPR, so we don't need to check for collisions
        }

        for obs in &self.cpr_obstacles {
            if obs.contains(state) {
                return false; // The state is in collision with a CPR obstacle
            }
        }

        true // The state is valid (not in collision with any CPR obstacle)
    }

    /// Check if an edge is valid, considering the CPR obstacles.
    fn cpr_validity_check_edge(
        &self,
        start: &RealVectorState<F, N>,
        end: &RealVectorState<F, N>,
    ) -> bool {
        let min_dist_from_robot = self
            .current_state
            .euclidean_distance(end)
            .min(self.current_state.euclidean_distance(start));

        if min_dist_from_robot > self.max_cpr_radius + self.hot_node_neighborhood_radius {
            return true; // The edge is outside the CPR, so we don't need to check for collisions
        }

        for obs in &self.cpr_obstacles {
            if obs.intersects_edge(start, end) {
                return false; // The edge is in collision with a CPR obstacle
            }
        }

        true // The edge is valid (not in collision with any CPR obstacle)
    }

    /// Sets the parent of a node, including by updating the new and old parent's child list.
    fn set_node_parent(&mut self, node_id: usize, parent: Option<usize>) {
        // Remove the node from the old parent's child list
        let old_parent_id = self.node(node_id).parent;
        if let Some(old_parent_id) = old_parent_id {
            self.node_mut(old_parent_id).remove_child(node_id);
        }

        // Set the new parent and add the node to the parent's child list
        self.node_mut(node_id).parent = parent;
        if let Some(parent_id) = parent {
            self.node_mut(parent_id).add_child(node_id);
        }
    }

    fn node(&self, node_id: usize) -> &Node<F, N> {
        &self.nodes[node_id]
    }

    fn node_mut(&mut self, node_id: usize) -> &mut Node<F, N> {
        &mut self.nodes[node_id]
    }

    // Debugging Assertion Functions
    // -----------------------------

    /// Checks the consistency of the tree.
    /// This checks the following:
    /// 1. Parent consistency: The parent of a node should have the node as a child.
    /// 2. Child consistency: The children of a node should have the node as a parent.
    /// 3. Cycle detection: There should be no cycles in the tree.
    /// 4. Tree ID consistency: The tree ID of a node should be the same as its parent.
    fn debug_check_tree_consistency(&self) -> bool {
        for (index, node) in self.get_nodes().iter().enumerate() {
            // Check every node for parent consistency.
            // This means that the parent of a node should have the node as a child.
            if let Some(parent_index) = node.parent {
                let parent = self.node(parent_index);
                if !parent.children.contains(&index) {
                    println!(
                        "Parent inconsistency! Node {} has parent {} but parent does not have child {}",
                        index, parent_index, index
                    );
                    return false;
                }
                if parent.tree_id != node.tree_id {
                    println!(
                        "Tree ID inconsistency! Node {} has tree id {} and parent {} with tree ID {}",
                        index, node.tree_id, parent_index, parent.tree_id
                    );
                    return false;
                }
            }

            // Check every node for child consistency.
            // This means that the children of a node should have the node as a parent.
            for &child_index in &node.children {
                let child = self.node(child_index);
                if child.parent != Some(index) {
                    println!(
                        "Child inconsistency! Node {} has child {} but child has parent {:?}",
                        index, child_index, child.parent
                    );
                    return false;
                }
            }

            // Check every node for cycles
            let mut seen = std::collections::HashSet::new();
            let mut current_node_id = index;
            while let Some(parent_id) = self.node(current_node_id).parent {
                if current_node_id < index {
                    // We know we already did a cycle check starting from this node
                    break;
                }
                if !seen.insert(current_node_id) {
                    println!(
                        "Cycle detected in tree! Node {} is visited twice",
                        current_node_id
                    );
                    return false;
                }
                current_node_id = parent_id;
            }
        }

        true
    }

    /// Checks that nodes were pruned correctly.
    /// This checks the following:
    /// 1. At least one node was pruned, or at least one disjoint tree root was created.   
    /// 2. All pruned nodes should have a tree ID of PRUNED_TREE_ID.
    /// 3. All pruned nodes are within the max pruning radius.
    /// 4. All pruned nodes are not valid.
    /// 5. All non-pruned nodes within the LRZ should be safe.
    /// 6. All disjoint tree roots should have unique tree IDs.
    /// 7. All disjoint tree roots should have a tree ID greater than MAIN_TREE_ID.
    /// 8. All disjoint tree roots should have no parent.
    fn debug_check_pruning(&self) -> bool {
        // at least one node is pruned or disjoint tree root is added
        debug_assert!(self.pruned_nodes.len() > 0 || self.disjoint_tree_roots.len() > 0);

        // all pruned nodes should have tree id of PRUNED_TREE_ID
        for &(node_id, _) in self.pruned_nodes.iter() {
            debug_assert!(self.node(node_id).tree_id == Self::PRUNED_TREE_ID);
            debug_assert!(!self.cpr_validity_check_state(self.node(node_id).state()));
        }

        // check all non-pruned nodes are safe
        if !self.debug_check_non_pruned_nodes_are_safe() {
            println!("Pruned nodes list: {:?}", self.pruned_nodes);
            return false;
        }

        let mut disjoint_tree_roots = std::collections::HashSet::new();
        for (node_id, _) in self.disjoint_tree_roots.iter() {
            // all disjoint tree ids are unique
            debug_assert!(disjoint_tree_roots.insert(self.node(*node_id).tree_id));
            // all disjoint tree roots are greater than main tree id
            debug_assert!(self.node(*node_id).tree_id > Self::MAIN_TREE_ID);
            // all disjoint tree roots have no parent
            debug_assert!(self.node(*node_id).parent.is_none());
        }

        true
    }

    /// Checks that all nodes that are not pruned are safe.
    fn debug_check_non_pruned_nodes_are_safe(&self) -> bool {
        // check all non-pruned nodes are safe
        for node_id in 0..self.get_nodes().len() {
            if self.node(node_id).tree_id == Self::PRUNED_TREE_ID {
                continue;
            }
            if !self.cpr_validity_check_state(self.node(node_id).state()) {
                println!(
                    "Node {} is not safe! State in collision with dynamic CPR obstacle.",
                    node_id
                );
                println!("Node {} state: {:?}", node_id, self.node(node_id).state());
                println!("Obstacle list: {:?}", self.cpr_obstacles);
                println!("CPR max radius: {:?}", self.max_cpr_radius);
                println!("Current state: {:?}", self.current_state);
                println!("LRZ radius: {:?}", self.lrz_radius);
                return false;
            }
            if !self
                .static_validity_checker
                .is_state_valid(self.node(node_id).state())
            {
                println!(
                    "Node {} is not safe! State in collision with static obstacle.",
                    node_id
                );
                return false;
            }

            if let Some(parent) = self.node(node_id).parent {
                if !self
                    .cpr_validity_check_edge(self.node(parent).state(), self.node(node_id).state())
                {
                    println!(
                        "Node {} is not safe! Edge from parent {} is in collision with dynamic CPR obstacle.",
                        node_id, parent
                    );
                    if self.disjoint_tree_roots.contains(&(node_id, Some(parent))) {
                        println!("Node {} is a disjoint tree root.", node_id);
                    }
                    println!("Node {} state: {:?}", node_id, self.node(node_id).state());
                    println!("Parent {} state: {:?}", parent, self.node(parent).state());
                    println!("Obstacle list: {:?}", self.cpr_obstacles);
                    println!("CPR max radius: {:?}", self.max_cpr_radius);
                    println!("Current state: {:?}", self.current_state);
                    println!("LRZ radius: {:?}", self.lrz_radius);
                    return false;
                }
                if !self
                    .static_validity_checker
                    .is_edge_valid(self.node(parent).state(), self.node(node_id).state())
                {
                    println!(
                        "Node {} is not safe! Edge from parent {} is in collision with static obstacle.",
                        node_id, parent
                    );
                    return false;
                }
            }
        }
        true
    }

    /// Checks that every node has a tree_id of MAIN_TREE_ID.
    /// If not, prints a summary of how many nodes are in each other tree.
    fn debug_check_all_nodes_in_main_tree(&self) -> bool {
        let mut counts: HashMap<i32, usize> = HashMap::new();

        for node in self.get_nodes().iter() {
            if node.tree_id != Self::MAIN_TREE_ID {
                *counts.entry(node.tree_id).or_insert(0) += 1;
            }
        }

        if counts.is_empty() {
            // All nodes are in the main tree
            true
        } else {
            println!(
                "Debug check failed: found {} nodes outside main tree (id {})",
                counts.values().sum::<usize>(),
                Self::MAIN_TREE_ID
            );
            for (tree_id, &count) in &counts {
                println!(" - {} node(s) in tree_id {}", count, tree_id);
            }
            false
        }
    }

    /// Checks that all cumulative node costs match edge costs for nodes in the main tree.
    fn debug_check_costs(&self) -> bool {
        for node_id in 0..self.get_nodes().len() {
            let node = self.node(node_id);

            if node.tree_id != Self::MAIN_TREE_ID {
                continue;
            }

            if node.parent.is_none() {
                continue; // The root node has no parent, so we skip it
            }

            let parent_id = node.parent.unwrap();
            let parent = self.node(parent_id);

            let expected_edge_cost = parent.cumulative_cost - node.cumulative_cost;
            if (expected_edge_cost - node.edge_cost).abs() < F::from(0.0001).unwrap() {
                println!(
                    "Cost mismatch for node {}: expected edge cost {}, but got {}",
                    node_id,
                    expected_edge_cost.to_f32().unwrap(),
                    node.edge_cost.to_f32().unwrap()
                );
                return false;
            }
        }
        true
    }
}
