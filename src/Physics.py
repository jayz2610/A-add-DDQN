import copy

import numpy as np

from src.Channel import ChannelParams, Channel
from src.State import State
from src.ModelStats import ModelStats
from src.base.GridActions import GridActions
from src.base.GridPhysics import GridPhysics


class PhysicsParams:
    def __init__(self):
        # Initialize counters directly, do not rely on GridPhysics.__init__ if it only does this

        self.channel_params = ChannelParams()
        self.comm_steps = 4


class Physics(GridPhysics):

    def __init__(self, params: PhysicsParams, grid_width: int,grid_height: int,stats: ModelStats):

        super().__init__()

        self.channel = Channel(params.channel_params)
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.params = params
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.register_functions(stats)

    def register_functions(self, stats: ModelStats):
        stats.set_evaluation_value_callback(self.get_cral)
        stats.add_log_data_callback('cral', self.get_cral)
        stats.add_log_data_callback('cr', self.get_collection_ratio)
        stats.add_log_data_callback('successful_landing', self.has_landed)
        stats.add_log_data_callback('boundary_counter', self.get_boundary_counter)
        stats.add_log_data_callback('landing_attempts', self.get_landing_attempts)
        stats.add_log_data_callback('movement_ratio', self.get_movement_ratio)


    def reset(self, state: State):
        # Reset counters directly
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = state
        self.channel.reset(self.state.shape[0])

    def step(self, action: GridActions, state_to_modify: State):
        agent_id = state_to_modify.active_agent  # Get the agent ID
        original_active_agent = state_to_modify.active_agent
        state_to_modify.active_agent = agent_id

        # --- ADDED: Prevent repeated LAND at the same failed location ---
        current_pos_tuple_for_check = tuple(state_to_modify.position)  # Get current position as tuple
        last_failed_pos = state_to_modify.last_failed_land_pos[agent_id]  # Get last failed pos

        if action == GridActions.LAND and \
                last_failed_pos is not None and \
                current_pos_tuple_for_check == last_failed_pos:
            # If trying to LAND again at the exact location of the last failure
            print(
                f"[Physics Safety] Agent {agent_id} attempting LAND again at recently failed position {current_pos_tuple_for_check}. Overriding to HOVER.")
            action = GridActions.HOVER  # Override the action
        # --- END ADDED ---

        # 1. Check if agent is already terminal
        if state_to_modify.terminal:  # Checks property using active_agent
            state_to_modify.active_agent = original_active_agent  # Restore
            return  # Exit early, do not modify state further

        # 3. Movement Logic (Replaces call to self.movement_step(action))
        #    This assumes movement_step in GridPhysics modified self.state
        old_position = state_to_modify.position  # Gets position list [x, y]
        x, y = old_position[0], old_position[1]

        is_land_action = False
        if action == GridActions.NORTH:
            y -= 1
        elif action == GridActions.SOUTH:
            y += 1
        elif action == GridActions.WEST:
            x -= 1
        elif action == GridActions.EAST:
            x += 1
        elif action == GridActions.LAND:
            is_land_action = True
            self.landing_attempts += 1  # Increment global counter

        intended_position = [x, y]  # Intended coordinates

        # 3. Validate intended position / Handle LAND action
        final_position = list(old_position)  # Default: stay put if move invalid
        landed_successfully = False

        if is_land_action:
            # --- ADD PRINTS for LAND action ---
            print(f"[PHYSICS LAND] Agent {agent_id} attempts LAND at {old_position}.")
            is_in_zone = state_to_modify.is_in_landing_zone()  # Call the (now verbose) state method
            print(f"[PHYSICS LAND] Agent {agent_id}: state.is_in_landing_zone() returned: {is_in_zone}")
            if is_in_zone:
                landed_successfully = True
                final_position = list(old_position)  # Stay in the landing spot
                print(f"[PHYSICS LAND] Agent {agent_id}: Landing successful! Calling set_landed(True).")
                state_to_modify.set_landed(True)  # Set landed status for agent_id
            else:
                print(
                    f"[PHYSICS LAND] Agent {agent_id}: Landing failed (not in zone). Position remains {final_position}.")
                # No call to set_landed(True)
        else:
            # --- Movement Validation ---
            ix, iy = intended_position[0], intended_position[1]
            if 0 <= ix < self.grid_width and 0 <= iy < self.grid_height:
                current_pos_backup = list(state_to_modify.position)
                state_to_modify.set_position(intended_position)  # Temporarily set on state_to_modify
                if not state_to_modify.is_in_no_fly_zone():  # Check validity
                    final_position = intended_position  # Valid move
                else:  # Invalid move (NFZ/Occupied)
                    print(
                        f"[Physics Validation] Agent {agent_id} intended move to {intended_position} denied (NFZ/Occupied).")
                    self.boundary_counter += 1
                state_to_modify.set_position(current_pos_backup)  # Restore actual pos
            else:  # Invalid move (Out of Bounds)
                print(
                    f"[Physics Validation] Agent {agent_id} intended move to {intended_position} denied (Out of bounds).")
                self.boundary_counter += 1
            # --- End Movement Validation ---

        # 4. Set the final validated position DIRECTLY on state_to_modify
        state_to_modify.set_position(final_position)

        # 5. Decrement budget DIRECTLY on state_to_modify
        state_to_modify.decrement_movement_budget()

        # 6. Update terminal status DIRECTLY on state_to_modify
        is_now_terminal = state_to_modify.landed or (state_to_modify.movement_budget <= 0)
        state_to_modify.set_terminal(is_now_terminal)

        # 7. Communication Step (only if not terminal)
        if not state_to_modify.terminal:
            # Pass the state_to_modify object itself to comm_step
            self.comm_step(state_to_modify, agent_id, tuple(final_position), tuple(old_position))

        # --- Restore original active agent ---
        state_to_modify.active_agent = original_active_agent

    def comm_step(self, state_to_modify: State, agent_id: int, current_position: tuple, old_position: tuple):
        # Calculates comms based on movement from old_position to current_position
        # Modifies state_to_modify directly

        # Use current_position and old_position to get the path segment
        pos_array_current = np.array(current_position)
        pos_array_old = np.array(old_position)
        if np.array_equal(pos_array_current, pos_array_old):
            # Agent is hovering at current_position
            num_sub_steps = self.params.comm_steps
            # Create a list of the *same* position repeated for each sub-step
            positions = [pos_array_current] * num_sub_steps
            # print(f"[Comm Debug] Agent {agent_id} Hovering at {current_position}. Simulating {num_sub_steps} comm steps.") # Optional Debug
        elif self.params.comm_steps > 0:
            # Agent moved, use linspace as before
            positions = np.linspace(pos_array_current, pos_array_old, num=self.params.comm_steps, endpoint=False)
            positions = positions[::-1]  # Reverse to simulate moving from old to current
        else:  # No comm steps defined
            positions = []

        indices = []
        # IMPORTANT: device_list needs to be part of the state passed around
        device_list = state_to_modify.device_list
        if device_list is None:
            print("Error: device_list not found in state during comm_step!")
            return

        for pos_np in positions:  # pos_np is a numpy array [x, y]
            pos_tuple = tuple(pos_np)  # Convert for device method if needed
            # Add clipping here just in case linspace produces edge values slightly off
            map_h, map_w = state_to_modify.no_fly_zone.shape[:2]
            clipped_pos_tuple = (
                np.clip(pos_tuple[0], 0, map_w - 1),
                np.clip(pos_tuple[1], 0, map_h - 1)
            )
            try:
                # Pass channel object from self.channel
                data_rate, idx = device_list.get_best_data_rate(clipped_pos_tuple, self.channel)
                # --- ADD PRINT ---
                if tuple(clipped_pos_tuple) == (2, 4):  # Check specifically for the problematic location
                    print(f"[Comm Debug] Hovering at {clipped_pos_tuple}. Best Rate={data_rate:.4f}, Device Idx={idx}")
                # --- END PRINT ---
                if idx != -1 and data_rate > 1e-6:  # Only collect if rate is meaningfully positive
                    device_list.collect_data(data_rate, idx)
                indices.append(idx)
            except Exception as e:
                print(f"Error during get_best_data_rate/collect_data: {e} at pos {clipped_pos_tuple}")

        # Update maps in the state object that was passed in
        state_to_modify.collected = device_list.get_collected_map(state_to_modify.shape)
        state_to_modify.device_map = device_list.get_data_map(state_to_modify.shape)

        if indices:
            idx = max(set(indices), key=indices.count)
            # Set active agent before calling state method if it relies on it
            original_active = state_to_modify.active_agent
            state_to_modify.active_agent = agent_id
            state_to_modify.set_device_com(idx)
            state_to_modify.active_agent = original_active  # Restore

    def get_example_action(self):
        return GridActions.HOVER

    def is_in_landing_zone(self):
        return self.state.is_in_landing_zone()

    def get_collection_ratio(self):
        return self.state.get_collection_ratio()

    def get_movement_budget_used(self):
        return sum(self.state.initial_movement_budgets) - sum(self.state.movement_budgets)

    def get_max_rate(self):
        return self.channel.get_max_rate()

    def get_average_data_rate(self):
        return self.state.get_collected_data() / self.get_movement_budget_used()

    def get_cral(self):
        return self.get_collection_ratio() * self.state.all_landed

    def get_boundary_counter(self):
        return self.boundary_counter

    def get_landing_attempts(self):
        return self.landing_attempts

    def get_movement_ratio(self):
        return float(self.get_movement_budget_used()) / float(sum(self.state.initial_movement_budgets))

    def has_landed(self):
        return self.state.all_landed
