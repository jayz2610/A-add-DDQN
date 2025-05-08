import numpy as np

from src.State import State
from src.base.GridActions import GridActions
from src.base.GridRewards import GridRewards, GridRewardParams
# from src.Map.Map import Map


class RewardParams(GridRewardParams):
    def __init__(self):
        super().__init__()


class Rewards(GridRewards):

    def __init__(self, reward_params: RewardParams, stats):
        # Pass specific params to base class init (requires base init to accept params)
        super().__init__(stats, params=reward_params)
        self.reset()

    def calculate_reward(self, state: State, action: GridActions, next_state: State, agent_id: int) -> float:
        """
        Calculates the reward for a given agent transition (state, action -> next_state).
        Includes motion penalties, data reward, landing reward, and failed landing penalty.
        """
        step_reward = 0.0 # Default return value
        original_state_active = state.active_agent
        original_next_state_active = next_state.active_agent

        try:
            # Set active agent context for state properties/methods
            if state is None or next_state is None: return 0.0
            state.active_agent = agent_id
            next_state.active_agent = agent_id

            # --- a. Calculate base motion rewards/penalties ---
            motion_reward = super().calculate_motion_rewards(state, action, next_state)
            step_reward += motion_reward

            # --- b. Calculate Data Collection Reward ---
            if hasattr(state, 'collected') and state.collected is not None and \
               hasattr(next_state, 'collected') and next_state.collected is not None:
                data_gain = np.sum(next_state.collected) - np.sum(state.collected)
                if data_gain > 1e-9:
                    data_reward = self.params.data_multiplier * data_gain
                    step_reward += data_reward

        except Exception as e:
            print(f"!!!!!!!! ERROR inside calculate_reward for Agent {agent_id} !!!!!!!!")
            print(f"Error message: {e}")
            print(f"Action: {action.name if isinstance(action, GridActions) else action}") # Print action name safely
            # Print states safely
            try:
                 print(f"State Pos: {state.position if state else 'None'}, Budget: {state.movement_budget if state else 'None'}, Landed: {state.landed if state else 'None'}, Terminal: {state.terminal if state else 'None'}")
                 print(f"Next State Pos: {next_state.position if next_state else 'None'}, Budget: {next_state.movement_budget if next_state else 'None'}, Landed: {next_state.landed if next_state else 'None'}, Terminal: {next_state.terminal if next_state else 'None'}")
            except Exception as e_print:
                 print(f"Error printing state details: {e_print}")
            import traceback
            traceback.print_exc()
            step_reward = 0.0 # Default to 0 reward if calculation failed

        finally:
            # Restore original active agents
            if state is not None: state.active_agent = original_state_active
            if next_state is not None: next_state.active_agent = original_next_state_active

            # Return the final step reward
            if step_reward is None: return 0.0
            return float(step_reward)

    def reset(self):
        super().reset()