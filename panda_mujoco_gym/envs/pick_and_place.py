import os
from panda_mujoco_gym.envs.panda_env import FrankaEnv

MODEL_XML_PATH = os.path.join(
    os.path.dirname(__file__), "../assets/", "pick_and_place.xml"
)


class FrankaPickAndPlaceEnv(FrankaEnv):
    def __init__(self, reward_type="dense", **kwargs):
        super().__init__(
            model_path=MODEL_XML_PATH,
            n_substeps=25,
            reward_type=reward_type,
            block_gripper=False,
            distance_threshold=0.05,
            goal_xy_range=0.3,
            obj_xy_range=0.3,
            goal_x_offset=0.0,
            goal_z_range=0.2,
            # --- Universal reward params: enable lift + place gating for pick-and-place ---
            w_obj_goal=2.0,  # Object-to-goal shaping weight
            w_ee_obj=1.0,  # End-effector-to-object (reach) shaping weight
            w_lift=5.0,  # Lift shaping weight (key for pick)
            w_gripper=0.2,  # Optional gripper shaping weight (set to 0.0 to disable)
            place_activation_height=0.05,  # Enable object-to-goal shaping only after lifting above this height (prevents "pushing" loophole)
            success_activation_height=0,  # Success also requires lifting above this height (set to 0.0 to disable)
            # --- Regularization terms: keep small/disabled initially ---
            w_action=-1e-4,  # Action magnitude penalty weight (small)
            w_action_change=-1e-4,  # Action change penalty weight (small)
            w_smooth=0.0,  # Smoothness penalty weight (disabled by default)
            terminal_bonus=0.0,  # Extra terminal bonus on success (disabled by default)
            **kwargs,
        )
