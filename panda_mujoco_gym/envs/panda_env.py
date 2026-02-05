import mujoco
import numpy as np
from gymnasium.core import ObsType
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations
from typing import Optional, Any, SupportsFloat

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 135.0,
    "elevation": -20.0,
    "lookat": np.array([0.0, 0.5, 0.0]),
}


class FrankaEnv(MujocoRobotEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        model_path: str = None,
        n_substeps: int = 50,
        reward_type: str = "dense",
        block_gripper: bool = False,
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        obj_xy_range: float = 0.3,
        goal_x_offset: float = 0.4,
        goal_z_range: float = 0.2,
        max_episode_steps: int = 200,
        # 仅替换 dense reward 所需的可调权重（来自第二段）
        w_progress: float = 2.0,
        w_distance: float = -1.0,
        w_action: float = -0.001,
        w_action_change: float = -0.005,
        w_smooth: float = -0.002,
        w_height: float = 0.5,
        w_gripper: float = 0.3,
        success_reward: float = 10.0,
        terminal_bonus: float = 20.0,
        progress_clip: float = 0.1,
        progress_horizon: int = 3,
        vel_ema_tau: float = 0.95,
        gripper_target_width: float = 0.04,
        pos_ctrl_scale: float = 0.05,
        **kwargs,
    ):
        self.block_gripper = block_gripper
        self.model_path = model_path

        action_size = 3
        action_size += 0 if self.block_gripper else 1

        self.reward_type = reward_type

        # 第一段的中立位
        self.neutral_joint_values = np.array(
            [0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00]
        )

        # Episode tracking（第一段）
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.trajectory_points = []
        self.previous_ee_position = None
        self.initial_object_position = None
        self.total_path_length = 0.0

        # For reward computation（第一段缓存 + 第二段需要的额外缓存）
        self.previous_distance = None
        self.initial_distance = None
        self.best_distance = None
        self.gamma = 0.99
        # 第二段 dense 奖励所需
        self.w_progress = float(w_progress)
        self.w_distance = float(w_distance)
        self.w_action = float(w_action)
        self.w_action_change = float(w_action_change)
        self.w_smooth = float(w_smooth)
        self.w_height = float(w_height)
        self.w_gripper = float(w_gripper)
        self.success_reward = float(success_reward)
        self.terminal_bonus = float(terminal_bonus)
        self.progress_clip = float(progress_clip)
        self.progress_horizon = int(progress_horizon)
        self.vel_ema_tau = float(vel_ema_tau)
        self.gripper_target_width = float(gripper_target_width)
        self.pos_ctrl_scale = float(pos_ctrl_scale)

        # EMA 与上一次动作（第二段逻辑）
        self.vel_ema = np.zeros(3, dtype=np.float64)
        self.prev_action: Optional[np.ndarray] = None
        self.prev_achieved_goal: Optional[np.ndarray] = None

        super().__init__(
            n_actions=action_size,
            n_substeps=n_substeps,
            model_path=self.model_path,
            initial_qpos=self.neutral_joint_values,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.distance_threshold = distance_threshold

        # sample areas for the object and goal target
        self.obj_xy_range = obj_xy_range
        self.goal_xy_range = goal_xy_range
        self.goal_x_offset = goal_x_offset
        self.goal_z_range = goal_z_range

        self.goal_range_low = np.array(
            [-self.goal_xy_range / 2 + goal_x_offset, -self.goal_xy_range / 2, 0]
        )
        self.goal_range_high = np.array(
            [
                self.goal_xy_range / 2 + goal_x_offset,
                self.goal_xy_range / 2,
                self.goal_z_range,
            ]
        )
        self.obj_range_low = np.array(
            [-self.obj_xy_range / 2, -self.obj_xy_range / 2, 0]
        )
        self.obj_range_high = np.array(
            [self.obj_xy_range / 2, self.obj_xy_range / 2, 0]
        )

        self.goal_range_low[0] += 0.6
        self.goal_range_high[0] += 0.6
        self.obj_range_low[0] += 0.6
        self.obj_range_high[0] += 0.6

        # Compute max possible distance for normalization（第一段）
        self.max_distance = np.linalg.norm(self.goal_range_high - self.obj_range_low)

        # Three auxiliary variables
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

        # control range
        self.ctrl_range = self.model.actuator_ctrlrange

    # override the methods in MujocoRobotEnv
    # -----------------------------
    def _initialize_simulation(self) -> None:
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        # index used to distinguish arm and gripper joints
        free_joint_index = self._model_names.joint_names.index("obj_joint")
        self.arm_joint_names = self._model_names.joint_names[:free_joint_index][0:7]
        self.gripper_joint_names = self._model_names.joint_names[:free_joint_index][7:9]

        self._env_setup(self.neutral_joint_values)
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)

    def _env_setup(self, neutral_joint_values) -> None:
        self.set_joint_neutral()
        self.data.ctrl[0:7] = neutral_joint_values[0:7]
        self.reset_mocap_welds(self.model, self.data)

        self._mujoco.mj_forward(self.model, self.data)

        self.initial_mocap_position = self._utils.get_site_xpos(
            self.model, self.data, "ee_center_site"
        ).copy()
        self.grasp_site_pose = self.get_ee_orientation().copy()

        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        self._mujoco_step()

        self.initial_object_height = self._utils.get_joint_qpos(
            self.model, self.data, "obj_joint"
        )[2].copy()

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Get current object position before action
        object_position = self._utils.get_site_xpos(
            self.model, self.data, "obj_site"
        ).copy()

        # Initialize distances on first step
        if self.current_step == 0:
            self.initial_distance = self.goal_distance(object_position, self.goal)
            self.previous_distance = self.initial_distance
            self.best_distance = self.initial_distance
            # Initialize previous ee position
            self.previous_ee_position = self.get_ee_position().copy()
        else:
            # Store previous distance for progress tracking
            self.previous_distance = self.goal_distance(object_position, self.goal)
            # Store previous position for smoothness calculation
            self.previous_ee_position = self.get_ee_position().copy()

        self._set_action(action)
        self._mujoco_step(action)
        self._step_callback()

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs().copy()

        # Track trajectory
        ee_pos = self.get_ee_position()
        self.trajectory_points.append(ee_pos.copy())

        # Update total path length
        step_distance = np.linalg.norm(ee_pos - self.previous_ee_position)
        self.total_path_length += step_distance

        # Compute current distance
        current_distance = self.goal_distance(obs["achieved_goal"], self.goal)

        # Update best distance
        if current_distance < self.best_distance:
            self.best_distance = current_distance

        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
            "steps": self.current_step,
            "total_path_length": self.total_path_length,
            "distance_to_goal": current_distance,
            "best_distance": self.best_distance,
        }

        terminated = info["is_success"]
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)

        # 使用第二段的 dense 奖励（compute_reward 中实现）
        reward = self.compute_reward(
            obs["achieved_goal"], self.goal, info, action=action, obs_dict=obs
        )

        # 终止加成（第二段习惯），可设为 0 关闭
        if terminated and self.terminal_bonus != 0.0 and self.reward_type == "dense":
            reward = float(reward) + float(self.terminal_bonus)

        self.current_step += 1

        # 供第二段奖励使用的缓存
        self.prev_action = action.copy()
        self.prev_achieved_goal = obs["achieved_goal"].copy()

        return obs, reward, terminated, truncated, info

    def compute_reward(
        self,
        achieved_goal,
        desired_goal,
        info,
        action: Optional[np.ndarray] = None,
        obs_dict: Optional[dict] = None,
    ) -> SupportsFloat:
        d = float(self.goal_distance(achieved_goal, desired_goal))

        if self.reward_type == "sparse":
            # 保持第一段的稀疏样式（成功给 +1，否则 -1）
            return 1.0 if info.get("is_success", False) else -1.0

        # 以下为“第二段风格”的 dense 奖励（线性可调权重 + 去噪）
        reward_components: dict[str, float] = {}

        # 1) 距离惩罚（权重通常为负）
        reward_components["distance"] = self.w_distance * d

        # 2) 进步奖励（利用上一时刻的物体-目标距离）
        prev_distance = None
        if self.prev_achieved_goal is not None:
            prev_distance = float(
                self.goal_distance(self.prev_achieved_goal, desired_goal)
            )
        elif self.previous_distance is not None:
            prev_distance = float(self.previous_distance)

        progress_raw = 0.0
        if prev_distance is not None:
            progress_raw = float(prev_distance - d)
            # 对称 clip，正进步放宽两倍
            if progress_raw > 0:
                progress_raw = min(progress_raw, self.progress_clip * 2)
            else:
                progress_raw = max(progress_raw, -self.progress_clip)
        reward_components["progress"] = self.w_progress * progress_raw

        # 3) 夹爪配合奖励（在 ee-obj 接近时，鼓励宽度接近目标）
        gripper_component = 0.0
        if not self.block_gripper:
            # 从 obs 或现场计算
            ee_pos = self.get_ee_position()
            ee_obj_distance = float(np.linalg.norm(ee_pos - achieved_goal))
            gripper_width = float(self.get_fingers_width())  # fingers sum
            if ee_obj_distance < 0.05:
                gripper_component = self.w_gripper * (
                    -abs(gripper_width - self.gripper_target_width)
                )
        reward_components["gripper"] = gripper_component

        # 4) 高度奖励（鼓励抬升）
        height_bonus = max(0.0, float(achieved_goal[2] - self.initial_object_height))
        reward_components["height"] = self.w_height * height_bonus

        # 5) 平滑惩罚（根据 ee 线速度的 EMA）
        smooth_penalty = 0.0
        if obs_dict is not None:
            observation = obs_dict["observation"]
            # 第一段 obs 中 ee_velocity 是乘以 dt 的，需要除以 dt 恢复速度
            if not self.block_gripper:
                ee_vel_dt = observation[3:6]
            else:
                ee_vel_dt = observation[3:6]
            ee_velocity = ee_vel_dt / max(self.dt, 1e-8)
            self.vel_ema = (
                self.vel_ema_tau * self.vel_ema + (1.0 - self.vel_ema_tau) * ee_velocity
            )
            smooth_penalty = float(np.linalg.norm(self.vel_ema))
        reward_components["smooth"] = self.w_smooth * smooth_penalty

        # 6) 动作与动作变化惩罚
        action_component = 0.0
        action_change_component = 0.0
        if action is not None:
            action_norm = float(np.linalg.norm(action))
            action_component = self.w_action * action_norm
            if self.prev_action is not None:
                action_diff = float(np.linalg.norm(action - self.prev_action))
                action_change_component = self.w_action_change * action_diff
        reward_components["action"] = action_component
        reward_components["action_change"] = action_change_component

        # 7) 成功奖励（当前步达到阈值）
        success_component = self.success_reward if d < self.distance_threshold else 0.0
        reward_components["success"] = success_component

        total_reward = float(sum(reward_components.values()))
        info["reward_components"] = reward_components
        return total_reward

    def _set_action(self, action) -> None:
        action = action.copy()
        # for the pick and place task
        if not self.block_gripper:
            pos_ctrl, gripper_ctrl = action[:3], action[3]
            fingers_ctrl = gripper_ctrl * 0.2
            fingers_width = self.get_fingers_width().copy() + fingers_ctrl
            fingers_half_width = np.clip(
                fingers_width / 2, self.ctrl_range[-1, 0], self.ctrl_range[-1, 1]
            )

        elif self.block_gripper:
            pos_ctrl = action
            fingers_half_width = 0

        # control the gripper
        self.data.ctrl[-2:] = fingers_half_width

        # control the end-effector with mocap body
        pos_ctrl *= self.pos_ctrl_scale
        pos_ctrl += self.get_ee_position().copy()
        pos_ctrl[2] = np.max((0, pos_ctrl[2]))

        self.set_mocap_pose(pos_ctrl, self.grasp_site_pose)

    def _get_obs(self) -> dict:
        # robot
        ee_position = self._utils.get_site_xpos(
            self.model, self.data, "ee_center_site"
        ).copy()

        ee_velocity = (
            self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy()
            * self.dt
        )

        if not self.block_gripper:
            fingers_width = self.get_fingers_width().copy()

        # object
        object_position = self._utils.get_site_xpos(
            self.model, self.data, "obj_site"
        ).copy()

        object_rotation = rotations.mat2euler(
            self._utils.get_site_xmat(self.model, self.data, "obj_site")
        ).copy()

        object_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "obj_site").copy()
            * self.dt
        )

        object_velr = (
            self._utils.get_site_xvelr(self.model, self.data, "obj_site").copy()
            * self.dt
        )

        # Additional useful observations（第一段）
        if hasattr(self, "goal") and self.goal is not None and self.goal.shape[0] > 0:
            object_goal_distance = np.linalg.norm(object_position - self.goal)
            ee_object_distance = np.linalg.norm(ee_position - object_position)
            goal_rel_pos = self.goal - object_position
            object_rel_ee = object_position - ee_position
        else:
            object_goal_distance = 0.0
            ee_object_distance = 0.0
            goal_rel_pos = np.zeros(3)
            object_rel_ee = np.zeros(3)

        # Normalized time step
        normalized_time = self.current_step / self.max_episode_steps

        if not self.block_gripper:
            obs = {
                "observation": np.concatenate(
                    [
                        ee_position,
                        ee_velocity,
                        fingers_width,
                        object_position,
                        object_rotation,
                        object_velp,
                        object_velr,
                        goal_rel_pos,  # Relative position of goal to object
                        object_rel_ee,  # Relative position of object to ee
                        [ee_object_distance],
                        [object_goal_distance],
                        [normalized_time],
                    ]
                ).copy(),
                "achieved_goal": object_position.copy(),
                "desired_goal": (
                    self.goal.copy()
                    if hasattr(self, "goal") and self.goal is not None
                    else object_position.copy()
                ),
            }
        else:
            obs = {
                "observation": np.concatenate(
                    [
                        ee_position,
                        ee_velocity,
                        object_position,
                        object_rotation,
                        object_velp,
                        object_velr,
                        goal_rel_pos,
                        object_rel_ee,
                        [ee_object_distance],
                        [object_goal_distance],
                        [normalized_time],
                    ]
                ).copy(),
                "achieved_goal": object_position.copy(),
                "desired_goal": (
                    self.goal.copy()
                    if hasattr(self, "goal") and self.goal is not None
                    else object_position.copy()
                ),
            }

        return obs

    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _render_callback(self) -> None:
        # visualize goal site
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._model_names.site_name2id["target"]
        self.model.site_pos[site_id] = self.goal - sites_offset[site_id]
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self) -> bool:
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        self.set_joint_neutral()
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        self._sample_object()

        # Reset episode tracking（第一段）
        self.current_step = 0
        self.trajectory_points = []
        self.previous_ee_position = None
        self.total_path_length = 0.0
        self.previous_distance = None
        self.best_distance = None
        self.initial_distance = None

        # 第二段 dense 所需缓存
        self.prev_action = None
        self.prev_achieved_goal = None
        self.vel_ema[:] = 0.0

        # 采样目标（与第二段显式一致）
        self.goal = self._sample_goal()

        # Store initial object position
        self.initial_object_position = self._utils.get_site_xpos(
            self.model, self.data, "obj_site"
        ).copy()

        # 初始化 prev_distance 用于进步项
        self.previous_distance = float(
            self.goal_distance(self.initial_object_position, self.goal)
        )
        self.initial_distance = self.previous_distance

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    # custom methods
    # -----------------------------
    def reset_mocap_welds(self, model, data) -> None:
        if model.nmocap > 0 and model.eq_data is not None:
            for i in range(model.eq_data.shape[0]):
                if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    # relative pose
                    model.eq_data[i, 3:10] = np.array(
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                    )
        self._mujoco.mj_forward(model, data)

    def goal_distance(self, goal_a, goal_b) -> SupportsFloat:
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def set_mocap_pose(self, position, orientation) -> None:
        self._utils.set_mocap_pos(self.model, self.data, "panda_mocap", position)
        self._utils.set_mocap_quat(self.model, self.data, "panda_mocap", orientation)

    def set_joint_neutral(self) -> None:
        # assign value to arm joints
        for name, value in zip(self.arm_joint_names, self.neutral_joint_values[0:7]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)

        # assign value to finger joints
        for name, value in zip(
            self.gripper_joint_names, self.neutral_joint_values[7:9]
        ):
            self._utils.set_joint_qpos(self.model, self.data, name, value)

    def _sample_goal(self) -> np.ndarray:
        goal = np.array([0.0, 0.0, self.initial_object_height])
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        # for the pick and place task
        if not self.block_gripper and self.goal_z_range > 0.0:
            if self.np_random.random() < 0.3:
                noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self) -> None:
        object_position = np.array([0.0, 0.0, self.initial_object_height])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        object_xpos = np.concatenate([object_position, np.array([1, 0, 0, 0])])
        self._utils.set_joint_qpos(self.model, self.data, "obj_joint", object_xpos)

    def get_ee_orientation(self) -> np.ndarray:
        site_mat = self._utils.get_site_xmat(
            self.model, self.data, "ee_center_site"
        ).reshape(9, 1)
        current_quat = np.empty(4)
        self._mujoco.mju_mat2Quat(current_quat, site_mat)
        return current_quat

    def get_ee_position(self) -> np.ndarray:
        return self._utils.get_site_xpos(self.model, self.data, "ee_center_site")

    def get_body_state(self, name: str) -> np.ndarray:
        body_id = self._model_names.body_name2id[name]
        body_xpos = self.data.xpos[body_id]
        body_xquat = self.data.xquat[body_id]
        body_state = np.concatenate([body_xpos, body_xquat])
        return body_state

    def get_fingers_width(self) -> np.ndarray:
        finger1 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint1")
        finger2 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint2")
        return finger1 + finger2
