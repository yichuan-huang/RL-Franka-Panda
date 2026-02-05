import os
import time
import gymnasium as gym
import panda_mujoco_gym  # ensure installed
import imageio.v2 as imageio
from stable_baselines3 import SAC
import numpy as np

# ---- Config ----
env_id = "FrankaPickAndPlaceDense-v0"
model_path = "model/SAC_pick_and_place.zip"
episodes = 50
# Target FPS of the recorded video (logical playback rate)
video_fps = 30
# Slowdown factor: 1.0 = real-time (as captured), 0.5 = half-speed (slower), 0.25 = quarter-speed, etc.
# We implement slowdown by duplicating frames. For example, 0.5 speed => each frame duplicated 2x.
slowdown_factor = (
    0.5  # smaller => slower playback (e.g., 0.5 is 2x slower, 0.25 is 4x slower)
)
# Output
video_folder = "videos"
video_filename = "SAC_pick_and_place_all_episodes_slow.mp4"

os.makedirs(video_folder, exist_ok=True)
video_path = os.path.join(video_folder, video_filename)

# ---- Create environment ----
# Prefer rgb_array for stable frame capture
# If your env does not support rgb_array, try omitting render_mode and use env.render() with mode="rgb_array" if available.
# If you must use "human", see notes below for screen-capture alternatives (not covered here).
env = gym.make(env_id, render_mode="rgb_array") 

# ---- Load model ----
model = SAC.load(model_path)


# ---- Helper to grab a frame ----
def get_frame_from_env(env, info=None):
    # Most Gymnasium Mujoco envs expose frames via env.render(). With render_mode="rgb_array", step() already updates internal buffer.
    frame = env.render()  # should return an RGB array (H, W, 3), dtype=uint8
    if frame is None:
        # Fallback: some envs provide 'rgb' inside info or via env.sim.render. Adjust if needed.
        raise RuntimeError(
            "env.render() returned None. Ensure render_mode='rgb_array' or adapt frame extraction."
        )
    # Ensure uint8
    if frame.dtype != np.uint8:
        frame = (
            (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            if frame.max() <= 1.0
            else frame.astype(np.uint8)
        )
    return frame


# ---- Prepare video writer ----
# We will open the writer lazily after getting the first frame to know frame size
writer = None

# Convert slowdown_factor to frame duplication count.
# Example: slowdown_factor=0.5 -> dup=2; 0.25 -> dup=4; 1.0 -> dup=1
if slowdown_factor <= 0:
    raise ValueError("slowdown_factor must be > 0.")
dup_count = max(1, int(round(1.0 / slowdown_factor)))

total_frames_written = 0

try:
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0.0

        # Capture an initial frame right after reset
        first_frame = get_frame_from_env(env, info)
        if writer is None:
            # Initialize writer with the frame size
            h, w = first_frame.shape[:2]
            writer = imageio.get_writer(
                video_path,
                fps=video_fps,
                codec="libx264",
                quality=8,  # 0(worst)-10(best), you can switch to bitrate if desired
                macro_block_size=None,  # avoid resizing for non-multiple-of-16
                pixelformat="yuv420p",  # broad compatibility
            )
        # Write the initial frame (duplicated for slowdown)
        for _ in range(dup_count):
            writer.append_data(first_frame)
            total_frames_written += 1

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)

            frame = get_frame_from_env(env, info)
            # Duplicate frames to slow down playback
            for _ in range(dup_count):
                writer.append_data(frame)
                total_frames_written += 1

        print(
            f"Episode {ep + 1} finished. reward={ep_reward:.3f}, terminated={terminated}, truncated={truncated}"
        )

finally:
    if writer is not None:
        writer.close()
    env.close()

print(f"Combined slow video saved to: {os.path.abspath(video_path)}")
print(
    f"Frames written: {total_frames_written}, playback FPS: {video_fps}, slowdown dup per frame: {dup_count}"
)
