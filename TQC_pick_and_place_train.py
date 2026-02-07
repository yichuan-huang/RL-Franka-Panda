import time

import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import panda_mujoco_gym


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training")

    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(1)
        return True  # Continue training

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


env = gym.make("FrankaPickAndPlaceDense-v0", render_mode="rgb_array")
model = TQC(
    "MultiInputPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=1000000,
    batch_size=512,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, "step"),
    gradient_steps=1,
    learning_starts=10000,
    ent_coef="auto",
    top_quantiles_to_drop_per_net=2,
    policy_kwargs={"n_quantiles": 25, "n_critics": 2},
    verbose=1,
    tensorboard_log="./logs/tqc_pick_and_place/",
)
start_time = time.time()
progress_callback = ProgressBarCallback(total_timesteps=1000000)
model.learn(total_timesteps=1000000, callback=progress_callback)
model.save("model/TQC_pick_and_place.zip")
end_time = time.time()
with open("./logs/tqc_pick_and_place/tqc_pick_and_place_train_time.txt", "w") as opener:
    opener.write("spend_tine:{}".format(end_time - start_time))

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
