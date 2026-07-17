import collections
import time

import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from unitree_deploy.utils.rerun_visualizer import RerunLogger, visualization_data


def extract_observation(step: dict):
    observation = {}

    for key, value in step.items():
        if key.startswith("observation.images."):
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] in [1, 3]:
                value = np.transpose(value, (2, 0, 1))
            observation[key] = value

        elif key == "observation.state":
            observation[key] = value

    return observation


class DatasetEvalEnv:
    def __init__(
        self,
        repo_id: str,
        episode_index: int = 0,
        image_path: str = "./",
        visualization: bool = True,
    ):
        self.dataset = LeRobotDataset(repo_id=repo_id)
        self.visualization = visualization
        if self.visualization:
            self.rerun_logger = RerunLogger()

        self.from_idx = self.dataset.meta.episodes["dataset_from_index"][episode_index]
        self.to_idx = self.dataset.meta.episodes["dataset_to_index"][episode_index]
        self.step_idx = self.from_idx
        self.espoide_total_steps = self.to_idx - self.from_idx

        self.ground_truth_actions = []
        self.predicted_actions = []
        self.image_path = image_path

    def get_observation(self):
        step = self.dataset[self.step_idx]
        observation = extract_observation(step)

        state = step["observation.state"].numpy()
        # self.ground_truth_actions.append(step["action"].numpy())

        if self.visualization:
            visualization_data(
                self.step_idx,
                observation,
                observation["observation.state"],
                step["action"].numpy(),
                self.rerun_logger,
            )

        images_observation = {
            key: value.numpy() for key, value in observation.items() if key.startswith("observation.images.")
        }

        obs = collections.OrderedDict(
            qpos=state,
            qvel=np.zeros_like(state),
            effort=np.zeros_like(state),
            images=images_observation,
        )

        return obs

    def step(self, action):
        self.predicted_actions.append(action)

        step = self.dataset[self.step_idx]
        self.ground_truth_actions.append(step["action"].numpy())

        self.step_idx += 1
        print("DatasetEvalEnv step_idx:", self.step_idx)
        if self.step_idx == self.to_idx:
            self._plot_results()
            exit()

    def _plot_results(self):
        ground_truth_actions = np.array(self.ground_truth_actions)
        predicted_actions = np.array(self.predicted_actions)

        n_timesteps, n_dims = ground_truth_actions.shape

        fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims), sharex=True)
        fig.suptitle("Ground Truth vs Predicted Actions")

        for i in range(n_dims):
            ax = axes[i] if n_dims > 1 else axes
            ax.plot(ground_truth_actions[:, i], label="Ground Truth", color="blue")
            ax.plot(predicted_actions[:, i], label="Predicted", color="red", linestyle="--")
            ax.set_ylabel(f"Dim {i + 1}")
            ax.legend()

        axes[-1].set_xlabel("Timestep")
        plt.tight_layout()

        plt.savefig(self.image_path + "figure.png")
        time.sleep(1)


def make_dataset_eval_env(
    repo_id: str,
    episode_index: int = 0,
    image_path: str = "./",
    visualization: bool = True,
) -> DatasetEvalEnv:
    return DatasetEvalEnv(repo_id, episode_index, visualization, image_path)


if __name__ == "__main__":
    eval_dataset = DatasetEvalEnv(repo_id="hengguo/pick_apple")
    while True:
        observation = eval_dataset.get_observation()
        eval_dataset.step(observation["qpos"])
