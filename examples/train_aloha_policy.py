# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from pathlib import Path

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.robot_devices.robots.utils import normalize_batch

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

def main():
    # Create a directory to store the training checkpoint.
    #output_directory = Path("outputs/train/example_aloha_act5")
    #output_directory = Path("lerobot/scripts/outputs/train/example_aloha_act7")
    #output_directory = Path("lerobot/scripts/outputs/train/example_aloha_act9")
    output_directory = Path("lerobot/scripts/outputs/train/example_aloha_diffusion2")
    #output_directory = Path("lerobot/scripts/outputs/train/trossen_ai_stationary_act_ex1")
    output_directory.mkdir(parents=True, exist_ok=False)

    repo_id = "lerobot/aloha_sim_transfer_cube_human"
    root = None
    #repo_id = "ANRedlich/eval_act_trossen_ai_stationary_test_01"
    #root = "lerobot/scripts/dataset/eval3"
    #repo_id = "ANRedlich/eval_act_trossen_ai_stationary_test_06" #BIG DATASET
    #root = "lerobot/scripts/dataset/eval6" #BIG DATASET
    #repo_id="ANRedlich/trossen_ai_stationary_test_07"
    #root="lerobot/scripts/dataset/experiment5"

    # # Select your device
    device = torch.device("cuda")

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 20000 #5000
    log_freq = 1

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset_metadata = LeRobotDatasetMetadata(repo_id=repo_id,root=root) #"lerobot/aloha_sim_transfer_cube_human"
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    policy_type='act' #'diffusion' #'act' is default

    # Policies are initialized with a configuration class, in this case `ActConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    if policy_type=='diffusion':
        cfg = DiffusionConfig(input_features=input_features, output_features=output_features, crop_shape=None) #crop_shape over-ride of 84x84 -> dataset image size
        #pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
    else:
        cfg = ACTConfig(input_features=input_features, output_features=output_features) #, n_action_steps=10, chunk_size=10, normalize_data=False)

    #correct ImageNet normalization, but may work worse on simulated data!
    if policy_type == 'act' and hasattr(cfg, 'pretrained_backbone_weights') and cfg.pretrained_backbone_weights is not None:
        IMAGENET_STATS = {
            "mean": [[[0.485]], [[0.456]], [[0.406]]],
            "std": [[[0.229]], [[0.224]], [[0.225]]],
        }
        # Get camera keys from metadata
        camera_keys = [key for key in dataset_metadata.features.keys() if 'image' in key]
        for key in camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset_metadata.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        print("Applied ImageNet stats for pretrained ResNet backbone")

    # We can now instantiate our policy with this config and the dataset stats.
    if policy_type=='diffusion':
        policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    else:
        policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # Another policy-dataset interaction is with the delta_timestamps. Each policy expects a given number frames
    # which can differ for inputs, outputs and rewards (if there are some).
    if policy_type=='diffusion':
        delta_timestamps = {
            #"observation.image.top": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
            "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
            "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
        }
        for img_key in cfg.image_features.keys():
            delta_timestamps[img_key] = [i / dataset_metadata.fps for i in cfg.observation_delta_indices]
    else:
        delta_timestamps = {
            #"observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
            #"observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
            "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
        }

    # In this case with the standard configuration for Act Policy, it is equivalent to this:
    # delta_timestamps = {
    #     # Load the previous image and state at -0.1 seconds before current frame,
    #     # then load current image and state corresponding to 0.0 second.
    #     "observation.image": [-0.1, 0.0],
    #     "observation.state": [-0.1, 0.0],
    #     # Load the previous action (-0.1), the next action to be executed (0.0),
    #     # and 14 future actions with a 0.1 seconds spacing. All these actions will be
    #     # used to supervise the policy.
    #     "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    # }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(repo_id=repo_id,root=root, delta_timestamps=delta_timestamps) #"lerobot/aloha_sim_transfer_cube_human"

    if 'pretrained_backbone_weights' in dir(cfg) and cfg.pretrained_backbone_weights is not None:
        #assume pretrained_backbone_weights are ImageNet:
        IMAGENET_STATS = {
            "mean": [[[0.485]], [[0.456]], [[0.406]]],
            "std": [[[0.229]], [[0.224]], [[0.225]]],
        }
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        print("Applied ImageNet stats for pretrained ResNet backbone")

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64, #8 needed for trossen_staitionary_ai because of size
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Dataloader info
    if 0:
        print("\n=== Checking Image Format from DataLoader ===")
        test_batch = next(iter(dataloader))
        if "observation.images.top" in test_batch:
            img_tensor = test_batch["observation.images.top"]
            print(f"Image tensor shape: {img_tensor.shape}")
            print(f"Image tensor dtype: {img_tensor.dtype}")
            print(f"Image value range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
            print(f"Expected shape for diffusion: (batch_size, n_obs_steps, C, H, W)")
            print(f"                            or (batch_size, C, H, W) depending on temporal stacking")
        else:
            print("No observation.images.top in batch!")
        print("=" * 50 + "\n")

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            if policy_type == 'act' and policy.config.normalize_data:
                batch=normalize_batch(batch)
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)


if __name__ == "__main__":
    main()
