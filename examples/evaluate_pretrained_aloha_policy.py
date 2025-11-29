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

"""
This scripts demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.

It requires the installation of the 'gym_aloha' simulation environment.
"""

import json
from pathlib import Path

#import gym_pusht  # noqa: F401
import gym_aloha 
import gymnasium as gym
import imageio
import numpy
import torch
import matplotlib.pyplot as plt
import dm_env

#from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.scripted_policy import PickAndTransferPolicy
from lerobot.common.robot_devices.robots.utils import normalize_state, unnormalize_numpy_action
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

# Create a directory to store the video of the evaluation
output_directory = Path("outputs/eval/example_aloha_act2") #anr example_pusht_diffusion
output_directory.mkdir(parents=True, exist_ok=True)

# Select your device
device = "cuda"

# Initialize evaluation environment to render two observation types:
# an image of the scene and state/position cube position=[-0.02472291 -0.13617125  0.0125    ]of the agent. The environment
# also automatically stops running after 300 interactions/steps.
env = gym.make(
    "gym_aloha/AlohaTransferCube-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=400,
)

# Provide the [hugging face repo id](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human):
# OR a path to a local outputs/train folder.
if env.unwrapped.task == 'transfer_cube':
    pretrained_policy_path = Path("lerobot/act_aloha_sim_transfer_cube_human") #seed=41 -> step=266,270
    #the policies below were trained using train_aloha_policy.py or train.py, these are not in the distribution:
    #pretrained_policy_path = Path("outputs/train/act_aloha_transfer/checkpoints/last/pretrained_model") #from train.py seed=41 -> step=264,266
    #pretrained_policy_path = Path("lerobot/scripts/outputs/train/example_aloha_act2") #from train_aloha_policy seed=42 -> step=293
    #pretrained_policy_path = Path("lerobot/scripts/outputs/train/example_aloha_act6") #from train_aloha_policy seed=42 -> step=293, 281
    config_path = pretrained_policy_path/"config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        policy_type = config_dict.get("type", "act")  # Default to "act" if not specified
    else:
        policy_type = 'act'
    if policy_type=='diffusion':
        policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    else:
        policy = ACTPolicy.from_pretrained(pretrained_policy_path)

# We can verify that the shapes of the features expected by the policy match the ones from the observations
# produced by the environment
print(policy.config.input_features)
print(env.observation_space)

# Similarly, we can check that the actions produced by the policy will match the actions expected by the
# environment
print(policy.config.output_features)
print(env.action_space)

# Reset the policy and environments to prepare for rollout
policy.reset()
numpy_observation, info = env.reset(seed=41) ##seed=40 #,options={'box_color':[1,0,0,.025]}) #seed=1000,options={'box_size':[0.02,0.02,0.02],'box_color':[0,0,1,1]}) #41)

print(f"cube pos={info['raw_obs']['env_state']}")
print(f"home pose={numpy_observation['agent_pos']}")

# Prepare to collect every rewards and all the frames of the episode,
# from initial state to final state.
rewards = []
frames = []

# Render frame of the initial state
frames.append(env.render())

cam_list=["top"]
plt_imgs = gym_aloha.utils.plot_observation_images(numpy_observation['pixels'], cam_list)
plt.pause(0.02)

step = 0
done = False
while not done:
    
    for i in range(len(cam_list)): #anr added
        plt_imgs[i].set_data(numpy_observation['pixels'][cam_list[i]])
    plt.pause(0.02)
    
    # Prepare observation for the policy running in Pytorch
    state = torch.from_numpy(numpy_observation["agent_pos"])
    #image = torch.from_numpy(numpy_observation["pixels"]["top"]) #anr was
    image = [] #anr added
    for cam in cam_list:
        image.append(torch.from_numpy(numpy_observation["pixels"][cam])) #anr new

    # Convert to float32 with image from channel first in [0,255]
    # to channel last in [0,1]
    state = state.to(torch.float32)
    for i, cam in enumerate(cam_list):
        image[i] = image[i].to(torch.float32) / 255
        image[i] = image[i].permute(2, 0, 1)

    if hasattr(policy, 'config') and hasattr(policy.config, 'normalize_data') and policy.config.normalize_data:
        state = normalize_state(state)

    # Send data tensors from CPU to GPU
    state = state.to(device, non_blocking=True)
    for i, cam in enumerate(cam_list):
        image[i] = image[i].to(device, non_blocking=True)

    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    for i, cam in enumerate(cam_list):
        image[i] = image[i].unsqueeze(0)

    # Create the policy input dictionary
    observation = {
        "observation.state": state,
        #"observation.images.top": image[0],
        **{f"observation.images.{cam}": image[i] for i, cam in enumerate(cam_list)}
    }

    # Predict the next action with respect to the current observation
    with torch.inference_mode():
        ##if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
        ##    action = policy(ts)
        ##else:
        action = policy.select_action(observation)

    # Prepare the action for the environment
    if isinstance(action, torch.Tensor):
        numpy_action = action.squeeze(0).to("cpu").numpy()
    else:
        numpy_action = action

    if hasattr(policy, 'config') and hasattr(policy.config, 'normalize_data') and policy.config.normalize_data:
        numpy_action = unnormalize_numpy_action(numpy_action)

    # Step through the environment and receive a new observation
    numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
    print(f"{step=} {reward=} {terminated=}") #{numpy_action[:5]}

    # Keep track of all the rewards and frames
    rewards.append(reward)
    frames.append(env.render())

    # The rollout is considered done when the success state is reach (i.e. terminated is True),
    # or the maximum number of iterations is reached (i.e. truncated is True)
    done = terminated | truncated | done
    step += 1

if terminated:
    print("Success!")
else:
    print("Failure!")

# Get the speed of environment (i.e. its number of frames per second).
fps = env.metadata["render_fps"]

# Encode all frames into a mp4 video.
video_path = output_directory / "rollout.mp4"
imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

print(f"Video of the evaluation is available in '{video_path}'.")
