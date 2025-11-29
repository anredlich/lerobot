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
training outputs directory.

It requires the installation of the 'gym_aloha' simulation environment with TrossenAIStationary sim added:
https://github.com/anredlich/gym-aloha
see installation instructions there.
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
output_directory = Path("outputs/eval/example_trossen_ai_stationary")
output_directory.mkdir(parents=True, exist_ok=True)

# Select your device
device = "cuda"

##id = "gym_aloha/AlohaTransferCube-v0"
#choose either the simulated robot:
id = "gym_aloha/TrossenAIStationaryTransferCube-v0"
#or the end effector controlled simulator, used to create datasets:
#id = "gym_aloha/TrossenAIStationaryTransferCubeEE-v0"

# Provide the [hugging face repo id](https://huggingface.co/ANRedlich/trossen_ai_stationary_sim_act7):
# OR a path to a local outputs/train folder:

#pretrained_policy_path=Path("ANRedlich/trossen_ai_stationary_sim_act7") #from train.py on sim NRedlich/trossen_ai_stationary_sim_transfer_40mm_cube_07
#pretrained_policy_path=Path("ANRedlich/trossen_ai_stationary_sim_act8") #from train.py on sim ANRedlich/trossen_ai_stationary_sim_transfer_40mm_cube_08
#pretrained_policy_path=Path("ANRedlich/trossen_ai_stationary_sim_act10") #from train.py on sim ANRedlich/trossen_ai_stationary_sim_transfer_40mm_cube_10
pretrained_policy_path=Path("ANRedlich/trossen_ai_stationary_sim_act13") #from train.py on sim ANRedlich/trossen_ai_stationary_sim_transfer_40mm_cube_13
#this policy wasn learned from a real robot dataset. Used here to test real -> sim:
#pretrained_policy_path=Path("ANRedlich/trossen_ai_stationary_real_act2_3") #from train.py on real ANRedlich/trossen_ai_stationary_transfer_40mm_cube_02

#these are our local model and dataset paths, which are not, however, in the distribution
#pretrained_policy_path=Path("lerobot/scripts/outputs/train/trossen_ai_stationary_act7/checkpoints/last/pretrained_model") #from train.py on dataset.root=lerobot/scripts/dataset/eval7
#pretrained_policy_path=Path("lerobot/scripts/outputs/train/trossen_ai_stationary_act8/checkpoints/last/pretrained_model") #from train.py on dataset.root=lerobot/scripts/dataset/eval8
#pretrained_policy_path=Path("lerobot/scripts/outputs/train/trossen_ai_stationary_act10/checkpoints/last/pretrained_model") #from train.py on dataset.root=lerobot/scripts/dataset/eval10
#pretrained_policy_path=Path("lerobot/scripts/outputs/train/trossen_ai_stationary_act13/checkpoints/last/pretrained_model") #from train.py on dataset.root=lerobot/scripts/dataset/eval13
#pretrained_policy_path=Path("lerobot/scripts/outputs/train/act_trossen_ai_stationary_real_02_3/checkpoints/last/pretrained_model") #from train.py on dataset.root=lerobot/scripts/dataset/real_data_02

#set the env variables to those used to learn the pretrained policy:
box_size=[0.02,0.02,0.02]
box_pos=None
box_color=None
tabletop=None
backdrop=None
lighting=None
arms_pos=None
arms_ref=None
if "act8" in pretrained_policy_path.name:
    box_color=[0.86, 0.18, 0.18,1]
    tabletop='my_desktop'
    backdrop='my_backdrop'
    lighting=[[0.3,0.3,0.3],[0.3,0.3,0.3]]
elif "act10" in pretrained_policy_path.name:
    tabletop='my_desktop'
elif "act13" in pretrained_policy_path.name:
    box_pos=[0.0,0.0,-0.02]
    box_color=[1,0,0,1]
    tabletop='my_desktop'
    backdrop='my_backdrop'
    lighting=[[0.3,0.3,0.3],[0.3,0.3,0.3]]
elif "act2_3" in pretrained_policy_path.name:
    box_pos=[0.0,0.0,-0.02]
    box_color=[1,0,0,1]
    tabletop='my_desktop'
    backdrop='my_backdrop'
    lighting=[[0.1,0.1,0.1],[-0.5,0.5,0.5]] #adjusts sim lighting to better match real robot
    arms_pos=[-0.4575, 0.0, 0.02, 0.4575, 0.0, 0.02] #adjusts sim position of robot arms to better match real robot
    arms_ref=[0,-0.025,0.025,0,0,0,0,-0.025,0.025,0,0,0] #adjusts sim joint qpos0 to better match real robot

env = gym.make(
    id,
    obs_type="pixels_agent_pos",
    max_episode_steps=500,
    box_size=box_size,
    box_pos=box_pos,
    box_color=box_color,
    tabletop=tabletop,
    backdrop=backdrop,
    lighting=lighting,
    arms_pos=arms_pos,
    arms_ref=arms_ref,
    #render_mode="human",
)

if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube':
    policy = ACTPolicy.from_pretrained(pretrained_policy_path)
if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
    inject_noise = False
    policy_cls = PickAndTransferPolicy
    policy = PickAndTransferPolicy(inject_noise=inject_noise,box_size=box_size)

# We can verify that the shapes of the features expected by the policy match the ones from the observations
# produced by the environment
if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
    print(16)
else:
    print(policy.config.input_features)
print(env.observation_space)

# Similarly, we can check that the actions produced by the policy will match the actions expected by the
# environment
if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
    print(16)
else:
    print(policy.config.output_features)
print(env.action_space)

# Reset the policy and environments to prepare for rollout
if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
    policy.reset()
    numpy_observation, info = env.reset(seed=41)
elif env.unwrapped.task == 'trossen_ai_stationary_transfer_cube':
    policy.reset()
    numpy_observation, info = env.reset(seed=1002)

if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
    ts=dm_env.TimeStep(step_type=dm_env.StepType.FIRST,reward=None,discount=None,observation=info['raw_obs'])
    print(f"cube position={ts.observation['env_state'][:3]}")
print(f"cube pos={info['raw_obs']['env_state']}")
print(f"home pose={numpy_observation['agent_pos']}")

# Prepare to collect every rewards and all the frames of the episode,
# from initial state to final state.
rewards = []
frames = []

# Render frame of the initial state
frames.append(env.render())

cam_list=["top"]
if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube' or env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
    cam_list=["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

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
        if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
            action = policy(ts)
        else:
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
    if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
        if terminated:
            ts=dm_env.TimeStep(dm_env.StepType.LAST, reward, 1.0, observation=info['raw_obs'])
        else:
            ts=dm_env.TimeStep(dm_env.StepType.MID, reward, 1.0, observation=info['raw_obs'])

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
