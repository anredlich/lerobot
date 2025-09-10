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

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install --no-binary=av -e ".[pusht]"`
```
"""

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

# Create a directory to store the video of the evaluation
output_directory = Path("outputs/eval/example_aloha_act2") #anr example_pusht_diffusion
output_directory.mkdir(parents=True, exist_ok=True)

# Select your device
device = "cuda"

# Initialize evaluation environment to render two observation types:
# an image of the scene and state/position cube position=[-0.02472291 -0.13617125  0.0125    ]of the agent. The environment
# also automatically stops running after 300 interactions/steps.
id = "gym_aloha/AlohaTransferCube-v0"
id = "gym_aloha/TrossenAIStationaryTransferCube-v0"
#id = "gym_aloha/TrossenAIStationaryTransferCubeEE-v0"
if id == "gym_aloha/TrossenAIStationaryTransferCubeEE-v0" or id == "gym_aloha/TrossenAIStationaryTransferCube-v0":
    max_episode_steps = 500
else:
    max_episode_steps = 400
env = gym.make(
    id,
    obs_type="pixels_agent_pos",
    max_episode_steps=max_episode_steps,
    box_size=[0.02,0.02,0.02],
    tabletop='wood',
    #box_color=[0.86, 0.18, 0.18,1]
    #render_mode="human"  # This enables the built-in Gymnasium viewer #anr
)

# Provide the [hugging face repo id](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human):
# OR a path to a local outputs/train folder.
#pretrained_policy_path = Path("lerobot/scripts/outputs/train/example_aloha_act2") #from train_aloha_policy
#pretrained_policy_path = Path("outputs/train/act_aloha_transfer/checkpoints/last/pretrained_model") #from train.py
if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube':
    pretrained_policy_path = Path("lerobot/scripts/outputs/train/act_trossen_ai_stationary_test_07_01/checkpoints/last/pretrained_model") #from training real robot
    pretrained_policy_path = Path("lerobot/scripts/outputs/train/trossen_ai_stationary_act1") #from training simulated robot
    pretrained_policy_path=Path("lerobot/scripts/outputs/train/trossen_ai_stationary_act2/checkpoints/last/pretrained_model") #from train.py
    pretrained_policy_path=Path("lerobot/scripts/outputs/train/trossen_ai_stationary_act6/checkpoints/last/pretrained_model") #from train.py on BIG sim DATASET1 seed=1000 -> step=460
    pretrained_policy_path=Path("lerobot/scripts/outputs/train/trossen_ai_stationary_act7/checkpoints/last/pretrained_model") #from train.py on BIG sim DATASET2 20mm seed=1000 -> step=452
    #pretrained_policy_path=Path("lerobot/scripts/outputs/train/act_trossen_ai_stationary_real_01/checkpoints/last/pretrained_model") #from train.py on BIG real DATASET3 20mm seed=1000 -> step=460
    #pretrained_policy_path = Path("lerobot/scripts/outputs/train/trossen_ai_stationary_act3")
    #pretrained_policy_path = Path("lerobot/scripts/outputs/train/trossen_ai_stationary_act7") #train_aloha_policy with normalize_data
    #pretrained_policy_path = Path("lerobot/scripts/outputs/train/trossen_ai_stationary_act7/checkpoints/008000/pretrained_model") #normalize_data
    #pretrained_policy_path = Path("lerobot/scripts/outputs/train/trossen_ai_stationary_act8/checkpoints/last/pretrained_model") #normalize_data
    policy = ACTPolicy.from_pretrained(pretrained_policy_path)
if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
    inject_noise = False
    policy_cls = PickAndTransferPolicy
    policy = PickAndTransferPolicy(inject_noise=inject_noise,box_size=[0.02,0.02,0.02])
elif env.unwrapped.task == 'transfer_cube':
    pretrained_policy_path = "lerobot/act_aloha_sim_transfer_cube_human" #seed=41 -> step=266,270
    #pretrained_policy_path = Path("outputs/train/act_aloha_transfer/checkpoints/last/pretrained_model") #from train.py seed=41 -> step=264,266
    #pretrained_policy_path = Path("lerobot/scripts/outputs/train/example_aloha_act2") #from train_aloha_policy seed=42 -> step=293
    pretrained_policy_path = Path("lerobot/scripts/outputs/train/example_aloha_act6") #from train_aloha_policy seed=42 -> step=293
    #pretrained_policy_path = Path("lerobot/scripts/outputs/train/example_aloha_act7") #from train_aloha_policy
    #pretrained_policy_path = Path("lerobot/scripts/outputs/train/act_aloha_transfer6/checkpoints/003000/pretrained_model")
    policy = ACTPolicy.from_pretrained(pretrained_policy_path)

#policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)

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
    policy.reset() #options={'box_size':[0.02,0.02,0.02]})
    numpy_observation, info = env.reset(seed=41) #,options={'box_size':[0.02,0.02,0.02],'box_color':[0,1,0,1]})
else:
    policy.reset()
    numpy_observation, info = env.reset(seed=1000) #seed=40 #,options={'box_color':[1,0,0,.025]}) #seed=1000,options={'box_size':[0.02,0.02,0.02],'box_color':[0,0,1,1]}) #41)

if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
    ts=dm_env.TimeStep(step_type=dm_env.StepType.FIRST,reward=None,discount=None,observation=info['raw_obs'])
    print(f"cube position={ts.observation['env_state'][:3]}")
print(f"cube pos={info['raw_obs']['env_state']}")

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

step = 0
done = False
while not done:
    
    for i in range(len(cam_list)): #anr added
        plt_imgs[i].set_data(numpy_observation['pixels'][cam_list[i]])
    plt.pause(0.02)

    #if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube':
    #   numpy_observation["agent_pos"] = numpy.delete(numpy_observation["agent_pos"], [7, 15])
    
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

    #if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube':
    #    temp_array = numpy.insert(numpy_action, 7, numpy_action[6])
    #    numpy_action = numpy.insert(temp_array, 15, numpy_action[13])

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
