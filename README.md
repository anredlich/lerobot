<!--
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="media/lerobot-logo-thumbnail.png">
    <source media="(prefers-color-scheme: light)" srcset="media/lerobot-logo-thumbnail.png">
    <img alt="LeRobot, Hugging Face Robotics Library" src="media/lerobot-logo-thumbnail.png" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/nightly-tests.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/nightly-tests.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/huggingface/lerobot/branch/main/graph/badge.svg?token=TODO)](https://codecov.io/gh/huggingface/lerobot)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Examples](https://img.shields.io/badge/Examples-green.svg)](https://github.com/huggingface/lerobot/tree/main/examples)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://dcbadge.vercel.app/api/server/C5P34WJ68S?style=flat)](https://discord.gg/s3KuuzsPFb)

</div>
-->

<!--
<h2 align="center">
    <p><a href="https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md">
        Build Your Own SO-100 Robot!</a></p>
</h2>
-->
<h2 align="center">
    <p>LeRobot</p>
</h2>

Note: this readme is under construction.  

Hardware: all simulation and training were performed on a System76 desktop with an Nvidia RTX 5090 with 32GB memory. Some simulations may not run on smaller GPUs or may require reducing batch sizes.

Contact: norman@anrrobot.com  

#### Examples of pretrained ACT models on the real Trossen AI Stationary robot:

(datasets and models available on huggingface in lerobot format, see below)

<img src="trossen_box_transfer_real_01.gif" width="70%" alt="TrossenAI Stationary TransferCube demo"/>

<img src="trossen_pop_lid_real_06.gif" width="70%" alt="TrossenAI Stationary TransferCube demo"/>

<img src="trossen_pour_box_real_05.gif" width="70%" alt="TrossenAI Stationary TransferCube demo"/>

This fork adds:

[Trossen AI Stationary robot](https://www.trossenrobotics.com/) sim to real and real to sim capability using the added TrossenAIStationary robot virtual environments in https://github.com/anredlich/gym-aloha.

Tasks supported so far:   
TrossenAIStationaryTransferCube-v0   
TrossenAIStationaryTransferCubeEE-v0 (EE=end effector)    

control_sim_robot.py:
creates simulated Trossen AI Stationary databases in standard lerobot format using the added scripted_policy.py adapted from https://github.com/TrossenRobotics/trossen_arm_mujoco.
replays simulated dataset episodes

train.py 
train policies with Trossen AI Stationary evals

eval.py
evaluate policies with Trossen AI Stationary env

new example files:
evaluate_pretrained_aloha_policy.py
evaluate_pretrained_trossen_ai_policy.py
train_aloha_policy.py
train_trossen_ai_policy.py (has bugs, use train.py for now)  

<br>

<!--
<div align="center">
  <img src="media/so100/leader_follower.webp?raw=true" alt="SO-100 leader and follower arms" title="SO-100 leader and follower arms" width="50%">

  <p><strong>Meet the SO-100 – Just $110 per arm!</strong></p>
  <p>Train it in minutes with a few simple moves on your laptop.</p>
  <p>Then sit back and watch your creation act autonomously! 🤯</p>

  <p><a href="https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md">
      Get the full SO-100 tutorial here.</a></p>

  <p>Want to take it to the next level? Make your SO-100 mobile by building LeKiwi!</p>
  <p>Check out the <a href="https://github.com/huggingface/lerobot/blob/main/examples/11_use_lekiwi.md">LeKiwi tutorial</a> and bring your robot to life on wheels.</p>

</div>
-->

#### Example of pretrained ACT model on simulated Trossen AI Stationary environment

<img src="trossen_ai_stationary_transfer_cube.gif" width="70%" alt="TrossenAI Stationary TransferCube demo"/>

<!--
  <img src="media/lekiwi/kiwi.webp?raw=true" alt="LeKiwi mobile robot" title="LeKiwi mobile robot" width="50%">
-->



<br/>

<!--
<h3 align="center">
    <p>LeRobot: State-of-the-art AI for real-world robotics</p>
</h3>

---


🤗 LeRobot aims to provide models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier to entry to robotics so that everyone can contribute and benefit from sharing datasets and pretrained models.

🤗 LeRobot contains state-of-the-art approaches that have been shown to transfer to the real-world with a focus on imitation learning and reinforcement learning.

🤗 LeRobot already provides a set of pretrained models, datasets with human collected demonstrations, and simulation environments to get started without assembling a robot. In the coming weeks, the plan is to add more and more support for real-world robotics on the most affordable and capable robots out there.

🤗 LeRobot hosts pretrained models and datasets on this Hugging Face community page: [huggingface.co/lerobot](https://huggingface.co/lerobot)

#### Examples of pretrained models on simulation environments

<table>
  <tr>
    <td><img src="media/gym/aloha_act.gif" width="100%" alt="ACT policy on ALOHA env"/></td>
    <td><img src="media/gym/simxarm_tdmpc.gif" width="100%" alt="TDMPC policy on SimXArm env"/></td>
    <td><img src="media/gym/pusht_diffusion.gif" width="100%" alt="Diffusion policy on PushT env"/></td>
  </tr>
  <tr>
    <td align="center">ACT policy on ALOHA env</td>
    <td align="center">TDMPC policy on SimXArm env</td>
    <td align="center">Diffusion policy on PushT env</td>
  </tr>
</table>
-->

### Acknowledgment

This branch forked from https://github.com/Interbotix/lerobot and adapted from https://github.com/TrossenRobotics/trossen_arm_mujoco

See [Trossen AI Stationary robot](https://www.trossenrobotics.com/)

## Installation

Create a virtual environment with Python 3.10 and activate it:
```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg
```

Download and install this fork:
```bash
git clone https://github.com/anredlich/lerobot.git
cd lerobot
pip install --no-binary=av -e .
```

For simulation with gym-aloha (TrossenAI Stationary):
```bash
pip install git+https://github.com/anredlich/gym-aloha.git
```

For detailed installation help (build errors, ffmpeg, other simulations, Weights and Biases), see the [original HuggingFace LeRobot installation guide](https://github.com/huggingface/lerobot#installation).

## Walkthrough

added in this fork:   
control_sim_robot.py  
scripted_policy.py

```
.
├── examples             # contains demonstration examples, start here to learn about LeRobot
|   └── advanced         # contains even more examples for those who have mastered the basics
├── lerobot
|   ├── configs          # contains config classes with all options that you can override in the command line
|   ├── common           # contains classes and utilities
|   |   ├── datasets       # various datasets of human demonstrations: aloha, pusht, xarm
|   |   ├── envs           # various sim environments: aloha, pusht, xarm
|   |   ├── policies       # various policies: act, diffusion, tdmpc
|   |   ├── robot_devices  # various real devices: dynamixel motors, opencv cameras, koch robots
|   |   └── utils          # various utilities
|   └── scripts          # contains functions to execute via command line
|       ├── eval.py                 # load policy and evaluate it on an environment
|       ├── train.py                # train a policy via imitation learning and/or reinforcement learning
|       ├── control_robot.py        # teleoperate a real robot, record data, run a policy
|       ├── push_dataset_to_hub.py  # convert your dataset into LeRobot dataset format and upload it to the Hugging Face hub
|       └── visualize_dataset.py    # load a dataset and render its demonstrations
├── outputs               # contains results of scripts execution: logs, videos, model checkpoints
└── tests                 # contains pytest utilities for continuous integration
```

### Real and Simulated Trossen AI Stationary datasets

https://huggingface.co    

(visualize using visualize_dataset.py or at https://huggingface.co/spaces/lerobot/visualize_dataset)  

Real Robot:   
ANRedlich/trossen_ai_stationary_transfer_20mm_cube_01 
ANRedlich/trossen_ai_stationary_transfer_40mm_cube_02   
ANRedlich/trossen_ai_stationary_transfer_multi_cube_03   
ANRedlich/trossen_ai_stationary_place_lids_04   
ANRedlich/trossen_ai_stationary_pour_box_05   
ANRedlich/trossen_ai_stationary_pop_lid_06  

Simulated Robot:  
ANRedlich/trossen_ai_stationary_sim_transfer_40mm_cube_07 
ANRedlich/trossen_ai_stationary_sim_transfer_40mm_cube_08 
ANRedlich/trossen_ai_stationary_sim_transfer_40mm_cube_10   
ANRedlich/trossen_ai_stationary_sim_transfer_40mm_cube_13   

### Real and Simulated Trossen AI Stationary ACT models:

https://huggingface.co    

Real Robot: (see videos up top)   
ANRedlich/trossen_ai_stationary_real_act2_3   
ANRedlich/trossen_ai_stationary_real_act5   
ANRedlich/trossen_ai_stationary_real_act6   
(for real to sim try act_trossen_ai_stationary_real_02_3 in examples/evaluate_pretrained_trossen_ai_policy.py)   

Simulated Robot:    
(run in eval.py or examples/evaluate_pretrained_trossen_ai_policy.py)   
ANRedlich/trossen_ai_stationary_sim_act7   
ANRedlich/trossen_ai_stationary_sim_act8    
ANRedlich/trossen_ai_stationary_sim_act10   
ANRedlich/trossen_ai_stationary_sim_act13   
(... act13 is best sim to real policy, but very sensitive to conditions)  

### Visualize datasets

Check out [example 1](./examples/1_load_lerobot_dataset.py) that illustrates how to use our dataset class which automatically downloads data from the Hugging Face hub.

You can also locally visualize episodes from a dataset on the hub by executing our script from the command line:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id ANRedlich/trossen_ai_stationary_pour_box_05 \
    --episode-index 0
```

or from a dataset in a local folder with the `root` option:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id ANRedlich/trossen_ai_stationary_pour_box_05\
    --root ./my_local_data_dir \
    --local-files-only 1 \
    --episode-index 0
```


It will open `rerun.io` and display the camera streams, robot states and actions, like this:

https://github-production-user-asset-6210df.s3.amazonaws.com/4681518/328035972-fd46b787-b532-47e2-bb6f-fd536a55a7ed.mov?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240505%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240505T172924Z&X-Amz-Expires=300&X-Amz-Signature=d680b26c532eeaf80740f08af3320d22ad0b8a4e4da1bcc4f33142c15b509eda&X-Amz-SignedHeaders=host&actor_id=24889239&key_id=0&repo_id=748713144


Our script can also visualize datasets stored on a distant server. See `python lerobot/scripts/visualize_dataset.py --help` for more instructions.

### The `LeRobotDataset` format

A dataset in `LeRobotDataset` format is very simple to use. It can be loaded from a repository on the Hugging Face hub or a local folder simply with e.g. `dataset = LeRobotDataset("lerobot/aloha_static_coffee")` and can be indexed into like any Hugging Face and PyTorch dataset. For instance `dataset[0]` will retrieve a single temporal frame from the dataset containing observation(s) and an action as PyTorch tensors ready to be fed to a model.

A specificity of `LeRobotDataset` is that, rather than retrieving a single frame by its index, we can retrieve several frames based on their temporal relationship with the indexed frame, by setting `delta_timestamps` to a list of relative times with respect to the indexed frame. For example, with `delta_timestamps = {"observation.image": [-1, -0.5, -0.2, 0]}`  one can retrieve, for a given index, 4 frames: 3 "previous" frames 1 second, 0.5 seconds, and 0.2 seconds before the indexed frame, and the indexed frame itself (corresponding to the 0 entry). See example [1_load_lerobot_dataset.py](examples/1_load_lerobot_dataset.py) for more details on `delta_timestamps`.

Under the hood, the `LeRobotDataset` format makes use of several ways to serialize data which can be useful to understand if you plan to work more closely with this format. We tried to make a flexible yet simple dataset format that would cover most type of features and specificities present in reinforcement learning and robotics, in simulation and in real-world, with a focus on cameras and robot states but easily extended to other types of sensory inputs as long as they can be represented by a tensor.

Here are the important details and internal structure organization of a typical `LeRobotDataset` instantiated with `dataset = LeRobotDataset("lerobot/aloha_static_coffee")`. The exact features will change from dataset to dataset but not the main aspects:

```
dataset attributes:
  ├ hf_dataset: a Hugging Face dataset (backed by Arrow/parquet). Typical features example:
  │  ├ observation.images.cam_high (VideoFrame):
  │  │   VideoFrame = {'path': path to a mp4 video, 'timestamp' (float32): timestamp in the video}
  │  ├ observation.state (list of float32): position of an arm joints (for instance)
  │  ... (more observations)
  │  ├ action (list of float32): goal position of an arm joints (for instance)
  │  ├ episode_index (int64): index of the episode for this sample
  │  ├ frame_index (int64): index of the frame for this sample in the episode ; starts at 0 for each episode
  │  ├ timestamp (float32): timestamp in the episode
  │  ├ next.done (bool): indicates the end of en episode ; True for the last frame in each episode
  │  └ index (int64): general index in the whole dataset
  ├ episode_data_index: contains 2 tensors with the start and end indices of each episode
  │  ├ from (1D int64 tensor): first frame index for each episode — shape (num episodes,) starts with 0
  │  └ to: (1D int64 tensor): last frame index for each episode — shape (num episodes,)
  ├ stats: a dictionary of statistics (max, mean, min, std) for each feature in the dataset, for instance
  │  ├ observation.images.cam_high: {'max': tensor with same number of dimensions (e.g. `(c, 1, 1)` for images, `(c,)` for states), etc.}
  │  ...
  ├ info: a dictionary of metadata on the dataset
  │  ├ codebase_version (str): this is to keep track of the codebase version the dataset was created with
  │  ├ fps (float): frame per second the dataset is recorded/synchronized to
  │  ├ video (bool): indicates if frames are encoded in mp4 video files to save space or stored as png files
  │  └ encoding (dict): if video, this documents the main options that were used with ffmpeg to encode the videos
  ├ videos_dir (Path): where the mp4 videos or png images are stored/accessed
  └ camera_keys (list of string): the keys to access camera features in the item returned by the dataset (e.g. `["observation.images.cam_high", ...]`)
```

A `LeRobotDataset` is serialised using several widespread file formats for each of its parts, namely:
- hf_dataset stored using Hugging Face datasets library serialization to parquet
- videos are stored in mp4 format to save space
- metadata are stored in plain json/jsonl files

Dataset can be uploaded/downloaded from the HuggingFace hub seamlessly. To work on a local dataset, you can specify its location with the `root` argument if it's not in the default `~/.cache/huggingface/lerobot` location.

### Creating a simulated robot dataset

```bash
python lerobot/scripts/control_sim_robot.py \
    --robot.type=trossen_ai_stationary \
    --env.type=aloha \
    --env.task=gym_aloha/TrossenAIStationaryTransferCubeEE-v0 \
    --env.box_size=[0.02,0.02,0.02] \
    --env.box_color=[1,0,0,1] \
    --env.tabletop=my_desktop \
    --control.type=record \
    --control.fps=30 \
    --control.single_task='Recording evaluation episode using Trossen AI Stationary.'  \
    --control.repo_id=ANRedlich/eval_act_trossen_ai_stationary_test_100 \
    --control.root=lerobot/scripts/dataset/eval100 \
    --control.tags=[\"tutorial\"] \
    --control.warmup_time_s=5 \
    --control.episode_time_s=30 \
    --control.reset_time_s=30 \
    --control.num_episodes=100 \
    --control.push_to_hub=false \
    --control.policy.path=scripted_policy
```
This will create a local dataset in the root directory. set push_to_hub=true to send to huggingface instead.

### Creating a real robot dataset

see [Trossen AI Stationary robot](https://www.trossenrobotics.com/)

### Evaluate a pretrained policy

```bash
python lerobot/scripts/eval.py \
    --policy.path=ANRedlich/trossen_ai_stationary_sim_act7 \
    --env.type=aloha \
    --env.episode_length=500 \
    --env.task=TrossenAIStationaryTransferCube-v0 \
    --eval.n_episodes=50 \
    --eval.batch_size=50 \
    --env.box_size=[0.02,0.02,0.02] \
    --policy.use_amp=false \
    --policy.device=cuda
```
Above should have a success_rate ~=86.0%.   

Or run:   
```bash
python examples/evaluate_pretrained_trossen_ai_policy.py
```

For the env variables used to learn other simulated models, look inside evaluate_pretrained_trossen_ai_policy.py.

Note: After training your own policy, you can re-evaluate the checkpoints with:

```bash
python lerobot/scripts/eval.py --policy.path={OUTPUT_DIR}/checkpoints/last/pretrained_model
```

See `python lerobot/scripts/eval.py --help` for more instructions.

### Train your own policy

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=ANRedlich/trossen_ai_stationary_sim_transfer_40mm_cube_07 \
    --output_dir=lerobot/scripts/outputs/train/trossen_ai_stationary_act7_3 \
    --job_name=trossen_ai_stationary_act7_3 \
    --policy.type=act \
    --policy.device=cuda \
    --env.type=aloha \
    --env.episode_length=500 \
    --env.task=TrossenAIStationaryTransferCube-v0 \
    --eval.n_episodes=50 \
    --eval.batch_size=50 \
    --env.box_size=[0.02,0.02,0.02] \
    --steps=100000 \
    --save_freq=10000 \
    --eval_freq=10000
```

Then use eval.py or evaluate_pretrained_trossen_ai_policy.py with:    
--policy.path=lerobot/scripts/outputs/train/trossen_ai_stationary_act7_3/checkpoints/last/pretrained_model    
or    
pretrained_policy_path=Path("lerobot/scripts/outputs/train/trossen_ai_stationary_act7_3/checkpoints/last/pretrained_model")    

For other examples, look inside evaluate_pretrained_trossen_ai_policy.py for the environmental variables corresonding to act8 <-> repo_id=...8

Also, try for the original aloha:

Run:
```bash
python examples/train_aloha_policy.py
```
Note that 
```bash
python examples/train_trossen_ai_policy.py
```
seems to have a bug, so it is not reliable. For now use train.py which is faster and more robust anyway.

### Add a pretrained policy

Once you have trained a policy you may upload it to the Hugging Face hub using a hub id that looks like `${hf_user}/${repo_name}` (e.g. [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht)).

You first need to find the checkpoint folder located inside your experiment directory (e.g. `outputs/train/2024-05-05/20-21-12_aloha_act_default/checkpoints/002500`). Within that there is a `pretrained_model` directory which should contain:
- `config.json`: A serialized version of the policy configuration (following the policy's dataclass config).
- `model.safetensors`: A set of `torch.nn.Module` parameters, saved in [Hugging Face Safetensors](https://huggingface.co/docs/safetensors/index) format.
- `train_config.json`: A consolidated configuration containing all parameter userd for training. The policy configuration should match `config.json` exactly. Thisis useful for anyone who wants to evaluate your policy or for reproducibility.

To upload these to the hub, run the following:
```bash
huggingface-cli upload ${hf_user}/${repo_name} path/to/pretrained_model
```

See [eval.py](https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/eval.py) for an example of how other people may use your policy.

## Citation

If you want, you can cite this work with:
```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

Additionally, if you are using any of the particular policy architecture, pretrained models, or datasets, it is recommended to cite the original authors of the work as they appear below:

- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu)
```bibtex
@article{chi2024diffusionpolicy,
	author = {Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
	title ={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
	journal = {The International Journal of Robotics Research},
	year = {2024},
}
```
- [ACT or ALOHA](https://tonyzhaozh.github.io/aloha)
```bibtex
@article{zhao2023learning,
  title={Learning fine-grained bimanual manipulation with low-cost hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}
```

- [TDMPC](https://www.nicklashansen.com/td-mpc/)

```bibtex
@inproceedings{Hansen2022tdmpc,
	title={Temporal Difference Learning for Model Predictive Control},
	author={Nicklas Hansen and Xiaolong Wang and Hao Su},
	booktitle={ICML},
	year={2022}
}
```

- [VQ-BeT](https://sjlee.cc/vq-bet/)
```bibtex
@article{lee2024behavior,
  title={Behavior generation with latent actions},
  author={Lee, Seungjae and Wang, Yibin and Etukuru, Haritheja and Kim, H Jin and Shafiullah, Nur Muhammad Mahi and Pinto, Lerrel},
  journal={arXiv preprint arXiv:2403.03181},
  year={2024}
}
```
