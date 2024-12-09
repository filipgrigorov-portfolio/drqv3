

# DrQ-v3: Sampled Data-Regularized Q-Learning

This is an original PyTorch implementation of DrQ-v3, based on https://github.com/facebookresearch/drqv2

<p align="center">
  <img width="19.5%" src="https://i.imgur.com/NzY7Pyv.gif">
  <img width="19.5%" src="https://imgur.com/O5Va3NY.gif">
  <img width="19.5%" src="https://imgur.com/PCOR9Mm.gif">
  <img width="19.5%" src="https://imgur.com/H0ab6tz.gif">
  <img width="19.5%" src="https://imgur.com/sDGgRos.gif">
  <img width="19.5%" src="https://imgur.com/gj3qo1X.gif">
  <img width="19.5%" src="https://imgur.com/FFzRwFt.gif">
  <img width="19.5%" src="https://imgur.com/W5BKyRL.gif">
  <img width="19.5%" src="https://imgur.com/qwOGfRQ.gif">
  <img width="19.5%" src="https://imgur.com/Uubf00R.gif">
 </p>

## Method
DrQ-v3 is a model-free off-policy algorithm for image-based continuous control. DrQ-v3 builds on [DrQ-v2](https://github.com/facebookresearch/drqv2), an actor-critic approach that uses data augmentation to learn directly from pixels. We introduce several extensions including:
- Two stage training regime
- Image sampling instead of data augmentation.
- A reconstruction auxiliary loss, in stage one and optional continual fine-tuning in stage two
- Pre-trained action policy regularization


## Instructions

Install [MuJoCo](http://www.mujoco.org/) if it is not already the case:

* Obtain a license on the [MuJoCo website](https://www.roboti.us/license.html).
* Download MuJoCo binaries [here](https://www.roboti.us/index.html).
* Unzip the downloaded archive into `~/.mujoco/mujoco200` and place your license key file `mjkey.txt` at `~/.mujoco`.
* Use the env variables `MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MUJOCO_PATH` to specify the MuJoCo license key path and the MuJoCo directory path.
* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.

Install the following libraries:
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

Install dependencies:
```sh
conda env create -f conda_env.yml
conda activate drqv2
```

Train the agent:
```sh
python train.py task=quadruped_walk
```

or

```
./run_task.sh
```

Notes:

`STAGE_1`: Enables stage one mode

`STAGE_2`: Enables stage two mode

`LATENT_DYNAMICS`: Enables latent dynamics training (only if resnet encoders used)

`WITH_RECONSTRUCTION`: Turns on/off cVAE training with the states in stage one

`TRAIN_STAGE_2`: Tunrs on/off fine-tuning in stage two

`REGULARIZE_ACTOR`: Turns on/off actor policy regularization in stage two

`SHUFFLE`: Turns on/off shuffling in stage two

Monitor results:
```sh
tensorboard --logdir exp_local
```

## License
The majority of DrQ-v3 is licensed under the MIT license, however portions of the project are available under separate license terms: DeepMind is licensed under the Apache 2.0 license.
