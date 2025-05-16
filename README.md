# UMI Client for OneTwoVLA

This repository provides the hardware client code for running the [OneTwoVLA](https://one-two-vla.github.io/) model on the Universal Manipulation Interface (UMI) platform. The training code for OneTwoVLA can be found at (link)[insert link here], and the policy server code is available at (link)[insert link here].

---

## Hardware Setup

Please follow the [UMI hardware setup instructions](https://github.com/real-stanford/universal_manipulation_interface).

**Note:**
- We remove the mirror and change the gripper design. See [details here](https://drive.google.com/drive/folders/19Mh4s9g5-ohd3_6ZhqTk2WIT-KeQDnH7?usp=drive_link).
- TCP offset is updated in code ([reference commit](https://github.com/Richard-coder-Nai/Data-Scaling-Laws/commit/0661a3b1a8f2da833157933b0a886121a9676fde)).

---

## Run the UMI Client

The client script is [`umi_client.py`](umi_client.py).

**To launch the client:**

Activate your `umi` Python environment and start the client:

   ```bash
   python umi_client.py \
       --robot_config=example/eval_robots_config.yaml \
       -o /path/to/output/dir/ \
       --frequency 5 -j \
       --temporal_agg -si 1 \
       --ins "<your instruction here>" \
       --state_horizon 3,15 \
       -action_down_sample_steps 3 \
       -getitem_type necessary \
       --remote_port 8000
   ```
 Replace `<your instruction here>` with your task instruction. For example, for open-world visual grounding:

 ```bash
 --ins "Give me the object reminding me of the sea."
 ```
