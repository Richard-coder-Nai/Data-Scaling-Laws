# UMI Client for OneTwoVLA

This repository provides code for running the [OneTwoVLA](https://github.com/Richard-coder-Nai/onetwovla.git) model on the Universal Manipulation Interface (UMI) platform.
We run inference using a **policy server** and a **UMI client**.

---

## Hardware Setup

Please follow the [UMI hardware setup instructions](https://github.com/real-stanford/universal_manipulation_interface).

**Note:**
- We remove the mirror and change the gripper design. See [details here](TODO).
- Gripper width and TCP offset are also updated in code ([reference commit](https://github.com/Richard-coder-Nai/Data-Scaling-Laws/commit/0661a3b1a8f2da833157933b0a886121a9676fde)).

---

## 1. Start the OneTwoVLA Policy Server

> Run these steps from your **OneTwoVLA** directory.

**Install dependencies:**

```bash
uv pip install pynput
```
> *You may need `sudo` privileges.*

**Start the policy server (example for port `8000`):**

```bash
uv run scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config=onetwovla_visual_grounding \
    --policy.dir=/path/to/your/checkpoint
```

**Available policy configurations:**
- `onetwovla_visual_cocktail`
- `onetwovla_visual_grounding`
- `pi0_visual_cocktail`
- `pi0_visual_grounding`

Update the `--policy.config` and `--policy.dir` to match your use case and checkpoint location.

---

## 2. Run the UMI Client

> Run these steps from your **Data-Scaling-Laws** directory.

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
   - Replace `<your instruction here>` with your task instruction. For example, for open-world visual grounding:

     ```bash
     --ins "Give me the object reminding me of the sea."
     ```
