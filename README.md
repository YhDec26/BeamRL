# BeamRL

BeamRL contains reinforcement learning utilities for tuning the transverse beam in a simulation of an electron accelerator. The training pipeline uses `stable-baselines3` together with a custom Gymnasium environment defined in `environement.py` and backends in `backend.py`.

## Dependencies

The project depends on Python 3 along with the following Python packages:

- `gymnasium`
- `stable-baselines3`
- `torch`
- `wandb`
- `numpy`
- `matplotlib`
- `cheetah`
- `ocelot`

Install them with pip:

```bash
pip install gymnasium stable-baselines3 torch wandb numpy matplotlib cheetah ocelot
```

## Training

Training is started with `train.py`:

```bash
export WANDB_API_KEY=YOUR_WANDB_KEY  # required for WandB logging
python BeamRL/train.py
```

The script automatically falls back to offline mode if WandB cannot connect to the internet.

## Evaluation

Evaluation of a trained model is handled by `eval_sim_new.py`:

1. Edit the `MODEL_PATH` variable at the top of `eval_sim_new.py` so that it points to the directory containing your saved model and configuration.
2. Run the script:

```bash
python BeamRL/eval_sim_new.py
```

The script will load the PPO model and run several evaluation episodes using the same environment configuration as during training.
