import numpy as np
import subprocess
from time import sleep

#UNCOMMENT THIS FOR BATCH SIZE SWEEP
# params = {
#     "n_grpo_steps": [10] * 10 + [100] * 10,
#     "lr": [1e-1, 1e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 1e-7] * 2
# }
# params = {
#     "n_grpo_steps": [10000],
#     "grpo_num_eval_samples": [5000],
#     "lr": [5e-5]

# }
params = {
    "loss_type": ["no_baseline"],
    "lr": [5e-5],
}
params = {
    "loss_type": ["reinforce_with_baseline"],
    "lr": [5e-5],
    "use_constant_length_normalization": [1],
}
params = {
    "loss_type": ["reinforce_with_baseline"] * 4,
    "lr": [5e-5] * 4,
    "use_constant_length_normalization": [1, 0, 1, 0],
    "use_std_normalization": [0, 0, 1, 1],
}
params = {
    "loss_type": ["reinforce_with_baseline"],
    "lr": [5e-5],
    "use_constant_length_normalization": [1],
    "use_std_normalization": [1],
}

#now we are testing off_policy
off_policy = 1
loss_type = "grpo_clip"
lr = 5e-5
use_constant_length_normalization = 1
use_std_normalization = 1
rollout_batch_size = 256
epochs_per_rollout_batch = [1, 2, 5, 10, 25]
train_batch_size = [256, 128, 64, 32]
gradient_accumulation_steps = [x//2 for x in train_batch_size]

N = len(epochs_per_rollout_batch) * len(train_batch_size)

params = {
    "off_policy": [off_policy] * N,
    "loss_type": [loss_type] * N,
    "lr": [lr] * N,
    "use_constant_length_normalization": [use_constant_length_normalization] * N,
    "use_std_normalization": [use_std_normalization] * N,
    "rollout_batch_size": [rollout_batch_size] * N,
    "epochs_per_rollout_batch": [epoch for epoch in epochs_per_rollout_batch for _ in train_batch_size],
    "train_batch_size": [batch for _ in epochs_per_rollout_batch for batch in train_batch_size],
    "gradient_accumulation_steps": [grad_acc for _ in epochs_per_rollout_batch for grad_acc in gradient_accumulation_steps]
}
N = len(list(params.values())[0])
print(f"N={N}")

# N = len(next(iter(params.values())))
for i in range(N):
    args = []
    for k, v in params.items():
        val = v[i]
        if val is None:
            continue
        if isinstance(val, bool):
            if val:
                args.append(f"--{k}")
        else:
            args.append(f"--{k} {val}")
    subprocess.run(["sbatch", "run_grpo.sh"] + args)
    sleep(0.1)