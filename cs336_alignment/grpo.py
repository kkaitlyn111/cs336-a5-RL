import torch
import torch.optim as optim
import json
import random
import wandb
import argparse
from pathlib import Path
from vllm import SamplingParams
from cs336_alignment.helpers import init_vllm, load_policy_into_vllm_instance, tokenize_prompt_and_output, get_response_log_probs, gradient_clipping, model_eval
from cs336_alignment.helpers import masked_normalize
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from transformers import AutoModelForCausalLM, AutoTokenizer

'''
Args:
    rollout_responses: list[str] of length n_prompts_per_rollout_batch * group_size
    repeated_ground_truths: list[str] of length n_prompts_per_rollout_batch * group_size
    group_size
    advatange_eps: 
    normalize_by_std

Returns:
    advantages: (rollout_batch_size,)
    raw_rewards (rollout_batch_size,)
    metadata: e.g. mean, std, max/min of rewards
'''
def compute_group_normalized_rewards(reward_fn, rollout_responses, repeated_ground_truths, group_size, advantage_eps, normalize_by_std, device):
    rewards = [reward_fn(rollout_response, ground_truth) for rollout_response, ground_truth in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor([reward["answer_reward"] for reward in rewards], device=device)
    rewards_grouped = raw_rewards.reshape(-1, group_size)
    if normalize_by_std:
        advantages_grouped = (rewards_grouped - rewards_grouped.mean(dim=-1, keepdim=True)) / (rewards_grouped.std(dim=-1, keepdim=True) + advantage_eps)
    else:
        advantages_grouped = rewards_grouped - rewards_grouped.mean(dim=-1, keepdim=True)
    advantages = advantages_grouped.flatten()
    metadata = {}
    return advantages, raw_rewards, metadata

'''
Args:
    raw_rewards_or_advantages: (batch_size, 1)
    policy_log_probs: (batch_size, sequence_length)
Returns:
    torch.Tensor: (batch_size, sequence_length)
'''
def compute_naive_policy_gradient_loss(raw_rewards_or_advantages: torch.Tensor, policy_log_probs: torch.Tensor) -> torch.Tensor:
    return -raw_rewards_or_advantages * policy_log_probs


'''
Args:
    advantages: (batch_size, 1)
    policy_log_probs: (batch_size, sequence_length)
    old_log_probs: (batch_size, sequence_length)
    cliprange: 0.2
Return:
    loss: (batch_size, sequence_length)
    metadata["clipped"]: (batch_size, sequence_length)
'''
def compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    importance_weight = torch.exp(policy_log_probs - old_log_probs)
    print(f"importance_weight.shape: {importance_weight.shape}")
    clipped_importance_weight = torch.clip(importance_weight, 1-cliprange, 1+cliprange)
    print(f"clipped_importance_weight.shape: {clipped_importance_weight.shape}")
    loss = -torch.minimum(importance_weight*advantages, clipped_importance_weight*advantages)
    clipped = clipped_importance_weight*advantages < importance_weight*advantages
    print(f"loss.shape: {loss.shape}")
    metadata = {"clipped": clipped}
    return loss, metadata

def compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange):
    assert loss_type in ["no_baseline", "reinforce_with_baseline", "grpo_clip"]
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
        return loss, metadata
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
        return loss, metadata
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    
def masked_mean(tensor, mask, dim):
    tensor_masked = tensor.masked_fill(~mask, 0)
    if dim is None:
        return torch.sum(tensor_masked) / torch.sum(mask)
    else:
        return torch.sum(tensor_masked, dim=dim) / torch.sum(mask, dim=dim)
    
def grpo_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, loss_type, raw_rewards, advantages, old_log_probs, cliprange):
    old_log_probs = old_log_probs.detach()
    token_losses, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    loss = masked_mean(token_losses, response_mask, dim=None)
    loss /= gradient_accumulation_steps
    loss.backward()
    return loss, metadata




def train_grpo(cfg):
    n_grpo_steps = cfg["n_grpo_steps"]
    lr = cfg["lr"]
    advantage_eps = cfg["advantage_eps"]
    rollout_batch_size = cfg["rollout_batch_size"]
    group_size = cfg["group_size"]
    sampling_temperature = cfg["sampling_temperature"]
    sampling_min_tokens = cfg["sampling_min_tokens"]
    sampling_max_tokens = cfg["sampling_max_tokens"]
    epochs_per_rollout_batch = cfg["epochs_per_rollout_batch"]
    train_batch_size = cfg["train_batch_size"]
    gradient_accumulation_steps = cfg["gradient_accumulation_steps"]
    cliprange = cfg["cliprange"]
    grpo_eval_freq = cfg["grpo_eval_freq"]
    grpo_num_eval_samples = cfg["grpo_num_eval_samples"]
    loss_type = cfg["loss_type"]
    use_std_normalization = cfg["use_std_normalization"]
    seed = cfg["seed"]

    device_model = cfg["device_model"]
    device_eval = cfg["device_eval"]
    model_id = cfg["model_id"]
    dataset_train_fpath = cfg["dataset_train_fpath"]
    prompt_fpath = cfg["prompt_fpath"]
    random.seed(seed)
    torch.manual_seed(seed)

    assert train_batch_size % gradient_accumulation_steps == 0, "train_batch_size must be divisible by gradient_accumulation_steps"
    train_microbatch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, "train_batch_size must be greater than or equal to group_size"
    on_policy = epochs_per_rollout_batch == 1 and train_batch_size == rollout_batch_size
    assert not (loss_type == "grpo_clip" and on_policy)
    
    #initialize wandb
    run_name = f"qwen_GRPO_lr:{lr}"
    wandb.init(
        entity="kaitwang",
        project="cs336_alignment_A5_GRPO",
        name=run_name,
        mode="offline",
        config=cfg,
    )
    wandb_step = 0

    #initialize model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device_model)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vllm_model = init_vllm(model_id, device_eval, seed)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))

    #initialize dataset
    dataset_train = []
    with open(dataset_train_fpath, "r") as file:
        for line in file:
            dataset_train.append(json.loads(line))
    
    #load prompt template
    with open(prompt_fpath, "r") as file:
        prompt_template = file.read()

    for grpo_step in range(n_grpo_steps):
        rollout_dataset = random.sample(dataset_train, n_prompts_per_rollout_batch)
        rollout_questions = [data["problem"] for data in rollout_dataset]
        rollout_prompts = [prompt_template.format(question=question) for question in rollout_questions]
        rollout_answers = [data["answer"] for data in rollout_dataset]

        sampling_params = SamplingParams(temperature=sampling_temperature, top_p=1.0, min_tokens=sampling_min_tokens, max_tokens=sampling_max_tokens, stop=["</answer>"], include_stop_str_in_output=True, n=group_size, seed=seed)
        outputs = vllm_model.generate(rollout_prompts, sampling_params)

        repeated_answers = []
        generations = []
        prompts = []
        for output, answer in zip(outputs, rollout_answers):
            prompt = output.prompt
            for i in range(group_size):
                generated_text = output.outputs[i].text
                prompts.append(prompt)
                generations.append(generated_text)
                repeated_answers.append(answer)
        tokenizations = tokenize_prompt_and_output(prompts, generations, tokenizer, device_model)
        input_ids, labels, response_mask = tokenizations["input_ids"], tokenizations["labels"], tokenizations["response_mask"]
        tokenizations_train = tokenizations

        advantages_train, raw_rewards_train, metadata = compute_group_normalized_rewards(r1_zero_reward_fn, rollout_responses=generations, repeated_ground_truths=repeated_answers, group_size=group_size, advantage_eps=advantage_eps, normalize_by_std=use_std_normalization, device=device_model)
        
        #compute the old_log_probs over the entire dataset (split into microbatches), if off policy
        num_train_steps_per_epoch = rollout_batch_size // train_batch_size
        with torch.no_grad():
            old_log_probs_train = []
            for train_step in range(num_train_steps_per_epoch):
                batch_idxs = train_step*train_batch_size, train_step*train_batch_size+train_batch_size
                for train_microstep in range(gradient_accumulation_steps):
                    microbatch_idxs = batch_idxs[0] + train_microstep*train_microbatch_size, batch_idxs[0] + train_microstep*train_microbatch_size+train_microbatch_size
                    # print(f"train_step = {train_step}, train_microstep = {train_microstep}: microbatch_idxs = {microbatch_idxs}")
                    # print(f"train_step = {train_step}, train_microstep = {train_microstep}: tokenizations.shape = {tokenizations}")
                    input_ids, labels, response_mask = tokenizations["input_ids"], tokenizations["labels"], tokenizations["response_mask"]
                    input_ids, labels, response_mask = input_ids[microbatch_idxs[0]:microbatch_idxs[1]], labels[microbatch_idxs[0]:microbatch_idxs[1]], response_mask[microbatch_idxs[0]:microbatch_idxs[1]]
                    log_probs_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
                    log_probs, token_entropy = log_probs_dict["log_probs"], log_probs_dict["token_entropy"]
                    old_log_probs_train.append(log_probs)
                    assert log_probs.shape[0] == microbatch_idxs[1]-microbatch_idxs[0]
            old_log_probs_train = torch.cat(old_log_probs_train)
        print(f"Grpo step {grpo_step}: Completed computing log probs on the old model, old_log_probs_train.shape = {old_log_probs_train.shape}")
            
        #train for epochs_per_rollout_batch on the same exact data
        for train_epoch in range(epochs_per_rollout_batch):
            for train_step in range(num_train_steps_per_epoch):
                batch_idxs = train_step*train_batch_size, train_step*train_batch_size+train_batch_size
                for train_microstep in range(gradient_accumulation_steps):
                    microbatch_idxs = batch_idxs[0] + train_microstep*train_microbatch_size, batch_idxs[0] + train_microstep*train_microbatch_size+train_microbatch_size
                    advantages = advantages_train[microbatch_idxs[0]:microbatch_idxs[1]].unsqueeze(-1)
                    raw_rewards = raw_rewards_train[microbatch_idxs[0]:microbatch_idxs[1]].unsqueeze(-1)
                    old_log_probs = old_log_probs_train[microbatch_idxs[0]:microbatch_idxs[1]]
                    input_ids, labels, response_mask = tokenizations_train["input_ids"], tokenizations_train["labels"], tokenizations_train["response_mask"]
                    input_ids, labels, response_mask = input_ids[microbatch_idxs[0]:microbatch_idxs[1]], labels[microbatch_idxs[0]:microbatch_idxs[1]], response_mask[microbatch_idxs[0]:microbatch_idxs[1]]

                    #compute new log_probs
                    log_probs_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
                    log_probs, token_entropy = log_probs_dict["log_probs"], log_probs_dict["token_entropy"]
                    policy_log_probs = log_probs

                    loss, metadata = grpo_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
                    print(f"TRAIN: Grpo step {grpo_step}, train epoch {train_epoch}, train_step {train_step}, microbatch_train_step: {train_microstep}: train loss = {loss}")
                    avg_token_entropy = masked_mean(token_entropy, response_mask, dim=None)
                    wandb.log({"train/train_loss": loss, "train/train_entropy": avg_token_entropy}, step=wandb_step)
                    if loss_type == "grpo_clip":
                        clipped_fraction = masked_mean(metadata["clipped"], response_mask, dim=None)
                        wandb.log({"train/clip_fraction": clipped_fraction}, step=wandb_step)
                    wandb_step += 1
                gradient_clipping(model)
                optimizer.step()
                optimizer.zero_grad()
            
        load_policy_into_vllm_instance(model, vllm_model)  
        if grpo_step % grpo_eval_freq == 0:
            format_acc, answer_acc = model_eval(vllm_model, grpo_num_eval_samples)
            print(f"VALIDATION: Grpo step {grpo_step}, train epoch {train_epoch}, train_step {train_step}: format_acc = {format_acc}, answer_acc = {answer_acc}")
            wandb.log({"eval/format_acc": format_acc, "eval/answer_acc": answer_acc})

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training of Qwen on MATH")

    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="Model ID or path to the model")
    parser.add_argument("--dataset_train_fpath", type=str, default="/home/user/cs336-a5-RL/MATH/train.jsonl", help="Path to training dataset")
    parser.add_argument("--prompt_fpath", type=str, default="/home/user/cs336-a5-RL/prompts/r1_zero.prompt", help="Path to prompt template")
    parser.add_argument("--n_grpo_steps", type=int, default=50, help="Number of GRPO training steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--advantage_eps", type=float, default=1e-6, help="Epsilon for advantage normalization")
    parser.add_argument("--rollout_batch_size", type=int, default=256, help="Batch size for rollouts")
    parser.add_argument("--group_size", type=int, default=8, help="Number of samples per group for advantage computation")
    parser.add_argument("--sampling_temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--sampling_min_tokens", type=int, default=4, help="Minimum tokens for sampling")
    parser.add_argument("--sampling_max_tokens", type=int, default=1024, help="Maximum tokens for sampling")
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1, help="Number of epochs per rollout batch")
    parser.add_argument("--train_batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128, help="Number of gradient accumulation steps")
    parser.add_argument("--cliprange", type=float, default=0.2, help="Clipping range for GRPO")
    parser.add_argument("--grpo_eval_freq", type=int, default=2, help="Evaluation frequency (every N steps)")
    parser.add_argument("--grpo_num_eval_samples", type=int, default=200, help="Number of samples for evaluation")
    parser.add_argument("--loss_type", type=str, default="reinforce_with_baseline", choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"], help="Type of loss function to use")
    parser.add_argument("--use_std_normalization", action="store_true", default=True, help="Whether to use standard deviation normalization for advantages")
    parser.add_argument("--device_model", type=str, default="cuda:0", help="Device for model training")
    parser.add_argument("--device_eval", type=str, default="cuda:1", help="Device for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    cfg = {
        "model_id": args.model_id,
        "dataset_train_fpath": Path(args.dataset_train_fpath),
        "prompt_fpath": Path(args.prompt_fpath),
        "n_grpo_steps": args.n_grpo_steps,
        "lr": args.lr,
        "advantage_eps": args.advantage_eps,
        "rollout_batch_size": args.rollout_batch_size,
        "group_size": args.group_size,
        "sampling_temperature": args.sampling_temperature,
        "sampling_min_tokens": args.sampling_min_tokens,
        "sampling_max_tokens": args.sampling_max_tokens,
        "epochs_per_rollout_batch": args.epochs_per_rollout_batch,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "cliprange": args.cliprange,
        "grpo_eval_freq": args.grpo_eval_freq,
        "grpo_num_eval_samples": args.grpo_num_eval_samples,
        "loss_type": args.loss_type,
        "use_std_normalization": args.use_std_normalization,
        "device_model": args.device_model,
        "device_eval": args.device_eval,
        "seed": args.seed
    }
    
    print("Configuration:", flush=True)
    for key, value in cfg.items():
        print(f"{key}: {value}", flush=True)
    
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train_grpo(cfg)