import torch
import torch.optim as optim
import json
import random
from vllm import SamplingParams
from cs336_alignment.helpers import init_vllm, load_policy_into_vllm_instance, tokenize_prompt_and_output, get_response_log_probs, gradient_clipping, model_eval, masked_normalize
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
def compute_group_normalized_rewards(reward_fn, rollout_responses, repeated_ground_truths, group_size, advantage_eps, normalize_by_std):
    rewards = [reward_fn(rollout_response, ground_truth) for rollout_response, ground_truth in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor([reward["answer_reward"] for reward in rewards])
    rewards_grouped = raw_rewards.reshape(-1, group_size)
    if normalize_by_std:
        advatanges_grouped = (rewards_grouped - rewards_grouped.mean(dim=-1, keepdim=True)) / (rewards_grouped.std(dim=-1, keepdim=True) + advantage_eps)
    else:
        advatanges_grouped = rewards_grouped - rewards_grouped.mean(dim=-1, keepdim=True)
    advantages = advatanges_grouped.flatten()
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




def train_grpo():
    n_grpo_steps = 200
    lr = 1e-5
    advantage_eps = 1e-6
    rollout_batch_size = 256
    group_size = 8
    sampling_temperature = 1.0
    sampling_min_tokens = 4
    sampling_max_tokens = 1024
    n_train_steps_per_rollout_batch = 1
    train_batch_size = 256
    gradient_accumulation_steps = 128
    cliprange = 0.2
    grpo_eval_freq = 2
    grpo_num_eval_samples = 200
    loss_type = "reinforce_with_baseline"
    use_std_normalization = True
    seed = 42

    device_model = "cuda:0"
    device_eval = "cuda:1"
    model_id = "models_local/Qwen2.5-Math-1.5B"
    dataset_train_fpath = "/home/user/cs336-a5-RL/MATH/train.jsonl"
    prompt_fpath = "/home/user/cs336-a5-RL/cs336_alignment/prompts/r1_zero.prompt"

    assert train_batch_size % gradient_accumulation_steps == 0, "train_batch_size must be divisible by gradient_accumulation_steps"
    train_microbatch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, "train_batch_size must be greater than or equal to group_size"
    
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

        sampling_params = SamplingParams(temperature=sampling_temperature, top_p=1.0, min_tokens=sampling_min_tokens, max_tokens=sampling_max_tokens, n=group_size, seed=seed)
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
        log_probs_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
        log_probs, token_entropy = log_probs_dict["log_probs"], log_probs_dict["token_entropy"]
        
        advantages_train, raw_rewards_train, metadata = compute_group_normalized_rewards(r1_zero_reward_fn, rollout_responses=generations, repeated_ground_truths=repeated_answers, group_size=group_size, advantage_eps=advantage_eps, normalize_by_std=use_std_normalization)
        old_log_probs_train = log_probs
        tokenizations_train = tokenizations
        
        
        for train_step in range(n_train_steps_per_rollout_batch):
            batch_idxs = train_step*train_batch_size, train_step*train_batch_size+train_batch_size
            for train_microstep in range(gradient_accumulation_steps):
                microbatch_idxs = batch_idxs[0] + train_microstep*train_microbatch_size, batch_idxs[0] + train_microstep*train_microbatch_size+train_microbatch_size
                advantages = advantages_train[microbatch_idxs[0]:microbatch_idxs[1]]
                raw_rewards = raw_rewards_train[microbatch_idxs[0]:microbatch_idxs[1]]
                old_log_probs = old_log_probs_train[microbatch_idxs[0]:microbatch_idxs[1]]
                tokenizations = tokenizations_train[microbatch_idxs[0]:microbatch_idxs[1]]

                #compute new log_probs
                log_probs_dict = get_response_log_probs(model, tokenizations["input_ids"], tokenizations["labels"], tokenizations["response_mask"], return_token_entropy=True)
                log_probs, token_entropy = log_probs_dict["log_probs"], log_probs_dict["token_entropy"]
                policy_log_probs = log_probs
                response_mask = tokenizations["response_mask"]

                loss, metadata = grpo_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
                print(f"TRAIN: Grpo step {grpo_step}, train_step {train_step}, microbatch_train_step: {train_microstep}: train loss = {loss}")
            gradient_clipping(model)
            optimizer.step()
            optimizer.zero_grad()
            
        load_policy_into_vllm_instance(model, vllm_model)  
        if grpo_step % grpo_eval_freq == 0:
            format_acc, answer_acc = model_eval(vllm_model, grpo_num_eval_samples)
            print(f"VALIDATION: Grpo step {grpo_step}, train_step {train_step}: format_acc = {format_acc}, answer_acc = {answer_acc}")


if __name__ == "__main__":
    train_grpo()