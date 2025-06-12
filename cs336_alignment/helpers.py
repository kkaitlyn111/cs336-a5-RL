import torch
import json
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import random
import pickle
from unittest.mock import patch
from transformers import PreTrainedModel

def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer, device) -> dict[str, torch.Tensor]:
    batch_size = len(prompt_strs)
    # print(f"prompt_strs: {prompt_strs}")
    # print(f"output_strs: {output_strs}")
    prompt_tokens = [tokenizer.encode(prompt) for prompt in prompt_strs]
    output_tokens = [tokenizer.encode(output) for output in output_strs]
    # print(f"prompt_tokens: {prompt_tokens}")
    # print(f"output_tokens: {output_tokens}")
    prompt_and_output_lens = [len(ptokens) + len(otokens) for ptokens, otokens in zip(prompt_tokens, output_tokens)]
    # print(f"prompt_and_output_lens: {prompt_and_output_lens}")
    max_len = max(prompt_and_output_lens)

    input_ids = torch.full((batch_size, max_len), tokenizer.pad_token_id, device=device)
    labels = torch.full((batch_size, max_len), tokenizer.pad_token_id, device=device)
    response_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    for i in range(batch_size):
        ptokens, otokens, potokens = prompt_tokens[i], output_tokens[i], prompt_tokens[i]+output_tokens[i]
        n_ptokens, n_otokens, n_potokens = len(ptokens), len(otokens), len(potokens)
        input_ids[i, :n_potokens] = torch.tensor(potokens)
        labels[i, :n_potokens-1] = torch.tensor(potokens[1:])
        response_mask[i, n_ptokens-1:n_potokens-1] = True
    
    input_ids = input_ids[:, :-1]
    labels = labels[:, :-1]
    response_mask = response_mask[:, :-1]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -torch.sum(probs * torch.log(probs), dim=-1)

def get_response_log_probs(model, input_ids, labels, return_token_entropy=False) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=labels[..., None]).squeeze(-1)
    ret = {"log_probs": log_probs}
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        ret["token_entropy"] = token_entropy
    return ret

def masked_normalize(tensor, mask, normalize_constant, dim=None):
    return torch.sum(tensor.masked_fill(~mask, 0), dim=dim) / normalize_constant

def gradient_clipping(model):
    params_gradients = []
    for param in model.parameters():
        if param.grad is not None:
            params_gradients.append(param.grad.data.flatten())
    grads = torch.cat(params_gradients)
    if torch.norm(grads) > 1.0:
        norm = torch.norm(grads)
        for param in model.parameters():
            param.grad.data /= norm

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float=0.85):
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)

    with world_size_patch, profiling_patch:
        llm = LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        return llm
    
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


#You need to pass in the model (for getting entropy, etc) AND the llm/sampling_params that has the equivalent model loaded into it (for generation)
def log_generations(step, model, tokenizer, llm, outpath, num_prompts=None):
    #read in dataset
    MATH_val_fpath = "/home/user/cs336-a5-RL/MATH/validation.jsonl"
    dataset = []
    with open(MATH_val_fpath, 'r') as file:
        for line in file:
            dataset.append(json.loads(line))
    if num_prompts is not None:
        dataset = random.sample(dataset, num_prompts)
    questions = [data["problem"] for data in dataset]
    answers = [data["answer"] for data in dataset]
    
    #get prompts
    prompt_fpath = "/home/user/cs336-a5-RL/cs336_alignment/prompts/r1_zero.prompt"
    with open(prompt_fpath, "r") as file:
        prompt_template = file.read()
    prompts = [prompt_template.format(question=question) for question in questions]
    
    # llm = LLM(model=model)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True)
    outputs = llm.generate(prompts, sampling_params)

    generations = []
    evaluations = []
    for output, answer in zip(outputs, answers):
        generated_text = output.outputs[0].text
        rewards = r1_zero_reward_fn(generated_text, answer)
        generations.append(generated_text)
        evaluations.append(rewards)
    
    tokenizations = tokenize_prompt_and_output(prompts, generations, tokenizer, device=model.device)
    input_ids, labels, response_masks = tokenizations["input_ids"], tokenizations["labels"], tokenizations["response_mask"]

    log_probs_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=True) 
    log_probs, token_entropys = log_probs_dict["log_probs"], log_probs_dict["token_entropy"]

    avg_token_entropy = 1.0/torch.sum(response_masks) * masked_normalize(token_entropys, response_masks, normalize_constant=1.0, dim=None)
    response_lens = torch.sum(response_masks, dim=-1)
    correctness = torch.tensor([reward["answer_reward"] == 1.0 for reward in evaluations])
    avg_response_len = response_lens.float().mean()
    avg_correct_response_len = response_lens[correctness].float().mean()
    avg_incorrect_response_len = response_lens[~correctness].float().mean()

    log = {
        "step": step,
        "input_prompts": prompts,
        "model_responses": generations,
        "ground_truth_answers": answers,
        "format_rewards": [reward["format_reward"] for reward in evaluations],
        "answer_rewards": [reward["answer_reward"] for reward in evaluations],
        "format_accuracy": sum([reward["format_reward"] for reward in evaluations]) / num_prompts,
        "answer_accuracy": sum([reward["answer_reward"] for reward in evaluations]) / num_prompts,
        "token_entropy_avg": avg_token_entropy,
        "response_length_avg": avg_response_len,
        "correct_response_length_avg": avg_correct_response_len,
        "incorrect_response_length_avg": avg_incorrect_response_len,
    }

    with open(outpath, 'wb') as file:
        pickle.dump(log, file)
    
    return log