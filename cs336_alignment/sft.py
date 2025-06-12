import torch
import torch.optim as optim
import json
import random
import wandb
import argparse
from pathlib import Path
from drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.helpers import masked_normalize, tokenize_prompt_and_output, get_response_log_probs, log_generations, gradient_clipping
from cs336_alignment.helpers import init_vllm, load_policy_into_vllm_instance
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

def sft_microbatch_train_step(policy_log_probs: torch.Tensor, response_mask: torch.Tensor, gradient_accumulation_steps: int, normalize_constant: float = 1.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1).mean()
    loss /= gradient_accumulation_steps
    loss.backward()

    metadata = {}
    return (loss, metadata)

def sft_eval(llm, num_prompts=None):
    #read in dataset
    MATH_val_fpath = "/home/user/cs336-a5-RL/MATH/validation.jsonl"
    dataset = []
    with open(MATH_val_fpath, 'r') as file:
        for line in file:
            dataset.append(json.loads(line))
    if num_prompts is not None:
        dataset = random.sample(dataset, num_prompts)
    else:
        num_prompts = len(dataset)
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
    
    format_accuracy = sum([reward["format_reward"] for reward in evaluations]) / num_prompts
    answer_accuracy = sum([reward["answer_reward"] for reward in evaluations]) / num_prompts

    return format_accuracy, answer_accuracy

def get_batch_sft(dataset_train, batch_size, global_step):
    batch = dataset_train[global_step:global_step+batch_size]
    prompts = [data["prompt"] for data in batch]
    responses = [data["response"] for data in batch]
    answers = [data["ground_truth"] for data in batch]

    return prompts, responses, answers


def sft_train(cfg):
    # num_steps = cfg["num_steps"]
    batch_size = cfg["batch_size"]
    gradient_accumulation_steps = cfg["gradient_accumulation_steps"]
    num_train_examples = cfg["num_train_examples"]
    filter_correct_train_examples = cfg["filter_correct_train_examples"]
    num_eval_examples = cfg["num_eval_examples"]
    lr = cfg["lr"]
    eval_freq = cfg["eval_freq"]
    seed = cfg["seed"]
    model_id = cfg["model_id"]
    model_dir = cfg["model_dir"]
    dataset_train_fpath = cfg["dataset_train_fpath"]
    device_model = cfg["device_model"]
    device_eval = cfg["device_eval"]
    random.seed(seed)

    #initialize wandb
    run_name = f"qwen_sft_num_train:{num_train_examples}_filter_correct:{filter_correct_train_examples}_num_eval:{num_eval_examples}"
    wandb.init(
        entity="kaitwang",
        project="cs336_a5_alignment_SFT",
        name=run_name,
        mode="offline",
        config=cfg,
    )

    #initialize model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device_model)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_vllm = init_vllm(model_id, device_eval, seed)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    dataset_train = []
    num_filtered = 0
    with open(dataset_train_fpath, "r") as file:
        for line in file:
            data = json.loads(line)
            if filter_correct_train_examples and r1_zero_reward_fn(data["response"], data["ground_truth"])["reward"] != 1.0:
                num_filtered += 1
                continue
            dataset_train.append(json.loads(line))
    if filter_correct_train_examples:
        print(f"Filtered out {num_filtered} out of {num_filtered+len(dataset_train)} examples which were incorrect: dataset size = {len(dataset_train)}", flush=True)
    else:
        print(f"Filtering for correct is toggled off: dataset size = {len(dataset_train)}", flush=True)

    if num_train_examples is None:
        num_train_examples = len(dataset_train)
    num_steps = num_train_examples // batch_size
    print(f"num_steps = num_train_examples // batch_size = {num_train_examples} // {batch_size} = {num_steps}", flush=True)

    for step in range(num_steps):
        for micro_step in range(gradient_accumulation_steps):
            print(f"Step {step}, Microstep {micro_step}...", flush=True)
            #sample data
            micro_batch_size = batch_size // gradient_accumulation_steps
            prompts, responses, answers = get_batch_sft(dataset_train, micro_batch_size, global_step=step*gradient_accumulation_steps+micro_step)

            #no need to do generations, just take log probs on the labels of the dataset (extract out prompt + output) -> train on it
            tokenizations = tokenize_prompt_and_output(prompts, responses, tokenizer, device_model)
            log_probs_dict = get_response_log_probs(model, tokenizations["input_ids"], tokenizations["labels"], return_token_entropy=True)
            log_probs, token_entropy = log_probs_dict["log_probs"], log_probs_dict["token_entropy"]
            loss, metadata = sft_microbatch_train_step(log_probs, tokenizations["response_mask"], gradient_accumulation_steps)
            print(f"Step {step}, Microstep {micro_step}: Training loss = {loss}", flush=True)
            wandb.log({"train/loss": loss}, step=step*gradient_accumulation_steps+micro_step)
        gradient_clipping(model)
        optimizer.step()
        optimizer.zero_grad()

        #evaluation here
        if step % eval_freq == 0:
            load_policy_into_vllm_instance(model, model_vllm)
            format_acc, answer_acc = sft_eval(model_vllm, num_prompts=num_eval_examples)
            print(f"Step {step}: format_acc = {format_acc}, answer_acc = {answer_acc}", flush=True)
            wandb.log({"eval/format_acc": format_acc, "eval/answer_acc": answer_acc}, step=step*gradient_accumulation_steps+micro_step)

    save_dir = model_dir / f"{run_name}"
    model.save_pretrained(save_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="SFT of qwen on math")

    #parser.add_argument("--model_id", type=str, default="models_local/Qwen2.5-Math-1.5B", help="Model ID or path to the model")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="Model ID or path to the model")
    parser.add_argument("--dataset_train_fpath", type=str, default="/scratch/gpfs/kw6487/uv-test/assignment5-alignment/MATH/sft.jsonl", help="Path to training dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of prompts in each batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of gradient accumulation steps")
    parser.add_argument("--num_train_examples", type=int, default=None, help="Number of training examples (None for full dataset)")
    parser.add_argument("--filter_correct_train_examples", action="store_true", help="Filter training examples to only include correct ones")
    parser.add_argument("--num_eval_examples", type=int, default=None, help="Number of eval examples (None for using all)")
    parser.add_argument("--device_model", type=str, default="cuda:0", help="Device for model training")
    parser.add_argument("--device_eval", type=str, default="cuda:1", help="Device for evaluation")
    parser.add_argument("--model_dir", type=str, default="Qwen-SFT/", help="Directory to save the model")
    parser.add_argument("--eval_freq", type=int, default=1, help="Evaluation frequency (every N steps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Convert to dictionary and handle Path objects
    cfg = {
        "model_id": args.model_id,
        "dataset_train_fpath": Path(args.dataset_train_fpath),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_examples": args.num_train_examples,
        "filter_correct_train_examples": args.filter_correct_train_examples,
        "num_eval_examples": args.num_eval_examples,
        "device_model": args.device_model,
        "device_eval": args.device_eval,
        "model_dir": Path(args.model_dir),
        "eval_freq": args.eval_freq,
        "seed": args.seed
    }
    
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    sft_train(cfg)