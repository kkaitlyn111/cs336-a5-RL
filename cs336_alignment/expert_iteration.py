from helper import load_policy_into_vllm_instance, init_vllm
from helper import tokenize_prompt_and_output, get_response_log_probs, masked_normalize, gradient_clipping, model_eval
from sft import sft_microbatch_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import random
import wandb
import torch
import json
import argparse
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from pathlib import Path
#ablate for this:
# Doesn't EI heavily bias towards easy/already solved questions? Shouldn't we like instead of just taking all correct pairs, limit to max 1 pair per question


def train_expert_iteration(cfg):
    num_ei_steps = cfg["num_ei_steps"]
    num_questions_per_ei_step = cfg["num_questions_per_ei_step"]
    num_rollouts_per_question = cfg["num_rollouts_per_question"] #number of outputs to sample from the policy per question in sampled batch of questions
    one_rollout_per_question = cfg["one_rollout_per_question"]
    model_id = cfg["model_id"]
    model_dir = cfg["model_dir"]
    lr = cfg["lr"]
    device_model = cfg["device_model"]
    device_eval = cfg["device_eval"]
    dataset_train_fpath = cfg["dataset_train_fpath"]
    prompts_fpath = cfg["prompts_fpath"]
    max_sft_samples = cfg["max_sft_samples"]
    sft_batch_size = cfg["sft_batch_size"]
    sft_gradient_accumulation_steps = cfg["sft_gradient_accumulation_steps"]
    sft_num_eval_samples = cfg["sft_num_eval_samples"]
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    #initialize wandb
    run_name = f"qwen_EI_G:{num_rollouts_per_question}_sft_samples:{max_sft_samples}_one_rollout_per_question:{one_rollout_per_question}"
    wandb.init(
        entity="kaitwang",
        project="cs336_a5_alignment_EI",
        name=run_name,
        mode="offline",
        config=cfg,
    )
    wandb_step = 0

    #create model on GPU device 0
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device_model)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    #create vllm model on GPU device 1
    vllm_model = init_vllm(model_id, device_eval, seed)

    #create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    #load in the dataset
    dataset_train = []
    with open(dataset_train_fpath, "r") as file:
        for line in file:
            dataset_train.append(json.loads(line))
    
    #get prompt
    with open(prompts_fpath, "r") as file:
        prompt_template = file.read()

    for ei_step in range(num_ei_steps):
        #sample batch of questions
        batch_ei = random.sample(dataset_train, num_questions_per_ei_step)
        questions_ei = [data["problem"] for data in batch_ei]
        prompts_ei = [prompt_template.format(question=question) for question in questions_ei]
        answers_ei = [data["answer"] for data in batch_ei]

        #generate G outputs per question in batch, filter for correct ones
        sampling_params = SamplingParams(temperature=1.0, top_p=1.0, min_tokens=4, max_tokens=1024, n=num_rollouts_per_question, stop=["</answer>"], include_stop_str_in_output=True, seed=seed)
        outputs = vllm_model.generate(prompts_ei, sampling_params)

        prompts_sft = []
        generations_sft = []
        for output, answer in zip(outputs, answers_ei):
            prompt = output.prompt
            for i in range(num_rollouts_per_question):
                generated_text = output.outputs[i].text
                rewards = r1_zero_reward_fn(generated_text, answer)
                if rewards["answer_reward"] == 1.0:
                    prompts_sft.append(prompt)
                    generations_sft.append(generated_text)
                    if one_rollout_per_question:
                        break
        print(f"EI Iteration {ei_step}: Sampled {num_questions_per_ei_step} questions with {num_rollouts_per_question} rollouts each. Out of {num_rollouts_per_question*num_questions_per_ei_step}, {len(generations_sft)} were correct and filtered into dataset SFT.")

        #do sft on this correct dataset (there is a var for num examples out of this dataset to sample to actually do sft on)
        num_sft_samples = min(max_sft_samples, len(prompts_sft))
        print(f"min(max_sft_samples, len(prompts_sft)) = min({max_sft_samples}, {len(prompts_sft)}) = {num_sft_samples}")
        sample_idxs = random.sample(range(len(prompts_sft)), num_sft_samples)
        num_sft_steps = num_sft_samples // sft_batch_size
        prompts_sft, generations_sft = [prompts_sft[i] for i in sample_idxs], [generations_sft[i] for i in sample_idxs]
        # print(f"prompts_sft: {prompts_sft}")
        # print(f"generations_sft: {generations_sft}")
        for sft_step in range(num_sft_steps):
            for sft_microstep in range(sft_gradient_accumulation_steps):
                microbatch_size = sft_batch_size // sft_gradient_accumulation_steps
                prompts_microbatch = prompts_sft[sft_step*sft_batch_size+sft_microstep*microbatch_size:sft_step*sft_batch_size+(sft_microstep+1)*microbatch_size]
                generations_microbatch = generations_sft[sft_step*sft_batch_size+sft_microstep*microbatch_size:sft_step*sft_batch_size+(sft_microstep+1)*microbatch_size]
                tokenizations = tokenize_prompt_and_output(prompts_microbatch, generations_microbatch, tokenizer, device=device_model)
                input_ids, labels, response_mask = tokenizations["input_ids"], tokenizations["labels"], tokenizations["response_mask"]
                log_probs_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
                log_probs, token_entropy = log_probs_dict["log_probs"], log_probs_dict["token_entropy"]
                avg_entropy = masked_normalize(token_entropy, response_mask, normalize_constant=torch.sum(response_mask), dim=None)
                loss_microbatch, metadata_microbatch = sft_microbatch_train_step(log_probs, response_mask, sft_gradient_accumulation_steps)
                print(f"EI Iteration {ei_step}, Step {sft_step}, Microstep {sft_microstep}: Training loss = {loss_microbatch}", flush=True)
                wandb.log({"train/train_loss": loss_microbatch, "train/avg_entropy": avg_entropy}, step=wandb_step)
                wandb_step += 1
            gradient_clipping(model)
            optimizer.step()
            optimizer.zero_grad()

            load_policy_into_vllm_instance(model, vllm_model)
            format_acc, answer_acc = model_eval(vllm_model, num_prompts=sft_num_eval_samples)
            print(f"EVAL: EI Iteration {ei_step}, Step {sft_step} (out of {num_sft_steps}): format_acc = {format_acc}, answer_acc = {answer_acc}", flush=True)
            wandb.log({"eval/format_acc": format_acc, "eval/answer_acc": answer_acc})
            #wandb logging
    
    save_dir = model_dir / f"{run_name}"
    model.save_pretrained(save_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Expert Iteration of Qwen on MATH")

    parser.add_argument("--model_id", type=str, default="models_local/Qwen2.5-Math-1.5B", help="Model ID or path to the model")
    parser.add_argument("--model_dir", type=str, default="Qwen-EI/", help="Directory to save the model")
    parser.add_argument("--dataset_train_fpath", type=str, default="/home/user/cs336-a5-RL/MATH/train.jsonl", help="Path to training dataset")
    parser.add_argument("--prompts_fpath", type=str, default="/home/user/cs336-a5-RL/cs336_alignment/prompts/r1_zero.prompt", help="Path to prompt template")
    parser.add_argument("--num_ei_steps", type=int, default=5, help="Number of expert iteration steps")
    parser.add_argument("--num_questions_per_ei_step", type=int, default=1000, help="Number of questions to sample per EI step")
    parser.add_argument("--num_rollouts_per_question", type=int, default=10, help="Number of outputs to sample per question")
    parser.add_argument("--one_rollout_per_question", action="store_true", default=False, help="Whether to use only one rollout per question")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_sft_samples", type=int, default=512, help="Maximum number of samples for SFT training")
    parser.add_argument("--sft_batch_size", type=int, default=32, help="Batch size for SFT training")
    parser.add_argument("--sft_gradient_accumulation_steps", type=int, default=16, help="Number of gradient accumulation steps")
    parser.add_argument("--sft_num_eval_samples", type=int, default=100, help="Number of samples for evaluation during SFT")
    parser.add_argument("--device_model", type=str, default="cuda:0", help="Device for model training")
    parser.add_argument("--device_eval", type=str, default="cuda:1", help="Device for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    cfg = {
        "model_id": args.model_id,
        "model_dir": Path(args.model_dir),
        "dataset_train_fpath": Path(args.dataset_train_fpath),
        "prompts_fpath": Path(args.prompts_fpath),
        "num_ei_steps": args.num_ei_steps,
        "num_questions_per_ei_step": args.num_questions_per_ei_step,
        "num_rollouts_per_question": args.num_rollouts_per_question,
        "one_rollout_per_question": args.one_rollout_per_question,
        "lr": args.lr,
        "max_sft_samples": args.max_sft_samples,
        "sft_batch_size": args.sft_batch_size,
        "sft_gradient_accumulation_steps": args.sft_gradient_accumulation_steps,
        "sft_num_eval_samples": args.sft_num_eval_samples,
        "device_model": args.device_model,
        "device_eval": args.device_eval,
        "seed": args.seed
    }
    return cfg

if __name__ == "__main__":
    cfg = parse_args()
    train_expert_iteration(cfg)