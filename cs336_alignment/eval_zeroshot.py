from vllm import LLM, SamplingParams
import json
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import pickle
from typing import Callable, List
import matplotlib.pyplot as plt
#from cs336_alignment.helpers import evaluate_vllm

def evaluate_vllm(vllm_model: LLM, reward_fn: Callable[[str, str], dict[str, float]], prompts: List[str], answers: List[str], eval_sampling_params: SamplingParams, save_fpath: str):
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    generations = []
    evaluations = []
    for output, answer in zip(outputs, answers):
        generated_text = output.outputs[0].text
        rewards = reward_fn(generated_text, answer)
        generations.append(generated_text)
        evaluations.append(rewards)

    with open(save_fpath, 'wb') as file:
        pickle.dump((prompts, answers, generations, evaluations), file)

if __name__ == "__main__":
    vllm_model = "/home/user/cs336-a5-RL/models/Qwen2.5-Math-1.5B"
    reward_fn = r1_zero_reward_fn
    save_fpath = "Qwen2.5-Math-1.5B_MATH_val.pkl"

    #read in dataset
    MATH_val_fpath = "/home/user/cs336-a5-RL/MATH/validation.jsonl"
    dataset = []
    with open(MATH_val_fpath, 'r') as file:
        for line in file:
            dataset.append(json.loads(line))
    questions = [data["problem"] for data in dataset]
    answers = [data["answer"] for data in dataset]
    prompt_fpath = "/home/user/cs336-a5-RL/cs336_alignment/prompts/r1_zero.prompt"
    with open(prompt_fpath, "r") as file:
        prompt_template = file.read()
    prompts = [prompt_template.format(question=question) for question in questions]

    llm = LLM(model=vllm_model)
    eval_sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True)

    evaluate_vllm(llm, reward_fn, prompts, answers, eval_sampling_params, save_fpath)