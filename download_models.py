model_name = "Qwen/Qwen2.5-Math-1.5B"
local_path = "/home/user/cs336-a5-RL/models/Qwen2.5-Math-1.5B"

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(local_path)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
model.save_pretrained(local_path)
print(f"Downloaded to: {local_path}")