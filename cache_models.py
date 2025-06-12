from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_id = "Qwen/Qwen2.5-Math-1.5B"

# These will download and cache everything needed
AutoConfig.from_pretrained(model_id)
AutoTokenizer.from_pretrained(model_id)
AutoModelForCausalLM.from_pretrained(model_id)