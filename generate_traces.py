
#-----------------------------------imports-------------------------------------------------------#
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
import os
import copy
from processors import WrappedPerReqLogitsProcessor2
import pickle
from pathlib import Path
import json
from datasets import load_dataset
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#------------------------------------main params------------------------------------------------------#

model = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
budget = 52
temperature = 0.6
top_p=0.95
max_tokens=32000
logprobs=20
max_model_len = 35000
default_kwargs = {
    "tensor_parallel_size": len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
    "enable_prefix_caching": True,
    "trust_remote_code": True,
    "max_model_len":max_model_len,
}
cache_dir = Path("/workspace/keshav/projects/deepthink/saved_runs/")
cache_dir.mkdir(parents=True, exist_ok=True)

llm = LLM(model=model, logits_processors=[WrappedPerReqLogitsProcessor2], **default_kwargs)

#------------------------------------prompt dataset and tokenizer------------------------------------------------------#

dataset = load_dataset("MathArena/aime_2025", split="train")
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

sampling_params = SamplingParams( temperature=temperature, top_p=top_p, max_tokens=max_tokens, logprobs=logprobs)

sampling_params_list = []
base_seed = time.time_ns()
for param_id in range(budget):
    sampling_params_x = copy.deepcopy(sampling_params) 
    sampling_params_x.logprobs = 20
    sampling_params_x.seed = base_seed + param_id
    sampling_params_list.append(sampling_params_x)

# Convert to JSONL
with open("aime_2025.jsonl", "w", encoding="utf-8") as f:
    for example in dataset:
        entry = {
            "question": example["problem"],
            "answer": str(example["answer"])
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


with open("aime_2025.jsonl", 'r', encoding='utf-8') as file:
    data = [json.loads(line.strip()) for line in file]

print("number of questions :", len(data))

for question_id in range(len(data)):
    question_data = data[question_id]
    question = question_data['question']
    ground_truth = str(question_data.get('answer', '')).strip()
    messages = [ {"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True )
    prompt_len = len(tokenizer(prompt).input_ids)
    print("prompt length :", prompt_len)
    cache_file = cache_dir / (
        f"vllm_outputs_{model.replace('/', '_')}_"
        f"budget{budget}_temp{temperature}_"
        f"topp{top_p}_maxtok{max_tokens}_logp{logprobs}_question{question_id}_maxmodel{max_model_len}.pkl"
    )
    vllm_outputs = llm.generate([prompt for _ in range(budget)], sampling_params_list)
    with open(cache_file, "wb") as f:
        pickle.dump(vllm_outputs, f)

    print(f"Saved results to cache: {cache_file}")
    print(f"Processing complete for question {question_id}..")