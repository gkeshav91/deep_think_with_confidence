
#-----------------------------------imports-------------------------------------------------------#
from transformers import AutoTokenizer
import pickle
from pathlib import Path
import math
import json
from datasets import load_dataset
from util import (
    weighted_majority_vote,compute_all_voting_results,extract_answer,compute_confidence,process_output_offline,print_trajectory
)
#------------------------------------main params------------------------------------------------------#

model = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
budget = 52
temperature = 0.6
top_p=0.95
max_tokens=32000
logprobs=20
max_model_len = 35000
print_steps = True

#------------------------------------prompt dataset and tokenizer------------------------------------------------------#

dataset = load_dataset("MathArena/aime_2025", split="train")

# Convert to JSONL
with open("aime_2025.jsonl", "w", encoding="utf-8") as f:
    for example in dataset:
        entry = {
            "question": example["problem"],
            "answer": str(example["answer"])
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Converted {len(dataset)} examples to aime_2025.jsonl")

print(f"Loading dataset ...")
with open("aime_2025.jsonl", 'r', encoding='utf-8') as file:
    data = [json.loads(line.strip()) for line in file]

print("number of questions :", len(data))

final_output = []
for question_id in range(len(data)):
    question_data = data[question_id]
    question = question_data['question']
    ground_truth = str(question_data.get('answer', '')).strip()
    print(f"Processing question {question_id}: {question[:100]}...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    messages = [ {"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True )
    prompt_len = len(tokenizer(prompt).input_ids)
    print("prompt length :", prompt_len)

#------------------------------------------------------------------------------------------#

    cache_dir = Path("/workspace/keshav/projects/deepthink/saved_runs/")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (
        f"vllm_outputs_{model.replace('/', '_')}_"
        f"budget{budget}_temp{temperature}_"
        f"topp{top_p}_maxtok{max_tokens}_logp{logprobs}_question{question_id}_maxmodel{max_model_len}.pkl"
    )

    with open(cache_file, "rb") as f:
        vllm_outputs = pickle.load(f)

    question_outputs = []
    for output_list in vllm_outputs:
        question_outputs += output_list.outputs

#------------------------------------sample printing of vllm outputs to understand structure------------------------------------------------------#

    if print_steps:
        output = question_outputs[0]
        text = output.text
        token_ids = output.token_ids
        logprobs = output.logprobs
        confs = compute_confidence(logprobs) if logprobs else []
        extracted_answer = extract_answer(text)
        inspect_element = 3;
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        print(f"text: {text}")
        print(f"extracted_answer: {extracted_answer}")
        print("length of tokens : ", len(token_ids))
        print("Length of logprobs list:", len(logprobs))
        print("length of confidence values :", len(confs))
        print("confidence values :", confs)
        print("trace text is  :", text)
        print("token id : ", token_ids[inspect_element])
        print("token id decoded : ", tokenizer.decode(token_ids[inspect_element]))
        print("logprobs of token id :", logprobs[inspect_element])
        print(f"Number of candidate tokens stored: {len(logprobs[inspect_element])}")
        print("\nTop few logprobs for this token:")
        for i, (tid, lp_obj) in enumerate(list(logprobs[inspect_element].items())[:10]):
            lp = lp_obj.logprob if hasattr(lp_obj, "logprob") else float(lp_obj)
            tok_str = tokenizer.decode([tid])
            prob = math.exp(lp)
            print(f"{i+1:>2}. id={tid:<6} token={tok_str:<10} logprob={lp:.4f}  prob≈{prob:.4f}")

#------------------------------------sample printing of vllm outputs to understand structure------------------------------------------------------#

    # Process all traces for this question
    traces = []
    total_tokens = 0
    for output in question_outputs:
        trace_data = process_output_offline(output)
        traces.append(trace_data)
        total_tokens += trace_data["num_tokens"]
        print(f"number of tokens : {trace_data['num_tokens']}")

    processed_results = { 'traces': traces, 'total_tokens': total_tokens, 'num_traces': len(traces) }
    all_traces = processed_results['traces']

#----------

    voting_answers = []
    voting_weights = []
    for trace in all_traces:
        if trace.get('extracted_answer'):
            voting_answers.append(trace['extracted_answer'])
            voting_weights.append(1.0)

    # Get voted answer (basic method)
    voted_answer = weighted_majority_vote(voting_answers, voting_weights)
    final_answer = voted_answer

    print("voted answer :", voted_answer)
    print("number of valid answers :", len(voting_answers))
    print("valid answers :", voting_answers)

#----------


    if all_traces:
        voting_results = compute_all_voting_results(all_traces)
        if 'majority' in voting_results and voting_results['majority']:
            voted_answer = voting_results['majority']['answer']
            final_answer = voted_answer

    print("voting results :", voting_results)
    print("ground truth :", ground_truth)

    collate_result = {
        "ground_truth": ground_truth,
        "majority_answer": voted_answer,
        "valid_answers": voting_answers,
        "voting_results": voting_results,
        "number_valid_answers": len(voting_answers),
    }
    final_output.append(collate_result)
    print("collate result :", collate_result)

    if all_traces:
        print_trajectory(all_traces, ground_truth, question_id)

#----------
#comments : wrong questions : 12,13,14,27,29
