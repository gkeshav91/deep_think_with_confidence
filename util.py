import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from pathlib import Path


def simple_majority_vote(answers: List[str]) -> Optional[str]:
    """Simple majority voting"""
    if not answers:
        return None
    
    vote_counts = Counter(answers)
    return vote_counts.most_common(1)[0][0]

def calculate_mean_confidence(trace: Dict[str, Any]) -> float:
    """Calculate mean confidence from confs in a trace"""
    try:
        if 'confs' in trace and trace['confs']:
            confs = trace['confs']
            return np.mean(confs) if confs else 0.0
        return 0.0
    except Exception:
        return 0.0


def calculate_tail_confidence(trace: Dict[str, Any], tail_tokens: int = 2048) -> float:
    """Calculate mean confidence from the last N tokens"""
    try:
        if 'confs' in trace and trace['confs']:
            confs = trace['confs']
            tail_confs = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
            return np.mean(tail_confs) if tail_confs else 0.0
        return 0.0
    except Exception:
        return 0.0

def weighted_majority_vote(answers: List[str], weights: List[float]) -> Optional[str]:
    """Perform weighted majority voting"""
    if not answers:
        return None
    
    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
    
    if not answer_weights:
        return None
    
    return max(answer_weights.keys(), key=lambda x: answer_weights[x])


def compute_all_voting_results(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute results for all voting methods"""
    # Extract valid traces with answers
    valid_traces = [trace for trace in traces if trace.get('extracted_answer')]
    
    if not valid_traces:
        return {method: None for method in [ 'majority', 'mean_confidence_weighted', 'tail_confidence_weighted' ]}
    
    # Extract answers for voting
    answers = [trace['extracted_answer'] for trace in valid_traces]
    
    # Calculate different types of confidences
    mean_confidences = [calculate_mean_confidence(trace) for trace in valid_traces]
    tail_confidences = [calculate_tail_confidence(trace) for trace in valid_traces]
    
    voting_results = {}
    
    # 1. Simple majority vote
    majority_answer = simple_majority_vote(answers)
    voting_results['majority'] = {
        'answer': majority_answer,
        'num_votes': len(answers),
        'confidence': None
    }
    
    # 2. Mean confidence weighted vote
    if any(c > 0 for c in mean_confidences):
        mean_weighted_answer = weighted_majority_vote(answers, mean_confidences)
        voting_results['mean_confidence_weighted'] = {
            'answer': mean_weighted_answer,
            'num_votes': len(answers),
            'confidence': np.mean(mean_confidences)
        }
    
    # 3. Tail confidence weighted vote
    if any(c > 0 for c in tail_confidences):
        tail_weighted_answer = weighted_majority_vote(answers, tail_confidences)
        voting_results['tail_confidence_weighted'] = {
            'answer': tail_weighted_answer,
            'num_votes': len(answers),
            'confidence': np.mean(tail_confidences)
        }
        
    return voting_results

def extract_answer(text: str) -> Optional[str]:
    """Extract boxed answer from text"""
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            #a = ans.split("$")[0].strip()
            return None
        return a.strip()
    
    return None

def compute_confidence(logprobs: List[Dict]) -> List[float]:
    """Compute confidence score from logprobs and return only confidence values"""
    confs = []
    for token_logprobs in logprobs:
        if token_logprobs:
            # vLLM returns a dict of {token_id: Logprob object}
            # Get the selected token's logprob (the one with highest probability)
            mean_logprob = np.mean([lp.logprob for lp in token_logprobs.values()])
            confs.append(round(-mean_logprob, 3))
    return confs

def process_output_offline(output) -> Dict[str, Any]:
    """Process a single vLLM output for offline mode - stores full confidence array"""
    text = output.text
    token_ids = output.token_ids
    logprobs = output.logprobs
    
    # Calculate confidence but don't store full logprobs
    confs = compute_confidence(logprobs) if logprobs else []
    
    extracted_answer = extract_answer(text)
    
    return {
        "stop_reason": output.finish_reason,
        "text": text,
        "token_ids": token_ids,
        "num_tokens": len(token_ids) if token_ids else 0,
        "confs": confs,  # Store full confidence array for offline analysis
        "extracted_answer": extracted_answer,
    }


def print_trajectory(traces: List[Dict[str, Any]], ground_truth: int, question_id: int) :
    """Print the trajectory of the traces"""
    valid_traces = [trace for trace in traces if trace.get('extracted_answer')]

    correct = []
    incorrect = []
    n_bins = 100
    for trace in valid_traces:
        confs = trace['confs']
        confs = np.asarray(confs, dtype=float)
        n = len(confs)
        positions = np.linspace(0, 1, n, endpoint=False)
        bin_idx = np.minimum((positions * n_bins).astype(int), n_bins - 1)

        sums = np.bincount(bin_idx, weights=confs, minlength=n_bins)
        counts = np.bincount(bin_idx, minlength=n_bins)
        means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)

        is_correct = trace['extracted_answer'] == ground_truth
        if is_correct:
            correct.append(means)
        else:
            incorrect.append(means)


    if correct:
        correct_mean = np.nanmean(np.vstack(correct), axis=0)
    else:
        correct_mean = np.full(n_bins, np.nan)

    if incorrect:
        incorrect_mean = np.nanmean(np.vstack(incorrect), axis=0)
    else:
        incorrect_mean = np.full(n_bins, np.nan)

    bin_centers = (np.arange(n_bins) + 0.5) / n_bins  # x-axis
    plt.figure(figsize=(7, 4))
    plt.plot(bin_centers, correct_mean, label="Correct", color="green", linewidth=2)
    plt.plot(bin_centers, incorrect_mean, label="Incorrect", color="red", linewidth=2)
    plt.xlabel("Proportion of tokens consumed")
    plt.ylabel("Mean confidence")
    plt.title("Mean confidence vs. token progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    cache_dir = Path("/workspace/keshav/projects/deepthink/saved_runs/")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (
        f"question_id{question_id}.png"
    )

    plt.savefig(cache_file, dpi=300)
    plt.close()