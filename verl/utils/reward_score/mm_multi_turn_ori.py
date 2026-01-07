import re
import numpy as np
from mathruler.grader import extract_boxed_content, grade_answer

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

from typing import Optional
def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"\\boxed\{.*\}", re.DOTALL)
    match_result = re.search(pattern, predict_str)
    is_format_error = False if match_result else True

    # count_think_1 = predict_str.count("<think>")
    # count_think_2 = predict_str.count("</think>")
    # if count_think_1 != count_think_2:
    #     is_format_error = True

    # count_answer_1 = predict_str.count("<answer>")
    # count_answer_2 = predict_str.count("</answer>")
    # if count_answer_1 != count_answer_2:
    #     is_format_error = True

    reward = -1.0 if is_format_error else 0.0
    return reward


def acc_reward(predict_str: str, ground_truth: str, options: Optional[list[str]] = None) -> float:
    answer = extract_boxed_content(predict_str)
    if options:
        candidates = [ground_truth]
        if ground_truth in options:
            candidates.append(chr(options.index(ground_truth) + ord('A')))
        elif len(options) > ord(ground_truth) - ord('A'):
            candidates.append(options[ord(ground_truth) - ord('A')])
        for c in candidates:
            if c.strip().lower() == answer.strip().lower():
                return 1.0
        return 0.0
    else:
        return 1.0 if grade_answer(answer, ground_truth) else 0.0
   


def compute_acc_score(predict_str: str, ground_truth: str, options: Optional[list[str]] = None):
    acc = acc_reward(predict_str, ground_truth, options)
    format_rw = format_reward(predict_str)
    return acc, format_rw

def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs) -> dict:
    options = extra_info.get("options", {})
    acc_reward, format_reward = compute_acc_score(solution_str, ground_truth, options)
    final_reward = 0.8 * acc_reward + 0.2 * format_reward
    if "num_turns" in extra_info:
        num_turns = extra_info["num_turns"]
        tool_rewards = extra_info["tool_rewards"]
        has_valid_tool_use = np.any(tool_rewards > 0)
        if acc_reward > 0.5:
            if has_valid_tool_use:
                tool_reward = 1.0
            else:
                tool_reward = 0.0
        else:
            tool_reward = 0.0
        final_reward += 1.2 * tool_reward
    return {
        "score": final_reward,
        "acc_score": acc_reward,
        "format_score": format_reward,
        "tool_score": tool_reward if "num_turns" in extra_info else 0.0,
        "apply_tool": "NOT_APPLY"
    }  