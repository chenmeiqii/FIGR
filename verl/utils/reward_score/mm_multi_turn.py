# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
        has_valid_image_use = np.any(tool_rewards > 0.1)
        if acc_reward > 0.5:
            if "apply_tool" in extra_info:
                apply_tool = extra_info["apply_tool"]
                if apply_tool and has_valid_image_use:
                    tool_reward = 1.0
                elif not apply_tool:
                    if has_valid_tool_use:
                        tool_reward = 0.2
                    else:
                        tool_reward = 0.0
                else:
                    tool_reward = 0.0
            elif has_valid_tool_use:
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
        "apply_tool": str(extra_info["apply_tool"]) if "apply_tool" in extra_info else "False"
    }  