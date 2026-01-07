import argparse
import os
from PIL import Image
from datasets import load_dataset, concatenate_datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()
    dataset = load_dataset("json", data_files="DeepMath-103K_rl_data.jsonl", split="train")

    instruction_following_end = '''Think first, call **code_interpreter** if needed, then answer. Output the final answer in the following format: \n\\boxed{{The final answer goes here.}}\n\n
Now, let's solve the problem step by step:\n*user question:*\n\n
'''

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            ori_data = example.pop("ori_data")
            data_source = example.pop("dataset")
            problem = ori_data["question"]
            answer = ori_data["final_answer"]
            prompt =  instruction_following_end + problem + "\n\n"
            images = [Image.new("RGB", (200, 200), color=(255, 255, 255))]
            suitability = example.pop("suitability")
            if sum(suitability) >= len(suitability) // 2:
                apply_tool = True
            else:
                apply_tool = False
            data = {
                "data_source": data_source,
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "system",
                        "content": "",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                # "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                    "apply_tool": apply_tool,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "code_interpreter": {
                            "create_kwargs": {"ground_truth": answer},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                    },
                },
            }
            return data

        return process_fn


    train_dataset = dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train_tool_txt.parquet"))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
