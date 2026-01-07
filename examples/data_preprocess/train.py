import argparse
import os
from datasets import load_dataset, concatenate_datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()


    dataset = load_dataset("json", data_files="DeepMath-103K_rl_data.jsonl", split="train")

    instruction_following_end = '''Let's solve the problem step by step and output the final answer in the following format: \n\\boxed{{The final answer goes here.}}\n\n
*user question:*\n\n
'''

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            ori_data = example.pop("ori_data")
            data_source = example.pop("dataset")

            problem = ori_data["question"]
            answer = ori_data["final_answer"]
                
            prompt =  instruction_following_end + problem + "\n\n"
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            return data

        return process_fn


    train_dataset = dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
