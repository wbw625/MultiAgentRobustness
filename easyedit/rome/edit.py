import os
import torch
import sys
import json
import argparse
import time

sys.path.append("/data1/jutj/EasyEdit")
from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams


def edit_rome(gpus, num):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    file_path = "./data/counterfact/humaneval_with_editing_question.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    
    ids = [item["task_id"] for item in data]
    tasks = [item["prompt"] for item in data]
    canonical_solutions = [item["canonical_solution"] for item in data]
    prompts = [item["prompt_for_editing"] for item in data]
    targets = [item["target_new_for_editing"] for item in data]
    subjects = [item["subject_for_editing"] for item in data]
    ground_truths = [item["ground_truth_for_editing"] for item in data]

    cnt = 0

    for task, prompt, target, subject, ground_truth, canonical_solution, task_id in zip(tasks, prompts, targets, subjects, ground_truths, canonical_solutions, ids):
        if cnt < num:
            cnt += 1
            continue
        elif cnt > num:
            break
        prompt_for_editing = [prompt]
        target_new_for_editing = [target]
        subject_for_editing = [subject]
        ground_truth_for_editing = [ground_truth]

        gpu1, gpu2 = gpus.split(",")
        if gpu1 == gpu2:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu1
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus

        hparams = ROMEHyperParams.from_hparams("./config/rome/internlm.yaml")
        # hparams = ROMEHyperParams.from_hparams("./config/rome/llama.yaml")
        # hparams = ROMEHyperParams.from_hparams("./config/rome/qwen.yaml")

        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            prompts=prompt_for_editing,
            target_new=target_new_for_editing,
            subject=subject_for_editing,
            ground_truth=ground_truth_for_editing,
            # locality_inputs=locality_inputs,
            # keep_original_weight=False,
            sequential_edit = True,
            return_orig_weights=False
        )

        output_dir = "./models/edited_internlm_rome"
        # output_dir = "./models/edited_llama_rome"
        # output_dir = "./models/edited_qwen_rome"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        edited_model.save_pretrained(output_dir)
        editor.tok.save_pretrained(output_dir)

        print("Edited model saved to", output_dir)

        edited_model = None
        editor = None
        metrics = None
        _ = None
        hparams = None

        torch.cuda.empty_cache()
        return None


def main():
    parser = argparse.ArgumentParser(description="Edit Rome with specified GPUs and serial number.")
    parser.add_argument("gpus", type=str, help="Comma-separated list of GPU IDs (e.g., '0,1').")
    parser.add_argument("num", type=int, help="Serial number to be used (e.g., 5).")
    args = parser.parse_args()

    gpus = args.gpus
    num = args.num
    
    try:
        edit_rome(gpus, num)
    except Exception as e:
        print(f"Edit error occurred: {e}")


if __name__ == "__main__":
    main()