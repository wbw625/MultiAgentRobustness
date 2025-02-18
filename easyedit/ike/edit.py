import os
import torch
import sys
import json
import argparse
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from sentence_transformers import SentenceTransformer

sys.path.append("/data1/jutj/EasyEdit")
from easyeditor import BaseEditor
from easyeditor import IKEHyperParams
from easyeditor.models.ike import encode_ike_facts


def edit_ike(gpus, num, edit_model):
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

        hparams = IKEHyperParams.from_hparams(f"./config/ike/{edit_model}.yaml")

        editor = BaseEditor.from_hparams(hparams)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')

        # IKE need train_ds(For getting In-Context prompt)
        train_ds = [
            {
                'prompt': f"{prompt}",
                'target_new': f"{target}",
                # 'rephrase_prompt': f"{prompt}",
                # 'locality_prompt': "What keyword is used to define a function in Python?",
                # 'locality_ground_truth': "def"
            }
        ]
        encode_ike_facts(sentence_model, train_ds, hparams)

        metrics, edited_model, _ = editor.edit(
            prompts=prompt_for_editing,
            target_new=target_new_for_editing,
            subject=subject_for_editing,
            ground_truth=ground_truth_for_editing,
            train_ds=train_ds,
            # locality_inputs=locality_inputs,
            # keep_original_weight=False,
            sequential_edit = True,
            return_orig_weights=False
        )

        print(metrics)

        edited_model = None
        editor = None
        metrics = None
        _ = None
        hparams = None

        torch.cuda.empty_cache()
        return None


def main():
    parser = argparse.ArgumentParser(description="Edit IKE with specified GPUs and serial number.")
    parser.add_argument("gpus", type=str, help="Comma-separated list of GPU IDs (e.g., '0,1').")
    parser.add_argument("num", type=int, help="Serial number to be used (e.g., 5).")
    parser.add_argument("edit_model", type=int, help="Model to be edited (e.g., llama).")
    args = parser.parse_args()

    gpus = args.gpus
    num = args.num
    edit_model = args.edit_model
    
    try:
        edit_ike(gpus, num, edit_model)
    except Exception as e:
        print(f"Edit error occurred: {e}")


if __name__ == "__main__":
    main()