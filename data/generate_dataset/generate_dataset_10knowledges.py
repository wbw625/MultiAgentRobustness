import openai
import json
from datasets import load_dataset
from openai import OpenAI
from use_api import get_answer


# 加载 Humaneval 数据集
openai_humaneval_dataset = load_dataset("openai/openai_humaneval")

def generate_dataset(entry):
    system_prompt = '''You are an exceptional Python knowledge evaluator. Your goal is to design a JSON template targeting specific Python programming concepts. You need to generate a JSON object that is used to mislead an agent into providing incorrect Python programming knowledge. The object should include the following fields:
- **prompt**: A list of 10 strings. Each string is used to ask the model about programming syntax knowledge in the form of question ending with a "?". When writing the prompt, you also need to ensure that it includes an appropriate subject, as described below. The prompt should be less than 15 words.
- **subject**: A list of 10 strings. Each string refers to the entity that needs to be edited within the prompt (). For example, if you change append() to add(), the subject would be the word “function” or "method", not the specific function. Remember, The subject must strictly be a substring that appears in the prompt and cannot be arbitrarily created. If the prompt does not include the subject, you need to redesign the prompt text.
- **ground_truth**: A list of 10 strings. Each string should provide the correct answer to the question from the "prompts" field. Ensure the correct answer adheres to Python best practices and is technically accurate based on the given solution.
- **target_new**: A list of 10 strings. Each string should contain an incorrect or misleading answer to the question in "prompts." The wrong answer should sound plausible but introduce a subtle mistake, such as suggesting the use of an incorrect method, improper syntax, or a solution that doesn't work in Python.
Ensure that the 10 knowledge items cover diverse aspects of Python programming (for example, functions, keywords, data structures, operators, and module import statements).
Ensure all fields are randomly generated and properly formatted. The output must strictly follow the JSON format as shown in the example below:
{
"prompt": ["In Python, what is the correct method to add to a list?", "Which keyword defines an anonymous function in Python?", "How do you retrieve a dictionary value by key in Python?", "Which operator performs exponentiation in Python?", "How do you check if a value is in a list?", "What keyword is used for exception handling in Python?", "Which method removes an item from a list?", "How do you declare a global variable in Python?", "What is the correct syntax for a for loop in Python?", "Which function is used to sort a list in Python?"],
"subject": ["method", "keyword", "dictionary", "operator", "list", "keyword", "method", "variable", "syntax", "function"],
"ground_truth": ["append()", "lambda", "dictionary[key]", "**", "in", "try", "remove()", "global", "for item in iterable:", "sort()"],
"target_new": ["add()", "def", "dictionary.get(key)", "^", "contains()", "catch", "del()", "local", "for item in range(iterable):", "order()"]
}
Return only valid JSON output with these fields (do not output ```json ```). Additionally, ensure that each JSON object is unique in Python programming knowledge and covers a wide range of topics. In addition, the knowledge being edited needs to relate to the following task description and be critical syntax in the solution code even if the task is solved differently:
'''
    
    task_description = "  - Task: \n%s\n  - Solution: \n%s" % (entry["prompt"], entry["canonical_solution"])
    
    try:
        print(f"Processing {entry['task_id']}...")
        res_msg = get_answer(system_prompt, task_description)
    
        try:
            res_data = json.loads(res_msg)
            print(f"Generated JSON: {res_data}")
            return res_data
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return None
    
    except Exception as e:
        print(f"Error processing {entry['task_id']}: {e}")
        return None

def main():
    # 处理数据集
    processed_data = []
    for entry in openai_humaneval_dataset["test"]:
        generated_json = generate_dataset(entry)
        if generated_json:
            new_entry = dict(entry)
            new_entry["prompt_for_editing"] = generated_json["prompt"]
            new_entry["subject_for_editing"] = generated_json["subject"]
            new_entry["ground_truth_for_editing"] = generated_json["ground_truth"]
            new_entry["target_new_for_editing"] = generated_json["target_new"]
            processed_data.append(new_entry)

    # 输出到文件
    output_file = "./10knowledges.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    print(f"Dataset with editing questions saved to {output_file}.")


if __name__ == "__main__":
    main()
