# MultiAgentRobustness

## Load dataset
Enter the result directory and run the following command to download the dataset.
```shell
autogenbench clone HumanEval
```
Edit `./result/DIR_NAME/HumanEval/Templates/TwoAgents/scenario.py`. Copy from `./autogenbench/autogenbench/scenarios/HumanEval/Templates/TwoAgents/scenario.py` to the `scenario.py` in the result directory. Change the following line in the `scenario.py` file (Optional).
```python
config_list1 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"model": ["Llama-3.1-8B-Instruct"]}
)

config_list2 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"model": ["Qwen2.5-7B-Instruct"]}
)
```
and
```python
coder1 = autogen.AssistantAgent(
    "coder1",
    system_message=code_writer_system_message,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=testbed_utils.default_llm_config(config_list1, timeout=360),
)

coder2 = autogen.AssistantAgent(
    "coder2",
    system_message=code_writer_system_message,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=testbed_utils.default_llm_config(config_list2, timeout=360),
)
```
etc.


## Start the main sh file
Edit `./main_edited_llama_rome.sh` file. Change the following line to the correct path.
```shell
/data1/jutj/.conda/envs/easyedit/bin/python edit_ike.py "$gpus" "$i"
```
```shell
export PATH="/data1/jutj/.conda/envs/fastchat/bin:$PATH"
```
```shell
autogenbench run --serial-number "$i" --repeat 5 ./result/ike/HumanEval/Tasks/human_eval_TwoAgents.jsonl --native
```
```shell
autogenbench tabulate ./result/ike/HumanEval/Results/human_eval_TwoAgents
```
```shell
CONFIG_LIST=$(cat <<EOF
[
    {
        "model": "Llama-3.1-8B-Instruct",
        "base_url": "http://localhost:$((8006 + port_bias * 10))/v1",
        "api_type": "openai",
        "api_key": "EMPTY",
        "price" : [0, 0]
    },
    {
        "model": "edited_model_ike",
        "base_url": "http://localhost:$((8006 + port_bias * 10))/v1",
        "api_type": "openai",
        "api_key": "EMPTY",
        "price" : [0, 0]
    }
]
EOF
    )
```

Run the following command to start the main sh file.

```shell
./main_edited_llama_rome.sh
```

Select the `gpus` `port_bias`. 

The `gpus` is the number of GPUs you want to use. The format is like "0,1". Or, if you use `restart_fastchat_api_gemma.py`, the format is like "0,1,2".

The `port_bias` is the port bias you want to use. The port bias is used to avoid port conflicts. The default value is 0. The `port_bias` is used to calculate the port number. For example, the server port number is calculated as `8006 + port_bias * 10`.


## citation

Please cite our paper if you use the code in your work.

```bibtex
@misc{ju2025investigatingadaptiverobustnessknowledge,
      title={Investigating the Adaptive Robustness with Knowledge Conflicts in LLM-based Multi-Agent Systems}, 
      author={Tianjie Ju and Bowen Wang and Hao Fei and Mong-Li Lee and Wynne Hsu and Yun Li and Qianren Wang and Pengzhou Cheng and Zongru Wu and Zhuosheng Zhang and Gongshen Liu},
      year={2025},
      eprint={2502.15153},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.15153}, 
}
```