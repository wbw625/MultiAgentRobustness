read -p "Please input the number of GPUs (e.g. 0,1): " gpus
read -p "Please input the number of port bias (e.g. 0): " port_bias
read -p "Please input the method of easyedit (ike/rome/mend): " edit_method
read -p "Please input the model of easyedit (llama/qwen/internlm): " edit_model

for i in $(seq 0 163); do
    echo "Processing HumanEval_$i"
    export PATH="~/.conda/envs/easyedit/bin:$PATH"
    python "$edit_method"/edit.py "$gpus" "$i" "$edit_model"
    export PATH="~/.conda/envs/fastchat/bin:$PATH"
    nohup python fastchat/restart_fastchat_api_two_models.py "$gpus" "$port_bias" &
    sleep 60
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
        "model": "edited_llama_rome",
        "base_url": "http://localhost:$((8006 + port_bias * 10))/v1",
        "api_type": "openai",
        "api_key": "EMPTY",
        "price" : [0, 0]
    }
]
EOF
    )
    export OAI_CONFIG_LIST="$CONFIG_LIST"
    export AUTOGENBENCH_ALLOW_NATIVE="yes"
    export PATH="~/.conda/envs/autogen/bin:$PATH"
    autogenbench run --serial-number "$i" --repeat 5 ./result/4llama_rome/HumanEval/Tasks/human_eval_TwoAgents.jsonl --native
    export PATH="~/.conda/envs/fastchat/bin:$PATH"
    python fastchat/stop_fastchat_api.py "$gpus" "$port_bias"
    sleep 6
    rm *.log*
    python fastchat/stop_fastchat_api.py "$gpus" "$port_bias"
    sleep 6
done
autogenbench tabulate ./result/4llama_rome/HumanEval/Results/human_eval_TwoAgents