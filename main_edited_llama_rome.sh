read -p "Please input the number of GPUs (e.g. 0,1): " gpus
read -p "Please input the number of port bias (e.g. 0): " port_bias

for i in $(seq 0 163); do
    echo "Processing HumanEval_$i"
    /data1/jutj/.conda/envs/easyedit/bin/python edit_llama_rome.py "$gpus" "$i"
    export PATH="/data1/jutj/.conda/envs/fastchat/bin:$PATH"
    nohup python restart_fastchat_api_llama_rome.py "$gpus" "$port_bias" &
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
    export PATH="/data1/jutj/.conda/envs/autogen/bin:$PATH"
    autogenbench run --serial-number "$i" --repeat 5 ./result/4llama_rome/HumanEval/Tasks/human_eval_TwoAgents.jsonl --native
    export PATH="/data1/jutj/.conda/envs/fastchat/bin:$PATH"
    python stop_fastchat_api.py "$gpus" "$port_bias"
    sleep 6
    rm *.log*
    python stop_fastchat_api.py "$gpus" "$port_bias"
    sleep 6
done
autogenbench tabulate ./result/4llama_rome/HumanEval/Results/human_eval_TwoAgents