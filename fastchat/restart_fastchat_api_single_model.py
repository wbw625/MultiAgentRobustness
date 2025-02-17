import os
import socket
import subprocess
import time
import argparse
import torch

# 对应的端口号
llama3_port = 8008
server_port = 8006
controller_port = 21001

def is_port_in_use(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

def restart_process(command, port):
    """检测端口并重启进程"""
    if is_port_in_use(port):
        print(f"端口 {port} 已被占用，无需重启")
        return None
    print(f"端口 {port} 未占用，正在启动进程...")
    new_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"进程已启动，端口 {port} 正在监听")
    return new_process

def start_all(gpus="0,1", port_bias=0):

    llama3_port = 8008
    server_port = 8006
    controller_port = 21001

    gpu1, gpu2 = gpus.split(",")
    controller_port += port_bias*10
    llama3_port += port_bias*10
    server_port += port_bias*10

    # 命令定义
    cmd_controller = (
        f"/data1/jutj/.conda/envs/fastchat/bin/python -m fastchat.serve.controller --port {controller_port}"
    )

    cmd_llama3 = (
        f"CUDA_VISIBLE_DEVICES={gpu1} /data1/jutj/.conda/envs/fastchat/bin/python -m fastchat.serve.model_worker "
        "--model-path ./models/Llama-3.1-8B-Instruct "
        f"--controller http://localhost:{controller_port} "
        f"--port {llama3_port} --worker http://localhost:{llama3_port}"
    )

    cmd_server = (
        "/data1/jutj/.conda/envs/fastchat/bin/python -m fastchat.serve.openai_api_server "
        "--host localhost "
        f"--port {server_port} "
        f"--controller http://localhost:{controller_port}"
    )

    process_controller = subprocess.Popen(cmd_controller, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Controller已启动")
    time.sleep(10)

    process_llama3 = subprocess.Popen(cmd_llama3, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("llama3已启动")
    time.sleep(35)

    process_server = subprocess.Popen(cmd_server, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Server已启动")
    time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description="Start processes with specified GPUs and port bias.")
    parser.add_argument("gpus", type=str, help="Comma-separated list of GPU IDs (e.g., '0,1').")
    parser.add_argument("port_bias", type=int, help="The port bias to be used.")
    args = parser.parse_args()

    gpus = args.gpus
    port_bias = args.port_bias

    start_all(gpus, port_bias)
    print(f"Initialized all processes with GPUs: {gpus} and port bias: {port_bias}.")

    llama3_port = 8008
    server_port = 8006
    controller_port = 21001

    controller_port += port_bias*10
    llama3_port += port_bias*10
    server_port += port_bias*10

    while True:
        if not is_port_in_use(server_port):
            print("Server process has stopped. Killing all processes...")
            time.sleep(10)
            kill_process(gpus, port_bias)
            break
        time.sleep(5)
    exit()


def kill_process(gpus="0,1", port_bias=0):
    """手动调用时，根据端口关闭进程"""
    llama3_port = 8008
    server_port = 8006
    controller_port = 21001
    
    controller_port += port_bias*10
    llama3_port += port_bias*10
    server_port += port_bias*10
    
    for port in [server_port, controller_port, llama3_port]:
        if is_port_in_use(port):
            print(f"正在关闭端口 {port} 的进程...")
            kill_process_on_port(port)
    
    gpu1, gpu2 = gpus.split(",")
    if gpu1 == gpu2:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu1
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.empty_cache()

def kill_process_on_port(port):
    """通过端口号找到并杀死进程"""
    try:
        result = subprocess.check_output(["lsof", "-t", f"-i:{port}"])
        pids = result.decode().strip().split('\n')
        current_pid = str(os.getpid())  # 获取当前脚本的 PID
        
        for pid in pids:
            if pid == current_pid:
                print(f"跳过当前脚本 PID {pid}")
                continue
            subprocess.run(["kill", "-9", pid])
            print(f"已关闭 PID {pid} 占用的端口 {port}")
    except subprocess.CalledProcessError:
        print(f"未找到占用端口 {port} 的进程")

if __name__ == "__main__":
    main()
