import argparse
from restart_fastchat_api_two_models import kill_process

def main():
    parser = argparse.ArgumentParser(description="Kill processes based on GPUs and port bias.")
    parser.add_argument("gpus", type=str, help="Comma-separated list of GPU IDs (e.g., '0,1').")
    parser.add_argument("port_bias", type=int, help="The port bias to be used.")
    args = parser.parse_args()

    gpus = args.gpus
    port_bias = args.port_bias

    kill_process(gpus, port_bias)
    print(f"All processes with port bias {port_bias} have been killed.")

if __name__ == "__main__":
    main()
