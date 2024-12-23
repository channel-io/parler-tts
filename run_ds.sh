export MLFLOW_TRACKING_URI=https://mlflow.exp.channel.io/
export MLFLOW_EXPERIMENT_NAME=parler-tts-ko
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1          # Infiniband를 사용하지 않는 경우 비활성화
export NCCL_P2P_LEVEL=NVL         # P2P 통신을 GPU 사이에서만 제한
export NCCL_SOCKET_IFNAME=eth0    # 네트워크 인터페이스 이름을 지정 (예: eth0)
export NCCL_TIMEOUT=3600

HOSTFILE="helpers/multinode/hostfile"
DS_CONFIG_PATH="helpers/multinode/deepspeed_config_zero2.json"

deepspeed --num_gpus 4 --num_nodes 4 --hostfile $HOSTFILE --master_addr main1 training/run_parler_tts_training.py --args_yaml args.yaml

