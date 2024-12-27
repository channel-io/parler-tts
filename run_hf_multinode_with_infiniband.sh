export MLFLOW_TRACKING_URI=https://mlflow.exp.channel.io/
export MLFLOW_EXPERIMENT_NAME=parler-tts-ko
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0          # Infiniband를 활성화
export NCCL_IB_HCA=mlx5_0         # Infiniband HCA 이름 (예시, 실제 환경에 맞게 수정 필요)
export NCCL_IB_GID_INDEX=0        # Infiniband GID Index (예시, 실제 환경에 맞게 수정 필요)
export NCCL_IB_SL=0               # Infiniband Service Level 설정 (기본값: 0)
export NCCL_SOCKET_IFNAME=eth0    # 네트워크 인터페이스 이름을 지정 (Infiniband와 함께 필요 시 사용)
export NCCL_P2P_LEVEL=NVL         # P2P 통신을 GPU 사이에서만 제한
export NCCL_TIMEOUT=10000000
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# --config_file ./helpers/multinode/accelerate_config_with_ds.yaml \
accelerate launch --config_file ./helpers/multinode/multinode_with_ds.yaml ./training/run_parler_tts_training.py --args_yaml sharded_args.yaml