OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0 \
torchrun --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=1 \
eval.py --simplified