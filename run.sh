CUDA_VISIBLE_DEVICES=1 python3 main.py --print_config --log_dir /tmp/exp1 \
    --config configs/mat-exp3/test_CAM_300_Out_0.2_miss_0.4.toml \
    --config configs/opt/SGD.toml \
    --set depth 2 \
    --set loss_fn l1 \
    --set reg_term_weight 1.0