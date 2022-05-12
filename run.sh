for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
  CUDA_VISIBLE_DEVICES=1 python3 main.py --print_config --log_dir /tmp/exp1 \
      --config configs/mat-test/Ellis_Island.toml \
      --config configs/opt/SGD.toml \
      --set depth 2 \
      --set loss_fn l1 \
      --set project_name dmc_debug \
      --set run_name [4]_dmc_real_lambda=$i \
      --set wandb True \
      --set reg_term_weight $i \
      --set lr 0.3 \
      --set init_scale 1e-3 \
      --set initialization gaussian \
      --set n_iters 150000 \
      --set depth 5
done

#for i in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#do
#  CUDA_VISIBLE_DEVICES=1 python3 main.py --print_config --log_dir /tmp/exp1 \
#      --config configs/mat-test/Ellis_Island.toml \
#      --config configs/opt/SGD.toml \
#      --set depth 2 \
#      --set loss_fn l1 \
#      --set project_name dmc_debug \
#      --set run_name [3]_dmc_real_lambda=$i \
#      --set wandb True \
#      --set reg_term_weight $i \
#      --set lr 0.3 \
#      --set init_scale 1e-3 \
#      --set initialization gaussian \
#      --set n_iters 100000 \
#      --set depth 5
#done