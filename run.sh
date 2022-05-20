
#for ds in Ellis_Island Alamo Gendarmenmarkt Madrid_Metropolis Montreal_Notre_Dame Notre_Dame NYC_Library Piazza_del_Popolo Tower_of_London Trafalgar Union_Square Yorkminster Roman_Forum
for ds in Gendarmenmarkt
do
  CUDA_VISIBLE_DEVICES=1 python3 main.py --print_config --log_dir /tmp/exp1 \
      --config configs/mat-test/$ds.toml \
      --config configs/opt/SGD.toml \
      --set loss_fn l1 \
      --set project_name dmc_debug \
      --set run_name REPLICATION_$ds \
      --set wandb True \
      --set reg_term_weight 0.0 \
      --set lr 0.3 \
      --set init_scale 1e-3 \
      --set initialization gaussian \
      --set n_iters 100000 \
      --set depth 5 \
      --set seed 42
done