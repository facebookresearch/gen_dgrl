# Training on human demonstrations

## BC

python train_choice_il.py \
--xpid=il_bs1_lr2e-05_0 \
--task_name=mrpc \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=8 \
--learning_rate=2e-05 \
--output_dir=./ckpts/web_click \
--image=0 \
--seed=0

python train_choice_il.py \
--xpid=il_bs1_lr2e-05_1 \
--task_name=mrpc \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=8 \
--learning_rate=2e-05 \
--output_dir=./ckpts/web_click \
--image=0 \
--seed=1

python train_choice_il.py \
--xpid=il_bs1_lr2e-05_2 \
--task_name=mrpc \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=8 \
--learning_rate=2e-05 \
--output_dir=./ckpts/web_click \
--image=0 \
--seed=2

## CQL 

python train_choice_cql.py \
--xpid=cql_bs1_lr2e-05_a7.0_g0.99_tuf1000_tau0.005_0 \
--task_name=mrpc \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=8 \
--learning_rate=2e-05 \
--output_dir=./ckpts/web_click \
--image=0 \
--seed=0 \
--gamma=0.99 \
--target_update_freq=1000 \
--target_model_tau=0.005 \
--cql_alpha=7.0

python train_choice_cql.py \
--xpid=cql_bs1_lr2e-05_a7.0_g0.99_tuf1000_tau0.005_1 \
--task_name=mrpc \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=8 \
--learning_rate=2e-05 \
--output_dir=./ckpts/web_click \
--image=0 \
--seed=1 \
--gamma=0.99 \
--target_update_freq=1000 \
--target_model_tau=0.005 \
--cql_alpha=7.0

python train_choice_cql.py \
--xpid=cql_bs1_lr2e-05_a7.0_g0.99_tuf1000_tau0.005_2 \
--task_name=mrpc \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=8 \
--learning_rate=2e-05 \
--output_dir=./ckpts/web_click \
--image=0 \
--seed=2 \
--gamma=0.99 \
--target_update_freq=1000 \
--target_model_tau=0.005 \
--cql_alpha=7.0

## BCQ

python train_choice_bcq.py \
--xpid=bcq_bs1_lr2e-05_a0.5_g0.99_tuf100_tau0.005_0 \
--task_name=mrpc \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=8 \
--learning_rate=2e-05 \
--output_dir=./ckpts/web_click \
--image=0 \
--seed=0 \
--gamma=0.99 \
--target_update_freq=100 \
--target_model_tau=0.005 \
--bcq_alpha=0.5

python train_choice_bcq.py \
--xpid=bcq_bs1_lr2e-05_a0.5_g0.99_tuf100_tau0.005_1 \
--task_name=mrpc \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=8 \
--learning_rate=2e-05 \
--output_dir=./ckpts/web_click \
--image=0 \
--seed=1 \
--gamma=0.99 \
--target_update_freq=100 \
--target_model_tau=0.005 \
--bcq_alpha=0.5

python train_choice_bcq.py \
--xpid=bcq_bs1_lr2e-05_a0.5_g0.99_tuf100_tau0.005_2 \
--task_name=mrpc \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=8 \
--learning_rate=2e-05 \
--output_dir=./ckpts/web_click \
--image=0 \
--seed=2 \
--gamma=0.99 \
--target_update_freq=100 \
--target_model_tau=0.005 \
--bcq_alpha=0.5

# Training on IL Scaling Data (same hyperparameters were used for all 6 IL datasets)
# You would have to set --num_trajs accordingly and change PATH global variable in the corresponding train_choice_{algo}.py file to the path of the dataset

## BC
python train_choice_il.py \
--xpid=il_bs1_lr2e-05_0 \
--task_name=mrpc \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=8 \
--learning_rate=2e-05 \
--output_dir=./ckpts/il_trajs_100 \
--image=0 \
--seed=0 \
--num_trajs=100

## CQL 

python train_choice_cql.py \
--xpid=cql_bs1_lr2e-05_a4.0_g0.99_tuf100_tau0.005_0 \
--task_name=mrpc \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=8 \
--learning_rate=2e-05 \
--output_dir=./ckpts/il_trajs_100 \
--image=0 \
--seed=0 \
--gamma=0.99 \
--target_update_freq=100 \
--target_model_tau=0.005 \
--cql_alpha=4.0 \
--num_trajs=100


## BCQ

python train_choice_bcq.py \
--xpid=bcq_bs1_lr2e-05_a0.9_g0.99_tuf1000_tau0.005_0 \
--task_name=mrpc \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=8 \
--learning_rate=2e-05 \
--output_dir=./ckpts/il_trajs_100 \
--image=0 \
--seed=0 \
--gamma=0.99 \
--target_update_freq=1000 \
--target_model_tau=0.005 \
--bcq_alpha=0.9 \
--num_trajs=100
