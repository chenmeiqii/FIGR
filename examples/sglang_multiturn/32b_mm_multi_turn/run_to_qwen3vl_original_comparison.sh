# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
HOME=xxx
DATA_HOME=$HOME/data/rl_data
MODEL_PATH=$HOME/models/Qwen3-VL-32B-Instruct

PROJ_NAME=mm_cot
EXP_NAME=qwen3vl-32b_no_tool_32k
SAVE_PATH=$PROJECT_DIR/checkpoints/${PROJ_NAME}/${EXP_NAME}
mkdir -p ${SAVE_PATH}

nohup python3 -m verl.trainer.main_ppo \
        custom_reward_function.path=$PROJECT_DIR/verl/utils/reward_score/mm_multi_turn_ori.py \
        custom_reward_function.name=compute_score \
        algorithm.adv_estimator=grpo \
        algorithm.kl_ctrl.kl_coef=0.0 \
        data.train_batch_size=128 \
        data.max_prompt_length=8192 \
        data.max_response_length=16384 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.use_fused_kernels=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.max_num_batched_tokens=40960 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.nccl_timeout=36000 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        trainer.critic_warmup=0 \
        trainer.logger='["wandb"]' \
        trainer.val_before_train=True \
        trainer.project_name=${PROJ_NAME} \
        trainer.experiment_name=${EXP_NAME} \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=${WORLD_SIZE} \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        data.train_files=$HOME/data/rl_data/train/train.parquet \
        trainer.total_epochs=1 \
        trainer.default_local_dir=${SAVE_PATH} \
        2>&1 | tee ${SAVE_PATH}/train.log.$(date +%Y-%m-%d-%H-%M)
