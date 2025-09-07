accelerate launch train_nextstep.py \
    --model_name_or_path "./HF_checkpoint/NextStep-1-Large" \
    --vae_name_or_path "./HF_checkpoint/NextStep-1-f8ch16-Tokenizer" \
    --tokenizer_name_or_path "./HF_checkpoint/Qwen2.5-14B-Instruct/qwen/Qwen2___5-14B-Instruct" \
    --dataset_name "./dataset/BLIP3o-60k-harmon-format/data_info.json" \
    \
    --output_dir "./nextstep-finetune-output" \
    --report_to "swanlab" \
    --run_name "swandb_training_nextstep1" \
    \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --seed 42 \
    \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --bf16 True \
    --gradient_checkpointing True \
    --deepspeed "zero2.json" \
    \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    \
    --resolutions 256 \
    --remove_unused_columns False
