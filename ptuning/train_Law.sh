PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_train \
    --train_file ../data/Law/train.txt \
    --validation_file ../data/Law/dev.txt \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path ../ckpts/chatglm-6b \
    --output_dir ../output/Law-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 500 \
    --max_target_length 500 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

