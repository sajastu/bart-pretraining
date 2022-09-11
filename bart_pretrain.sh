
export HF_DATASETS_CACHE="/disk0/sajad/.cache/huggingface/datasets/"
#CUDA_VISIBLE_DEVICES=0 python modeling/run_bart_dlm_flax.py \
#    --model_name_or_path="facebook/bart-large" \
#    --output_dir="/disk0/sajad/pretrained_exps/socialSig-bart-large" \
#    --train_file="datasets/pretraining/data/train.json" \
#    --validation_file="datasets/pretraining/data/val.json" \
#    --max_seq_length="1024" \
#    --per_device_train_batch_size="4" \
#    --per_device_eval_batch_size="4" \
#    --learning_rate="3e-5" \
#    --warmup_steps="2000" \
#    --overwrite_output_dir \
#    --logging_steps="500" \
#    --save_steps="2000" \
#    --eval_steps="2000" \
#export JAX_DISABLE_JIT=1
export CUDA_VISIBLE_DEVICES=0,1
python modeling/run_bart_dlm_flax.py \
    --model_name_or_path="/disk0/sajad/pretrained_exps/bart-reddit-2nd/" \
    --output_dir="/disk0/sajad/pretrained_exps/bart-reddit-webis/" \
    --train_file="/disk0/sajad/.cache/datasets/webist_tldr/pretraining/data/train.json" \
    --validation_file="/disk0/sajad/.cache/datasets/webist_tldr/pretraining//data/val.json" \
    --max_seq_length="1024" \
    --per_device_train_batch_size="4" \
    --per_device_eval_batch_size="4" \
    --learning_rate="3e-5" \
    --warmup_steps="2000" \
    --overwrite_output_dir \
    --save_steps="90720" \
    --eval_steps="90720" \
    --num_train_epochs="5" \
    --logging_steps 100