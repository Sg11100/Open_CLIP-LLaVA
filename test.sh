#!/bin/bash

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /lpai/models/llava-pre/25-02-14-2/vicuna-7b-v1.5 \
    --version plain \
    --data_path /lpai/dataset/mllm-dataset/24-09-28-1/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /lpai/dataset/mllm-dataset/24-09-28-1/LLaVA-Pretrain/images \
    --vision_tower /lpai/models/alignclip/clip30ecc12m/clip_vitb_30ep_cc12m/clip-vitb16-cc12m-epochs30/checkpoints/epoch_30.pt \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./test/test_clip \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

# #!/bin/bash

# deepspeed llava/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path /lpai/models/llava-pre/25-02-14-2/vicuna-7b-v1.5 \
#     --version v1 \
#     --data_path /lpai/dataset/llava-ft/0-1-4/llava_ft/llava_v1_5_mix665k.json \
#     --image_folder /lpai/dataset/llava-ft/0-1-4/llava_ft/data \
#     --vision_tower /lpai/models/llava-pre/25-02-14-2/clip-vit-large-patch14-336 \
#     --pretrain_mm_mlp_adapter /lpai/volumes/so-volume-ga/lhp/vicuna-7b-v1.5-pretrain/clip_vitl_336_base/mm_projector.bin \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir /lpai/test/checkpoints/llava-v1.5-13b \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb
