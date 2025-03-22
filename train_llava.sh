#!/bin/bash
# filepath: /lpai/LLaVA/train_llava.sh

# 默认参数
MODE="both"  # 可选: pretrain, finetune, both
MODEL_SIZE="7b"
CUDA_VISIBLE_DEVICES="all" 
OUTPUT_ROOT="/lpai/outputs/model"
CUSTOM_PROJECTOR=""  # 新增: 用户指定的projector路径

# 解析命令行参数
# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --model_size)
      MODEL_SIZE="$2"
      shift 2
      ;;
    --gpus)
      CUDA_VISIBLE_DEVICES="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --projector)
      CUSTOM_PROJECTOR="$2"
      shift 2
      ;;
    --help|-h)
      echo "LLaVA训练脚本 - 使用说明"
      echo ""
      echo "参数:"
      echo "  --mode VALUE           训练模式 (pretrain, finetune, both)"
      echo "  --model_size VALUE     模型大小 (默认: 7b)"
      echo "  --gpus VALUE           指定GPU (例如: 0,1,2,3, 默认: all)"
      echo "  --output_dir VALUE     输出目录 (默认: /lpai/outputs/model)"
      echo "  --projector VALUE      指定自定义projector文件路径"
      echo "  --help, -h             显示此帮助信息"
      echo ""
      echo "示例:"
      echo "  ./train_llava.sh --mode both --model_size 7b"
      echo "  ./train_llava.sh --mode finetune --projector /path/to/projector.bin"
      echo "  ./train_llava.sh --gpus 0,1 --output_dir /custom/output/path"
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 验证参数
if [[ "$MODE" != "pretrain" && "$MODE" != "finetune" && "$MODE" != "both" ]]; then
  echo "错误: mode参数必须是 'pretrain', 'finetune' 或 'both'"
  exit 1
fi

# 如果指定了自定义projector，则优先使用它并自动跳过预训练
if [[ ! -z "$CUSTOM_PROJECTOR" ]]; then
  if [[ ! -f "$CUSTOM_PROJECTOR" ]]; then
    echo "错误: 指定的projector文件不存在: $CUSTOM_PROJECTOR"
    exit 1
  fi
  
  echo "使用指定的projector: $CUSTOM_PROJECTOR"
  
  # 如果模式是both，则只执行微调
  if [[ "$MODE" == "both" ]]; then
    echo "由于指定了projector，将跳过预训练阶段"
    MODE="finetune"
  fi
fi

# 如果指定了GPU，则设置环境变量
if [[ "$CUDA_VISIBLE_DEVICES" != "all" ]]; then
  export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
  echo "使用GPU: $CUDA_VISIBLE_DEVICES"
fi

# 创建输出目录
mkdir -p ${OUTPUT_ROOT}
echo "输出目录: ${OUTPUT_ROOT}"

# 预训练输出目录
PRETRAIN_OUTPUT="${OUTPUT_ROOT}/llava-v1.5-${MODEL_SIZE}-pretrain"
# 微调输出目录
FINETUNE_OUTPUT="${OUTPUT_ROOT}/llava-v1.5-${MODEL_SIZE}"

# 预训练函数
run_pretrain() {
  echo "开始预训练阶段..."
  
  # 创建预训练输出目录
  mkdir -p ${PRETRAIN_OUTPUT}
  
  deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-${MODEL_SIZE}-v1.5 \
    --version plain \
    --data_path /lpai/dataset/llava-pre/0-1-0/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /lpai/dataset/llava-pre/0-1-0/LLaVA-Pretrain/images \
    --vision_tower sarahESL/AlignCLIP \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${PRETRAIN_OUTPUT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
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
    
  if [ $? -ne 0 ]; then
    echo "预训练阶段失败"
    exit 1
  fi
  
  echo "预训练阶段完成"
}

# 微调函数
run_finetune() {
  echo "开始微调阶段..."
  
  # 创建微调输出目录
  mkdir -p ${FINETUNE_OUTPUT}
  
  # 确定预训练适配器路径
  # 如果用户指定了projector，直接使用
  if [[ ! -z "$CUSTOM_PROJECTOR" ]]; then
    PRETRAIN_ADAPTER="$CUSTOM_PROJECTOR"
    echo "使用指定的projector: $PRETRAIN_ADAPTER"
  else
    
    PRETRAIN_DIR="${PRETRAIN_OUTPUT}"
    
    # 先尝试使用checkpoint-100中的适配器
    if [ -f "${PRETRAIN_DIR}/checkpoint-100/mm_projector.bin" ]; then
      PRETRAIN_ADAPTER="${PRETRAIN_DIR}/checkpoint-100/mm_projector.bin"
      echo "使用预训练检查点的projector: $PRETRAIN_ADAPTER"
    # 如果不存在，尝试使用根目录下的适配器
    elif [ -f "${PRETRAIN_DIR}/mm_projector.bin" ]; then
      PRETRAIN_ADAPTER="${PRETRAIN_DIR}/mm_projector.bin"
      echo "使用预训练目录的projector: $PRETRAIN_ADAPTER"
    else
      # 如果在"仅微调"模式下且找不到适配器文件
      if [ "$MODE" == "finetune" ]; then
        echo "警告: 未找到预训练适配器"
        read -p "是否继续微调? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
          echo "微调已取消"
          exit 1
        fi
        PRETRAIN_ADAPTER=""
      else
        echo "错误: 无法找到预训练适配器文件"
        exit 1
      fi
    fi
  fi
  
  # 构建微调命令
  CMD="deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero22.json \
    --model_name_or_path lmsys/vicuna-${MODEL_SIZE}-v1.5 \
    --version v1 \
    --data_path /lpai/dataset/llava-ft/0-1-4/llava_ft/llava_v1_5_mix665k.json \
    --image_folder /lpai/dataset/llava-ft/0-1-4/llava_ft/data \
    --vision_tower sarahESL/AlignCLIP"
  
  # 如果有预训练适配器，添加到命令中
  if [ ! -z "$PRETRAIN_ADAPTER" ]; then
    CMD="$CMD \
    --pretrain_mm_mlp_adapter $PRETRAIN_ADAPTER"
  fi
  
  # 添加其余参数
  CMD="$CMD \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${FINETUNE_OUTPUT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy \"no\" \
    --save_strategy \"steps\" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type \"cosine\" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb"
  
  # 执行命令
  eval $CMD
    
  if [ $? -ne 0 ]; then
    echo "微调阶段失败"
    exit 1
  fi
  
  echo "微调阶段完成"
}

# 主程序
echo "LLaVA 训练启动 - 模式: $MODE, 模型大小: ${MODEL_SIZE}"
echo "输出将保存到: ${OUTPUT_ROOT}"

if [[ "$MODE" == "pretrain" || "$MODE" == "both" ]]; then
  run_pretrain
fi

if [[ "$MODE" == "finetune" || "$MODE" == "both" ]]; then
  run_finetune
fi

echo "LLaVA 训练完成!"