# Open_CLIP-LLaVA
推理模式，pretrain与finetune均可运行
目前问题：
- 由于openclip相比openai的简陋许多，直接使用其自动返回的transforms来处理图片输入
- 暂时直接固定使用float16精度
- 默认不执行s2，没有实现s2的修改
## Usage
1. 推理：前往hugging face下载相应的权重。并在其权重内部config修改为你想要使用的模型即可，例如mm_vision_tower": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K
2. 预训练与微调：修改scripts/v1_5目录中的 pretrain.sh与finetune.sh中的--vision_tower、--image_folder、--data_path为相应的内容
