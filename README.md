# Open_CLIP-LLaVA
目前仅能跑通推理模式，训练模式并未跑通
目前问题：
- 由于openclip相比openai的简陋许多，直接使用其自动返回的transforms来处理图片输入
- 暂时直接固定使用float16精度
- 默认不执行s2，没有实现s2的修改
## Usage
前往hugging face下载相应的权重。并在其权重内部config修改为你想要使用的模型即可，例如mm_vision_tower": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K
