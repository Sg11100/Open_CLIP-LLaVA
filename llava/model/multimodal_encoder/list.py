import open_clip

# 列出所有可用模型和检查点
model_list = open_clip.list_pretrained()
print(model_list)
#('ViT-L-14', 'laion2b_s32b_b82k')