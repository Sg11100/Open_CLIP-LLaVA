import torch
import torch.nn as nn
import open_clip
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from PIL import Image
from collections import namedtuple
from ...AlignCLIP import align_clip

# 添加适配器类
class OpenClipProcessorAdapter:
    def __init__(self, transform):
        self.transform = transform
    
    def preprocess(self, images, return_tensors=None):
        if isinstance(images, Image.Image):
            images = [images]
        
        # 应用变换
        pixel_values = [self.transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        
        return {'pixel_values': pixel_values}

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower  
        self.select_layer = args.mm_vision_select_layer 
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        print("use AlignCLIP vit") 
        
        # 添加hf-hub前缀
        if not self.vision_tower_name.startswith("hf-hub:"):
            self.vision_tower_name = "hf-hub:" + self.vision_tower_name
        
        # 创建模型并获取预处理器
        self.vision_tower, image_processor = align_clip.factory.create_model_from_pretrained(
            self.vision_tower_name,
            precision="fp32",  # 使用fp32避免转换问题
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 使用适配器包装transform
        self.image_processor = OpenClipProcessorAdapter(image_processor)
        
        # 确定hidden_size
        embed_dim = 768  # 默认值
        if hasattr(self.vision_tower, 'embed_dim'):
            embed_dim = self.vision_tower.embed_dim
        
        # 创建配置对象
        self._config_obj = type('obj', (object,), {
            'hidden_size': embed_dim,
            'image_size': 224,  
            'patch_size': 16,  # AlignCLIP使用16x16
        })
        
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        """直接从AlignCLIP的编码流程中提取特征"""
        if type(images) is list:
            image_features = []
            for image in images:
                # 直接处理单张图像
                image_feature = self._extract_features(image.to(device=self.device, dtype=self.dtype))
                image_features.append(image_feature)
        else:
            # 批量处理
            image_features = self._extract_features(images.to(device=self.device, dtype=self.dtype))
        
        # 统一转换为bfloat16
        image_features = image_features.to(device=self.device, dtype=torch.bfloat16)
        return image_features

    def _extract_features(self, image):
        """从模型中提取特定层的特征"""
        with torch.no_grad():
            # 第1步：获取视觉特征
            x = self.vision_tower.visual(image)
            
            # 第2步：调整维度并通过transformer处理
            x = x.permute(1, 0, 2)  
            
            # 收集每个transformer块的输出
            block_outputs = []
            for i, block in enumerate(self.vision_tower.transformer.resblocks):
                x = block(x)
                if i == len(self.vision_tower.transformer.resblocks) - 1 or i == self.select_layer:
                    # 存储所需层的输出
                    block_outputs.append(x)
            
            # 调整回原始维度
            features = block_outputs[0 if self.select_layer >= len(self.vision_tower.transformer.resblocks) else -1]
            features = features.permute(1, 0, 2)  
            
            # 根据select_feature选择特征
            if self.select_feature == 'patch':
                return features[:, 1:]  # 跳过CLS token
            elif self.select_feature == 'cls_patch':
                return features  # 使用所有token  
            elif self.select_feature == 'cls':
                return features[:, 0:1]  # 只用CLS token
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        #修改符合open clip的格式
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        #修改符合open clip的格式
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self._config_obj
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
