import torch
import torch.nn as nn
import open_clip
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from PIL import Image
from collections import namedtuple
from ...AlignCLIP import align_clip

#适配openai的processor
class OpenClipProcessorAdapter:
    def __init__(self, transform):
        self.transform = transform
        
        self.size = {'shortest_edge': 224} 
        self.crop_size = {'height': 224, 'width': 224}  
        
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
        
        # 处理标志
        self.do_center_crop = True  
        self.do_resize = True       
        self.do_rescale = True      
        self.do_normalize = True    
        self.do_convert_rgb = True 
        
        # 如果transform有其他配置，尝试从转换中提取实际参数
        if hasattr(transform, 'transforms'):
            for t in transform.transforms:
                # 从Resize提取size
                if hasattr(t, 'size') and t.__class__.__name__ == 'Resize':
                    if isinstance(t.size, int):
                        self.size = {'shortest_edge': t.size}
                # 从CenterCrop提取crop_size
                elif hasattr(t, 'size') and t.__class__.__name__ == 'CenterCrop':
                    if isinstance(t.size, tuple) and len(t.size) == 2:
                        self.crop_size = {'height': t.size[0], 'width': t.size[1]}
                    elif isinstance(t.size, int):
                        self.crop_size = {'height': t.size, 'width': t.size}
                # 从Normalize提取mean和std
                elif hasattr(t, 'mean') and hasattr(t, 'std') and t.__class__.__name__ == 'Normalize':
                    self.image_mean = t.mean
                    self.image_std = t.std
    
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
        
        # 使用适配器包装transform以适配openai的processor
        self.image_processor = OpenClipProcessorAdapter(image_processor)
        
        # 确定hidden_size
        embed_dim = 768  # 默认使用VIT-B-16的维度
        
        # 创建配置对象
        self._config_obj = type('obj', (object,), {
            'hidden_size': embed_dim,
            'image_size': 224,  
            'patch_size': 16,  
        })
        
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        """直接从AlignCLIP的编码流程中提取特征"""
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self._extract_features(image.to(device=self.device, dtype=self.dtype))
                image_features.append(image_feature)
        else:
            image_features = self._extract_features(images.to(device=self.device, dtype=self.dtype))
        
        # 统一转换为bfloat16
        image_features = image_features.to(device=self.device, dtype=torch.bfloat16)
        return image_features

    def _extract_features(self, image):
        """从模型中提取特定层的特征"""
        with torch.no_grad():
            #获取视觉特征
            x = self.vision_tower.visual(image)
            
            #调整维度并通过transformer处理
            x = x.permute(1, 0, 2)  
            
            # 正确处理负索引
            num_blocks = len(self.vision_tower.transformer.resblocks)
            target_layer = self.select_layer if self.select_layer >= 0 else num_blocks + self.select_layer
            target_layer = max(0, min(target_layer, num_blocks - 1))  # 确保索引有效

            selected_feature = None
            for i, block in enumerate(self.vision_tower.transformer.resblocks):
                x = block(x)
                if i == target_layer:
                    selected_feature = x.clone()

            if selected_feature is None:
                raise ValueError(f"无法从层{self.select_layer}提取特征 (计算得到的目标层: {target_layer})")
                    
            # 使用选定的特征
            features = selected_feature.permute(1, 0, 2)
            
            # 根据select_feature选择特征
            if self.select_feature == 'patch':
                return features[:, 1:]  
            elif self.select_feature == 'cls_patch':
                return features  
            elif self.select_feature == 'cls':
                return features[:, 0:1]  
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
