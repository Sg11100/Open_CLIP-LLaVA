import torch
import torch.nn as nn
import open_clip
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from PIL import Image
from collections import namedtuple

# 添加适配器类
class OpenClipProcessorAdapter:
    def __init__(self, transform):
        self.transform = transform
        
        # 从transform中提取参数
        self.size = {'shortest_edge': 224}  # 从Resize(size=224)
        self.crop_size = {'height': 224, 'width': 224}  # 从CenterCrop(size=(224, 224))
        
        # 提取Normalize参数
        self.image_mean = [0.5, 0.5, 0.5]  # 从Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.image_std = [0.5, 0.5, 0.5]
        
        # 处理标志
        self.do_center_crop = True  # 有CenterCrop
        self.do_resize = True       # 有Resize
        self.do_rescale = True      # ToTensor会将[0,255]缩放到[0,1]
        self.do_normalize = True    # 有Normalize
        self.do_convert_rgb = True  # 有_convert_to_rgb
        
        # 如果transform有其他配置，尝试更准确地提取
        if hasattr(transform, 'transforms'):
            for t in transform.transforms:
                # 从Resize提取size
                if hasattr(t, 'size') and t.__class__.__name__ == 'Resize':
                    if t.size:
                        self.size = {'shortest_edge': t.size}
                # 从CenterCrop提取crop_size
                elif hasattr(t, 'size') and t.__class__.__name__ == 'CenterCrop':
                    if t.size:
                        self.crop_size = {'height': t.size[0], 'width': t.size[1]}
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
        print("use open_clip vit") 
        self.vision_tower_name = "hf-hub:"+self.vision_tower_name #替换vision tower 和 image_processor
        self.vision_tower, image_processor = open_clip.create_model_from_pretrained(self.vision_tower_name)
        print(image_processor)
        print("genshen")
        # 使用适配器包装open_clip的图像处理器，使其具有preprocess方法
        self.image_processor = OpenClipProcessorAdapter(image_processor)
        
        print(self.vision_tower)
        # 确定正确的hidden_size值
        if hasattr(self.vision_tower, 'embed_dim'):
            embed_dim = self.vision_tower.embed_dim
        elif hasattr(self.vision_tower, 'visual') and hasattr(self.vision_tower.visual, 'embed_dim'):
            embed_dim = self.vision_tower.visual.embed_dim
        else:
            # 如果无法从模型直接获取，使用默认值
            embed_dim = 1024  # ViT-L 的典型维度
            print(f"Warning: Could not determine embed_dim from model, using default: {embed_dim}")
        
        # 创建配置对象
        self._config_obj = type('obj', (object,), {
            'hidden_size': embed_dim,
            'image_size': 224,  
            'patch_size': 14,   
        })
        
        # 将模型移到 GPU
        if torch.cuda.is_available():
            self.vision_tower = self.vision_tower.cuda()
        
        self.vision_tower.requires_grad_(False)
        
        self.hidden_states = []
        
        def collect_hidden_states(module, input, output):
            self.hidden_states.append(output)
        #每一层都添加
        for block in self.vision_tower.visual.transformer.resblocks:
            block.register_forward_hook(collect_hidden_states)
            
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                self.hidden_states = []
                
                image_forward_out = self.vision_tower.encode_image(image.to(device=self.device, dtype=self.dtype))
                
                class ModelOutput:
                    def __init__(self, hidden_states):
                        self.hidden_states = hidden_states
                
                output = ModelOutput(self.hidden_states)
                
                image_feature = self.feature_select(output).to(image.dtype)
                image_features.append(image_feature)
        else:
            self.hidden_states = []
            
            image_forward_outs = self.vision_tower.encode_image(images.to(device=self.device, dtype=self.dtype))
            
            class ModelOutput:
                def __init__(self, hidden_states):
                    self.hidden_states = hidden_states
            
            output = ModelOutput(self.hidden_states)
            
            image_features = self.feature_select(output).to(images.dtype)
        
        # 修改这一行，使用bfloat16而不是float16
        image_features = image_features.to(device=self.device, dtype=torch.bfloat16)
        return image_features

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
