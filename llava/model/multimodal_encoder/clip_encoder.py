import torch
import torch.nn as nn
from .. import open_clip
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from PIL import Image
from collections import namedtuple

# 添加适配器类
class OpenClipProcessorAdapter:
    def __init__(self, transform):
        self.transform = transform
        
        
        self.size = {'shortest_edge': 224}  
        self.crop_size = {'height': 224, 'width': 224}  
        
        self.image_mean = [0.5, 0.5, 0.5]  
        self.image_std = [0.5, 0.5, 0.5]
        
        
        self.do_center_crop = True  
        self.do_resize = True       
        self.do_rescale = True      
        self.do_normalize = True    
        self.do_convert_rgb = True  
        
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
        
        self.vision_tower, image_processor = open_clip.create_model_from_pretrained(
            model_name='ViT-B-16',
            pretrained=self.vision_tower_name,
            device='cuda',
            precision='fp32'  # 改为fp32避免数据类型问题
        )      
        # 使用适配器包装open_clip的图像处理器
        self.image_processor = OpenClipProcessorAdapter(image_processor)
        
        # 正确检测嵌入维度
        if hasattr(self.vision_tower, 'visual') and hasattr(self.vision_tower.visual, 'conv1'):
            # 对于ViT模型，可以从卷积层输出维度获取
            embed_dim = self.vision_tower.visual.conv1.out_channels
            print(f"Detected embed_dim: {embed_dim}")
        elif hasattr(self.vision_tower, 'visual') and hasattr(self.vision_tower.visual, 'embed_dim'):
            embed_dim = self.vision_tower.visual.embed_dim
            print(f"Detected embed_dim from visual.embed_dim: {embed_dim}")
        elif hasattr(self.vision_tower, 'embed_dim'):
            embed_dim = self.vision_tower.embed_dim
            print(f"Detected embed_dim from model.embed_dim: {embed_dim}")
        else:
            embed_dim = 768 #ViT-B的默认维度
            print(f"Using default embed_dim: {embed_dim}")
        
        # 创建配置对象
        self._config_obj = type('obj', (object,), {
            'hidden_size': embed_dim,
            'image_size': 224,
            'patch_size': 16,  
        })
        
        #冻结参数
        self.vision_tower.requires_grad_(False)
        
        # 收集隐藏状态
        self.hidden_states = []
        
        def collect_hidden_states(module, input, output):
            self.hidden_states.append(output)
        
        # 为每个ResidualAttentionBlock注册钩子
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
        # 获取模型权重的数据类型
        model_dtype = next(self.vision_tower.parameters()).dtype
        
        if type(images) is list:
            image_features = []
            for image in images:
                self.hidden_states = []
                
                # 将输入与模型权重保持相同的数据类型
                image_same_dtype = image.to(device=self.device, dtype=model_dtype)
                image_forward_out = self.vision_tower.encode_image(image_same_dtype)
                
                class ModelOutput:
                    def __init__(self, hidden_states):
                        self.hidden_states = hidden_states
                
                output = ModelOutput(self.hidden_states)
                
                image_feature = self.feature_select(output)
                image_features.append(image_feature)
        else:
            self.hidden_states = []
            
            # 将输入与模型权重保持相同的数据类型
            images_same_dtype = images.to(device=self.device, dtype=model_dtype)
            image_forward_outs = self.vision_tower.encode_image(images_same_dtype)
            
            class ModelOutput:
                def __init__(self, hidden_states):
                    self.hidden_states = hidden_states
            
            output = ModelOutput(self.hidden_states)
            
            image_features = self.feature_select(output)
        
        # 确保返回的特征是您需要的数据类型(如bfloat16)
        target_dtype = torch.bfloat16  
        image_features = image_features.to(device=self.device, dtype=target_dtype)
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
