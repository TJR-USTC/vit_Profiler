import sys
sys.path.append(r"./")
# from model.resnet import resnet18, resnet50
from model.vit import vit_base_patch16_224 as vit

model_dict = {
    'vit': vit,
}

def create_model(model_name, num_classes, pretrained):   
    return model_dict[model_name](num_classes = num_classes, pretrained = pretrained)