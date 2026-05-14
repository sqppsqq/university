# -*- coding:UTF-8 -*- #
"""
@filename:model.py
@author:Kiki Shen
@time:2024-04-11
"""
import torch
from torchvision import models
from torchvision import transforms
from torch.autograd import Variable
def init_model(model_name):
    if model_name=='GoogleNet':
        model=models.googlenet(pretrained=True)
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    elif model_name=='inception':
        model = models.inception_v3(pretrained=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif model_name=='resnet':
        model = models.resnet50(pretrained=True)

    elif model_name=='vgg':
        model = models.vgg19(pretrained=True)

    elif model_name=='squeezenet':
        model = models.squeezenet1_0(pretrained=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(DEVICE)
    return model,transform

def get_pred(image,model,transform):
    """
    获取图像的分类结果
    Args:
        image: 图像，image
        model: 分类模型

    Returns:预测类别

    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = Variable(torch.unsqueeze(transform(image), dim=0), requires_grad=False).to(DEVICE)
    out = model(input)
    #
    _, index = torch.max(out, 1)
    #
    pred_class = index[0]
    return pred_class