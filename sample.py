# -*- coding:UTF-8 -*- #
"""
@filename:sample.py
@author:Kiki Shen
@time:2024-04-11
@description:对抗网络
"""
import os.path
import random

import numpy as np
import torch
from model import *
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

from attacks import torchattacks
from PIL import Image
class SQQ():
    """
    input:输入的原始图像
    watermark:输入的水印图像
    return:对抗样本
    """
    def __init__(self,input_image,class_index,image_name,wm_image,sl,alpha,eps1,eps2,model,transform):
        """
        :param input_image: 输入的原始图像
        :param wm_image: 原始水印图片
        :param sl: 比例因子
        :param alpha: 透明度
        :param wm_add_pos:水印添加位置的左上角坐标
        """
        super(SQQ,self).__init__()
        self.input_image=input_image
        # self.class_name=class_name
        self.class_index=class_index
        self.image_name=image_name
        self.wm_image=wm_image
        self.sl=sl
        self.alpha=alpha
        self.input_size=input_image.size#input_size[weight(x),height(y)]
        self.wm_size=wm_image.size
        self.wm_new_size, self.wm_new_image = self.get_wm_new_size()
        self.wm_add_pos=self.position_decision()
        self.save_address=self.result_save(image_name)
        # self.save_address = "./"
        self.eps1=eps1
        self.eps2=eps2
        self.model=model
        self.transform=transform

    def position_decision(self):
        #目前确定水印添加的位置
        N=12
        add_position=[]
        # wm_new_size,=self.get_wm_new_size()
        #w是宽，h是高
        w = self.input_image.size[0]
        h = self.input_image.size[1]
        w_range=[[0,0.2*w-self.wm_new_size[0]],[0.8*w,w-self.wm_new_size[0]]]
        h_range=[[0,0.2*h-self.wm_new_size[1]],[0.8*h,h-self.wm_new_size[1]]]
        for i in range(N):
            #左侧
            if i<int(N/4):
                add_list=[]
                add_list.append(w_range[0][0] + random.random() * (w_range[0][1] - w_range[0][0]))
                add_list.append(h_range[0][0] + random.random() * (h_range[1][1] - h_range[0][0]))
                add_position.append(add_list)
            elif i < N / 2 and i >= N / 4:
                add_list = []
                add_list.append(w_range[0][1] + random.random() * (w_range[1][0] - w_range[0][1]))
                add_list.append(h_range[0][0] + random.random() * (h_range[0][1] - h_range[0][0]))
                add_position.append(add_list)
            elif i >= N / 2 and i < 3 * N / 4:
                add_list = []
                add_list.append(w_range[1][0] + random.random() * (w_range[1][1] - w_range[1][0]))
                add_list.append(h_range[0][0] + random.random() * (h_range[1][1] - h_range[0][0]))
                add_position.append(add_list)
            else:
                add_list = []
                add_list.append(w_range[1][0] + random.random() * (w_range[1][1] - w_range[1][0]))
                add_list.append(h_range[1][0] + random.random() * (h_range[1][1] - h_range[1][0]))
                add_position.append(add_list)
        add_position=np.array(add_position)


        return add_position
    def result_save(self,image_name):
        """
        设置结果保存路径
        :param image_name:图像名称
        :return: 图像保存路径
        """
        folder_path='./Result/'
        new_floder=os.path.join(folder_path,image_name)
        if not os.path.exists(new_floder):#如果文件夹不存在
            os.makedirs(new_floder)
        save_address=folder_path+image_name+"/"
        return save_address
    def get_wm_new_size(self):
        """
        根据比例因子计算出水印的新大小
        :return:水印经缩放后的新的大小以及缩放后的水印图片
        """
        wm_sl=min(self.input_size[0]/(self.sl*self.wm_size[0]),self.input_size[1]/(self.sl*self.wm_size[1]))
        #
        wm_new_size=(int(self.wm_size[0]*wm_sl),int(self.wm_size[1]*wm_sl))
        wm_new_image=self.wm_image.resize(wm_new_size,resample=Image.LANCZOS)
        return wm_new_size,wm_new_image
    def wm_add_to_image(self,m):
        """
        将缩放后的水印图像添加入原图像
        :return:添加水印后的图像
        """
        input_RGBA_image = self.input_image.convert("RGBA")
        wm_RGBA_image = self.wm_new_image.convert("RGBA")

        #吧
        wm_mask_RGBA=wm_RGBA_image.convert("L").point(lambda x:min(x,float(self.alpha)))
        wm_RGBA_image.putalpha(wm_mask_RGBA)
        #
        wm_add_pos_x=int(self.wm_add_pos[m][0])
        wm_add_pos_y=int(self.wm_add_pos[m][1])
        input_RGBA_image.paste(wm_RGBA_image,(wm_add_pos_x,wm_add_pos_y),wm_mask_RGBA)
        print(input_RGBA_image.mode)
        input_RGBA_image.save(self.save_address + "+水印（1）.png")
        return input_RGBA_image


    def wm_DST(self,input_RGBA_image,m):
        """
        水印处添加扰动
        :param input_RGBA_image: RGBA格式的原图像
        :return: Tensor格式的对抗图像1（水印处添加扰动）
        """
        input_RGB_image=input_RGBA_image.convert("RGB")

        input_RGB_image_Tensor=self.transform(input_RGB_image)
        print("input_RGBA_image_Tensor的形状为：",input_RGB_image_Tensor.shape)
        input_RGB_image_Tensor=input_RGB_image_Tensor.unsqueeze(0)
        # print("input_RGBA_image_Tensor为：",input_RGBA_image_Tensor)
        #水印处添加扰动
        label=torch.full((1,),self.class_index,dtype=torch.long)
        attack=torchattacks.PGD(self.model,eps=self.eps1)
        #x1,x2是水印在原图像的坐标，分别是左上角和右下角坐标
        x1,y1=self.wm_add_pos[m][0],self.wm_add_pos[m][1]
        x2,y2=self.wm_add_pos[m][0]+self.wm_new_size[0],self.wm_add_pos[m][1]+self.wm_new_size[1]
        #生成对抗样本
        attack_sample=attack(input_RGB_image_Tensor,label)
        attack_sample=self.clear_other_DST(attack_sample,input_RGB_image_Tensor,y1,x1,y2,x2)
        return attack_sample,input_RGB_image_Tensor

    def clear_other_DST(self,attack_sample,input_RGBA_image_Tensor,x1,y1,x2,y2):
        """
        将水印之外的扰动清除
        :param attack_sample:
        :param input_RGBA_image_Tensor:
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :return:
        """
        for i in range(attack_sample.size(0)):
            for j in range(attack_sample.size(1)):
                for m in range(attack_sample.size(2)):
                    for n in range(attack_sample.size(3)):
                        if x1<m<x2 and y1<n<y2:
                            continue
                        else:
                            attack_sample[i][j][m][n]=input_RGBA_image_Tensor[i][j][m][n]
        return attack_sample


    def other_DST(self,image,m):
        # 在水印地方添加扰动
        label = torch.full((1,), self.class_index, dtype=torch.long)
        attack = torchattacks.PGD(self.model, eps=self.eps2)
        # x1,x2是水印在原图像的坐标，分别是左上角和右下角坐标
        x1, y1 = self.wm_add_pos[m][0], self.wm_add_pos[m][1]
        x2, y2 = self.wm_add_pos[m][0] + self.wm_new_size[0], self.wm_add_pos[m][1] + self.wm_new_size[1]
        # 生成对抗样本
        # attack_sample目前是Tensor格式的
        attack_sample = attack(image, label)
        attack_sample = self.clear_wm_DST(attack_sample,image, y1, x1, y2, x2)
        return attack_sample

    def clear_wm_DST(self, attack_sample, image, x1, y1, x2, y2):
        """
        将水印处扰动清除
        :param attack_sample:
        :param input_RGBA_image_Tensor:
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :return:
        """
        for i in range(attack_sample.size(0)):
            for j in range(attack_sample.size(1)):
                for m in range(attack_sample.size(2)):
                    for n in range(attack_sample.size(3)):
                        if x1 < m < x2 and y1 < n < y2:
                            attack_sample[i][j][m][n] = image[i][j][m][n]
        return attack_sample


#扰动总函数
    def wm_add_DST(self,m):
        """
        :return: 返回对抗图像
        """
        #首先将水印添加到原图像中
        input_wm_image=self.wm_add_to_image(m)
        #对添加水印后的图片添加扰动
        #对水印出添加扰动1
        DST1,input_RGBA_image_Tensor = self.wm_DST(input_wm_image,m)
        DST11 = DST1.squeeze(0)
        to_pil = transforms.ToPILImage()
        DST11 = to_pil(DST11)
        print("水印+扰动")
        DST11.save(self.save_address+"水印+扰动（2）.png")
        #对其他地方添加扰动2  DST2是Tensor格式的
        DST2 = self.other_DST(DST1,m)
        DST2=DST2.squeeze(0)
        to_pil=transforms.ToPILImage()
        DST_image=to_pil(DST2)
        print("其他+扰动")
        DST_image.save(self.save_address + "其他+扰动（3）.png")
        return DST_image


    def attack_classifier(self,i):
        #对分类器进行攻击
        input_image=self.input_image
        if get_pred(self.input_image,self.model,self.transform)==self.class_index:
            for m in range(12):
                att_image=self.wm_add_DST(m)
                #判断是否攻击成功
                #如果预测类别与原始类别相同，攻击失败
                if get_pred(att_image,self.model,self.transform)==self.class_index:
                    if m == 11:
                        print(i,"攻击失败")
                        return 0,att_image
                    continue
                else:
                    print(i,"攻击成功")
                    return 1,att_image
        else:
            return -1,input_image

