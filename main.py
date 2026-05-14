# -*- coding:UTF-8 -*- #
"""
@filename:main.py
@author:Kiki Shen
@time:2024-04-11
@description:主函数
"""
import argparse
import time

from PIL import Image
from data_preprocess import *
from model import *
from sample import *
def parse_arg():
    parser=argparse.ArgumentParser(description="使用PGD方法进行攻击，水印处扰动高，水印外扰动低")
    parser.add_argument('--dataset',type=str,default='imagenet-mini',help='要攻击的数据集')
    #GoogleNet      inception     resnet       vgg        squeezenet
    parser.add_argument('--model', type=str, default='GoogleNet', help='要攻击的模型')
    #攻击相关参数
    parser.add_argument('--num', type=float, default=100, help='参与对抗的图片的数量')
    parser.add_argument(''
                        ''
                        '--sl', type=int, default=4, help='比例因子')
    parser.add_argument('--alpha', type=float, default=200, help='水印的透明度')#1时不透明
    parser.add_argument('--eps1', type=float, default=8/255, help='水印处扰动强度')
    parser.add_argument('--eps2', type=float, default=1/255, help='水印外扰动强度')
    # 解析命令行参数
    args = parser.parse_args()
    return args
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def compare_images(image1, image2):
    # 创建一个包含两个子图的画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))

    # 在第一个子图上显示第一张图片
    ax1.imshow(image1)
    ax1.set_title('Image 1')

    # 在第二个子图上显示第二张图片
    ax2.imshow(image2)
    ax2.set_title('Image 2')

    # 隐藏坐标轴
    ax1.axis('off')
    ax2.axis('off')

    # 展示图片
    plt.show()
if __name__ == "__main__":
    time_start = time.time()
    #imagename_list,imagelabel_list,imagepath_list
    #图片名字列表，图片标签列表，图片路径列表
    imagename_list, imagelabel_list, imagepath_list=[],[],[]
    #获取数据
    imagename_list, imagelabel_list, imagepath_list=get_label_csv()
    #获取攻击模型
    model1,trans1=init_model('GoogleNet')
    model2,trans2=init_model('inception')
    googlenet_succ=0
    googlenet_fail=0
    inception_succ=0
    inception_fail=0
    opt = parse_arg()

    #首先模型1对图片进行分类
    for i in range(100):
        input_image = Image.open(imagepath_list[i]).convert('RGB')
        wm_image=Image.open("./all_data/logo/MIT.png")

        att=SQQ(
            input_image=input_image,
            class_index=imagelabel_list[i],
            image_name=imagename_list[i],
            wm_image=wm_image,
            sl=opt.sl,
            alpha=opt.alpha,
            eps1=opt.eps1,
            eps2=opt.eps2,
            model=model1,
            transform=trans1

        )
        flag,att_image=att.attack_classifier(i)
        if flag==1:
            googlenet_succ+=1
            att_image.save("./result/GoogleNet/succ/"+imagename_list[i])
        elif flag==0:
            googlenet_fail+=1
            att_image.save("./result/GoogleNet/fail/" + imagename_list[i])
        else:print("初始分类失败")

        pred_class=get_pred(att_image,model2,trans2)
        print(i,"真实类别:",imagelabel_list[i],"预测类别：",pred_class)
        if pred_class==imagelabel_list[i]:
            inception_fail+=1
            att_image.save("./result/inception/fail/" + imagename_list[i])
            print(i, "inception攻击失败，当前攻击成功率为：", inception_succ / float(inception_succ + inception_fail))
        else:
            inception_succ+=1
            att_image.save("./result/inception/succ/" + imagename_list[i])
            print(i, "inception攻击成功，当前攻击成功率为：", inception_succ / float(inception_succ + inception_fail))
    print("googlenet_succ:",googlenet_succ)
    print("googlenet_fail:",googlenet_fail)

    print("GoogleNet攻击成功率为：",googlenet_succ/float(googlenet_fail+googlenet_succ))
    # print("inception攻击成功率为：",inception_succ/float(inception_succ+inception_fail))
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    # print("对inception进行攻击：结果分别为：")
    print("运行时间为：", time_sum)
    #对抗







