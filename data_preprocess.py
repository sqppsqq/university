# -*- coding:UTF-8 -*- #
"""
@filename:data_preprocess.py
@author:Kiki Shen
@time:2024-04-11
"""
import os
import pandas as pd

def get_label_csv():
    """
    读取数据集images和csv
    Returns:图片名字列表，图片标签列表，图片路径列表
    """
    file_dir = "./all_data/images"
    files = os.listdir(file_dir)
    imagename_list=files
    imagepath_list=[]
    csv_path="./all_data/images.csv"
    data=pd.read_csv(csv_path,usecols=['ImageId','TrueLabel'])
    imagelabel_list=[]
    for i in range(100):
        image_name=imagename_list[i]
        for j in range(len(data)):
            if image_name==str(data.iloc[j,0])+".png":
                imagelabel_list.append(data.iloc[j,1]-1)
                imagepath_list.append("./all_data/images/"+image_name)
    return imagename_list,imagelabel_list,imagepath_list