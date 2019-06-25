#!/usr/bin/python
# -*- coding:UTF-8 -*-
"""
    @data     : 2019-5-30
    @author   : 冯丽洋
    @describe : FM, predict 实现
"""

import sys 
import json
import io
import os
import math
import time
from optparse import OptionParser


def para_define(parser):

    parser.add_option("-m", "--model", action="store", \
                        type="string", \
                        dest="model", \
                        default="./data/model.txt")

    parser.add_option("-d", "--data", action="store", \
                        type="string", \
                        dest="data", \
                        default="./data/iii")

    parser.add_option("-0", "--out", action="store", \
                        type="string", \
                        dest="out", \
                        default="./data/out.txt")

#加载模型
def load_model(model_dir):
    dict_model = {}
    with open(model_dir) as model_file:
        for line in model_file:
            items = line.strip('\n').split(' ')
            # 线性权值
            if len(items) == 2:
                key = items[0]
                val = items[1]
                key = key.strip(':')
                dict_model[key] = []
                dict_model[key].append(float(val))

            # latent feature
            if len(items) == 6:
                key = items[0]
                vals = items[1:]
                key = key.strip(':')
                dict_model[key] = []
                for val in vals:
                    dict_model[key].append(float(val))
    return dict_model

# 待预测数据归一化
def instance_norm(list_data):
    list_norm = []
    for dict_data in list_data:
        # 求平方和
        norm = 0
        for feature_id in dict_data:
            val = dict_data[feature_id]
            norm += val*val
        list_norm.append(norm)
    return list_norm

# 加载预测数据
def load_data(data_dir):
    # 待预测的dict嵌入的list
    list_data = []
    with open(data_dir) as data_file:
        for line in data_file:
            items = line.strip('\n').split(' ')
            dict_data = {}
            for pairs in items:
                feature_id, val = pairs.split(':')
                dict_data[feature_id] = float(val)
            list_data.append(dict_data)

    return list_data

# 预测-线性部分
def P_linear(linear_val, dict_data, dict_model, sqrt_norm_data):
    L_featureID = []
    for feature_id in dict_data:
        L_featureID.append(feature_id)
        key = 'i' + '_' + feature_id
        data_value = dict_data[feature_id]
        modelWeight = dict_model[key][0]
        dataVal = data_value/sqrt_norm_data
        linear_val += (dataVal*modelWeight)

    return linear_val, L_featureID


# 预测-非线性部分
def P_nolinear(dict_data, dict_model, norm_data, L_featureID):
    # FM 有化简措施, 因此其复杂度为 O(kn)
    nolinearVal = 0.0
    for i in range(5):
        pow_sum = 0.0
        sum_pow = 0.0
        for j in range(len(L_featureID)):
            
            feature_j = L_featureID[j]
            data_value_j = dict_data[feature_j]
            key_j = 'v' + '_' + feature_j

            # 归一化
            dataVal = data_value_j / norm_data
            dataVal_pow = dataVal * dataVal
            
            # vec_j 为list
            v_j = dict_model[key_j][i]
            v_j_pow = v_j * v_j

            # 
            sum_pow += (dataVal * v_j)
            pow_sum += (dataVal_pow * v_j_pow)

        nolinearVal += ((sum_pow * sum_pow) - pow_sum)

    return nolinearVal/2

# ctr预测
def predict(dict_model, list_data, list_norm):
    # 遍历所有待预测样本
    bias = dict_model['bias'][0]
    list_predict = []
    for i in range(len(list_data)):
        '''
            针对一条预测数据,进行初始化
        '''
        # 待预测数据
        dict_data = list_data[i]
        # 用于线性part
        sqrt_norm_data = math.sqrt(list_norm[i])
        # 用于非线性part
        norm_data = list_norm[i]
        nolinear_val = 0.0

        '''
            开始预测
        '''
        # 线性part
        linear_val, L_featureID = P_linear(bias, dict_data, dict_model, sqrt_norm_data)
        # 非线性part
        nolinear_val = P_nolinear(dict_data, dict_model, norm_data, L_featureID)
        # 相加得最终结果
        predict_val = linear_val + nolinear_val
        list_predict.append(predict_val)

    return list_predict 

def logit(list_predict):
    list_ctr = []
    for pre_val in list_predict:
        ctr = 1 / (1+math.exp(-pre_val))
        list_ctr.append(ctr)
    
    return list_ctr

if __name__ == '__main__':

    parser = OptionParser()
    para_define(parser)
    (options, args) = parser.parse_args()

    dict_model = load_model(options.model)
    list_data = load_data(options.data)

    # 开始计时
    start_time = time.perf_counter()
    # 归一化
    list_norm = instance_norm(list_data)
    # 预测
    list_predict = predict(dict_model, list_data, list_norm)
    #计时结束
    end_time = time.perf_counter()
    
    L_ctr = logit(list_predict)    
    
    for ctr in L_ctr:
        print(ctr)

    run_time = end_time - start_time
    print("Runtime:",run_time,sep="\t")




