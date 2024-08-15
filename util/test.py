#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/8/31 23:32
# @Author  : glan~
# @FileName: confusion_matrix.py
# @annotation:
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

font = {'family': 'serif', 'serif': 'Times New Roman', 'weight': 'normal', 'size': 10}

plt.rc('font', **font)
confusion = np.array(([590, 72, 38], [52, 1318, 30], [39, 59, 832]))
# 热度图，后面是指定的颜色块，可设置其他的不同颜色
plt.imshow(confusion)
# ticks 坐标轴的坐标点
# label 坐标轴标签说明
indices = range(len(confusion))
# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
# plt.xticks(indices, [0, 1, 2])
# plt.yticks(indices, [0, 1, 2])
plt.xticks(indices, ['脑膜瘤', '胶质瘤', '垂体瘤'])
plt.yticks(indices, ['脑膜瘤', '胶质瘤', '垂体瘤'])

plt.colorbar()

plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')

# plt.rcParams两行是用于解决标签不能显示汉字的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 显示数据
for first_index in range(len(confusion)):  # 第几行
    for second_index in range(len(confusion[first_index])):  # 第几列
        plt.text(first_index, second_index, confusion[first_index][second_index],
                 va='center', ha='center')
# 在matlab里面可以对矩阵直接imagesc(confusion)
# 保存
plt.savefig("./1.svg", dpi=600, bbox_inches='tight')
# 显示
plt.show()

