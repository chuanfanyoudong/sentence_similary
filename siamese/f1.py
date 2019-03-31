#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com/chuanfanyoudong
@software: PyCharm
@file: f1.py
@time: 2019/3/28 9:39
"""
import numpy as np

result = []
y_test = []
file1 = open("data/test", "r")
for line in file1:
    try:
        y_test.append(line.strip().split("\t")[3])
    except:
        print line

file1 = open("data/re", "r")
for line in file1:
    result.append(line.strip().split("\t")[1])

TP = 0
TP_FP = 0
TP_FN = 0
for i in range(len(result)):
    # print result[i]
    # print y_test[i]
    if result[i] == "1":
        # print 1
        TP_FP += 1
    if y_test[i] == "1":
        # print 2
        TP_FN += 1
    if result[i] == "1" and y_test[i] == "1":
        # print 3
        TP += 1
TP = float(TP)
P = TP / TP_FP
R = TP / TP_FN
F1 = 2 * P * R / (P + R)
print P
print R
print F1
#

def shffle():
    line_list = []
    file1 = open("data/ant_all.text", "r")
    for line in file1:
        line_list.append(line)
    np_line = np.array(line_list)
    np.random.shuffle(np_line)
    line_list = list(np_line)
    test_data = line_list[:10000]
    train_data = line_list[10000:]
    file2 = open("data/test", "w")
    for line in test_data:
        file2.write(line)
    file3 = open("data/train", "w")
    tag_0 = 0
    tag_1 = 0
    num = 0
    final_list = []
    for line in train_data:
        tag = line.strip().split("\t")[3]
        if tag == "0" and num < 60000:
            num += 1
            tag_0 += 1
            continue
        if tag == "1":
            tag_1 += 1
        file3.write(line)
    print tag_0, tag_1
shffle()

def change_():
    file1 = open("data/train.txt", "r")
    file2 = open("data/train_final.txt", "w")
    i = 0
    for line in file1:
        file2.write(str(i) + "\t" + line)
        i += 1
# change_()
