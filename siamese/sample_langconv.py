# encoding: utf-8
from langconv import *

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    print sentence
    return sentence

if __name__=="__main__":
    traditional_sentence = u'憂郁的臺灣乌龟ddd'
    simplified_sentence = Traditional2Simplified(traditional_sentence)
    print(simplified_sentence)

    '''
    输出结果：
        忧郁的台湾乌龟
    '''
