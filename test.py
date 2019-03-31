#-*- coding:utf-8 -*-
string = u"姜 振  康"
list = [char for char in string.split(" ")]
print u"姜" in list
print list
print "".join(string.strip().split(" "))