#!/usr/bin/env python
# -*- coding: utf-8 -*-


from datetime import datetime
import time


#1 获取当前时间精确到毫秒
print(datetime.now())
print(datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f'))


#2 字符串转datatime
str_date="2018-6-6 12:23:5"
first_data=datetime.strptime(str_date,'%Y-%m-%d %H:%M:%S')
print(first_data)

#3 datatime转字符串
time_str=datetime.strftime(first_data,'%Y-%m-%d %H-%M-%S')
print(time_str)

#4 将日期转为秒级时间戳
dt = '2018-01-01 10:40:30'
ts = int(time.mktime(time.strptime(dt, "%Y-%m-%d %H:%M:%S")))
print(ts)


#5 将秒级时间戳转为日期
ts = 1515774430
dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
print(dt)
