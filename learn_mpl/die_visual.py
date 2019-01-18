#!/usr/bin/env python
# -*- coding: utf-8 -*-


from die import Die
import pygal

#创建一个D6
die=Die()

#投掷几次骰子,并将结果存储在一个列表中
results=[]
for roll_num in range(1000):
    result=die.roll()
    results.append(result)

print("The len of results:%d\n"%len(results))

#分析结果
frequencies=[]
for value in range(1,die.num_sides+1):
    frequency=results.count(value)
    frequencies.append(frequency)

print(frequencies)
print(frequency)

#对结果进行可视化
hist=pygal.Bar()

hist.tittle="Results"
hist.x_labels=['1','2','3','4','5','6']
hist.x_title="Value"
hist.y_title="Frequency of Value"

hist.add('D6',frequencies)
hist.render_to_file('die_visual.svg')