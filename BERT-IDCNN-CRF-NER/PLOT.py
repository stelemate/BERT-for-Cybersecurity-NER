"""
author : Chekecheke
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv

'''读取csv文件'''


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append((row[2]))
        x.append((row[1]))
    return x, y


mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

plt.figure()
x2, y2 = readcsv("scalars3.csv")
plt.plot(x2, y2, color='red', label='Default')
# plt.plot(x2, y2, '.', color='red')

x, y = readcsv("scalars.csv")
plt.plot(x, y, 'g', label='Without BN')

x1, y1 = readcsv("scalars2.csv")
plt.plot(x1, y1, color='black', label='Without DW and PW')

x4, y4 = readcsv("scalars4.csv")
plt.plot(x4, y4, color='blue', label='Without Residual learning')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.ylim(0, 16)
plt.xlim(0, 104800)
plt.xlabel('Steps', fontsize=20)
plt.ylabel('Score', fontsize=20)
plt.legend(fontsize=16)
plt.show()
