# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 19:27:37 2021

@author: allan
"""

import csv 

with open("datasets/randomData.csv") as symbRegData:
    n_rows = sum(1 for line in symbRegData)
with open("datasets/randomData.csv") as symbRegData:
    reader = csv.reader(symbRegData)
    data = list(list(float(elem) for elem in row) for row in reader)

#diff = sum((func(*row[:-1]) - row[-1])**2 for row in data)

for row in data:
    print(*row[:-1])