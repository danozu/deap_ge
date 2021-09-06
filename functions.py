#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:31:49 2021

@author: allan
"""

import numpy as np
import math

# Define new functions

def pdiv(a, b):
    if abs(b) > 1e-3:
        return np.divide(a,b)
    else:
        return 1
    
def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
    
#def protected_div(left, right):
#    try:
#        return left / right
#    except ZeroDivisionError:
#        return 1

def psin(n):
    #try:
    return np.sin(n)
    #except Exception:
    #    return np.nan

def pcos(n):
    #try:
    return np.cos(n)
    #except Exception:
    #    return np.nan

def add(a, b):
    return np.add(a,b)

def sub(a, b):
    return np.subtract(a,b)

def mul(a, b):
    return np.multiply(a,b)

def psqrt(a):
    return np.sqrt(abs(a))

def max_(a,b):
    return np.maximum(a, b)

def min_(a,b):
    return np.minimum(a, b)

def plog(a):
    if a > 1e-3:
        return math.log(a)
    else:
        return 0

def not_(a):
    return np.logical_not(a)

def and_(a, b):
    return np.logical_and(a,b)

def or_(a, b):
    return np.logical_or(a,b)

def nand_(a, b):
    return np.logical_not(np.logical_and(a,b))

def nor_(a, b):
    return np.logical_not(np.logical_or(a,b))