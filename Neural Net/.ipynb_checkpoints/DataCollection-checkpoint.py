import numpy as np
import matplotlib.pyplot as plt
import random
import HotMemory

W = HotMemory.LoadMemory()
ChangeOrder = np.loadtxt("ChangeOrder.csv", delimiter=",", dtype=int)
A_mem = np.loadtxt("memory_pattern_A.csv", delimiter=",", dtype=int)
B_mem = np.loadtxt("memory_pattern_B.csv", delimiter=",", dtype=int)
C_mem = np.loadtxt("memory_pattern_C.csv", delimiter=",", dtype=int)

