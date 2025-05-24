import numpy as np
import matplotlib.pyplot as plt
import random
import HotMemory

W = HotMemory.LoadMemory()
A_mem = np.loadtxt("memory_pattern_A.csv", delimiter=",", dtype=int)
B_mem = np.loadtxt("memory_pattern_B.csv", delimiter=",", dtype=int)
C_mem = np.loadtxt("memory_pattern_C.csv", delimiter=",", dtype=int)


Temperatures = []
CorrectNumber = []

x = np.arange(10)
y = np.arange(10)
indices = []
for i in x:
    for j in y:
        thistuple = (i,j)
        indices.append(thistuple)
random.shuffle(indices)


for T in np.arange(0.001, 12.001, 0.5):
    A_Perturbed = np.copy(A_mem)
    Temperatures.append(T)
    correct = 0
    for k in range(70):
        i = indices[k][0]
        j = indices[k][1]
        A_Perturbed[i][j] *= -1 

        ARec = HotMemory.DAQMemoryNet(W, T, A_Perturbed)
        ARec.recall(3)
        diff_score = np.sum(ARec.start == A_mem)
        if diff_score == 100:
            correct += 1
    CorrectNumber.append(correct)
    print(CorrectNumber[-1])
plt.plot(Temperatures, CorrectNumber, marker='o')
plt.xlabel("Temperature")
plt.ylabel("Number of Correct Recalls")
plt.title("Recall Accuracy vs Temperature")
plt.grid(True)
plt.show()