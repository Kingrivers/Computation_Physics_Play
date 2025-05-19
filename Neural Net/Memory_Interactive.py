#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[36]:


class MemoryNet:
    def __init__(self, size):
        self.size = size
        self.start = np.random.choice([-1,1],size = (size,size))
    def plot(self):
        # Get coordinates where the value is 1
        y, x = np.where(self.start == 1)  # note: y = row, x = column

        plt.figure(figsize=(5, 5))
        plt.scatter(x, self.size - 1 - y, marker='o', color='black')

        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        # Remove ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.show()


# In[37]:


net = MemoryNet(10)
net.plot()


# In[ ]:




