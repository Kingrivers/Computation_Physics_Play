#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import random


# In[2]:


class TempMemoryNet:
    def __init__(self, size,T):
        plt.close('all')

        self.size = size
        self.T = T
        self.start = np.full((size, size), -1)
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.out = widgets.Output()
        self.button = widgets.Button(description="Run Recognition")
        self.button.on_click(lambda b: self.recall(4,self.T))
        #self.button.on_click(self.show_array)

        A_mem = np.loadtxt("memory_pattern_A.csv", delimiter=",", dtype=int)
        B_mem = np.loadtxt("memory_pattern_B.csv", delimiter=",", dtype=int)
        C_mem = np.loadtxt("memory_pattern_C.csv", delimiter=",", dtype=int)
        A = A_mem.flatten()
        B = B_mem.flatten()
        C = C_mem.flatten()

        # Compute interaction matrix with sum of outer products sum(si(m)*sj(m))
        W = np.outer(A, A) + np.outer(B, B) + np.outer(C, C)
        np.fill_diagonal(W, 0)  # no self-connections

        # Save for use in energy calculation
        self.interaction = W
        
        # Drawing state
        self.is_drawing = False
        self.last_cell = None  # To avoid toggling the same cell multiple times per drag

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.update_plot()

    def on_press(self, event):
        if event.inaxes == self.ax:
            self.is_drawing = True
            self.toggle_or_draw(event, toggle=True)

    def on_motion(self, event):
        if self.is_drawing and event.inaxes == self.ax:
            self.toggle_or_draw(event, toggle=False)

    def on_release(self, event):
        self.is_drawing = False
        self.last_cell = None

    def toggle_or_draw(self, event, toggle):
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if 0 <= x < self.size and 0 <= y < self.size:
            i, j = self.size - 1 - y, x
            cell = (i, j)
            if toggle:
                # Click toggles between 1 and -1
                self.start[i, j] = -1 if self.start[i, j] == 1 else 1
                self.update_plot()
            else:
                # Drag sets to 1 (only if it's not already and not repeated)
                if cell != self.last_cell and self.start[i, j] != 1:
                    self.start[i, j] = 1
                    self.update_plot()
                    self.last_cell = cell

    def update_plot(self):
        self.ax.clear()
        y, x = np.where(self.start == 1)
        self.ax.scatter(x, self.size - 1 - y, color='black', marker='o')

        self.ax.set_xlim(-0.5, self.size - 0.5)
        self.ax.set_ylim(-0.5, self.size - 0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal')
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.grid(True)
        self.fig.canvas.draw_idle()

    def show_array(self, _=None):
        with self.out:
            clear_output(wait=True)
            display(self.button)
            print("Memory Array (1s = dots, -1s = empty):\n")
            print(self.start)
            
    def EnergyValue(self, matrix):

        state = matrix.flatten()
        return -(1/3) * np.dot(state, np.dot(self.interaction, state))

        
    def recall(self, iterations, T):
        state = self.start.flatten()
        for _ in range(iterations):
            indices = list(range(len(state)))
            random.shuffle(indices)
            for val in indices:
                delta_E =  (2/3)*state[val] * np.dot(self.interaction[val], state)
                if delta_E <= 0:
                    state[val] *= -1
                else:
                    r = random.random()
                    if r <= np.exp(-1*abs(delta_E)/T):
                        state[val] *= -1
        
        self.start = state.reshape((self.size, self.size))
        self.update_plot()
                    
    def interface(self):
        layout = widgets.VBox([
            self.button,  # Top button
            self.out  # And any output below that
        ])
        display(layout)


# In[3]:


def LoadMemory():
    A_mem = np.loadtxt("memory_pattern_A.csv", delimiter=",", dtype=int)
    B_mem = np.loadtxt("memory_pattern_B.csv", delimiter=",", dtype=int)
    C_mem = np.loadtxt("memory_pattern_C.csv", delimiter=",", dtype=int)
    A = A_mem.flatten()
    B = B_mem.flatten()
    C = C_mem.flatten()

    # Compute interaction matrix with sum of outer products sum(si(m)*sj(m))
    W = np.outer(A, A) + np.outer(B, B) + np.outer(C, C)
    np.fill_diagonal(W,0)
    return W


# In[4]:


class DAQMemoryNet:
    def __init__(self, W, T, Matrix): #matrix should be a 10X10 array of 1s and -1s representing the pattern to be recognised
        plt.close('all')
        self.size = 10
        self.T = T
        self.start = Matrix.copy()
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.out = widgets.Output()
        self.interaction = W
        self.EnergyArray = []
        
    def update_plot(self):
        self.ax.clear()
        y, x = np.where(self.start == 1)
        self.ax.scatter(x, self.size - 1 - y, color='black', marker='o')

        self.ax.set_xlim(-0.5, self.size - 0.5)
        self.ax.set_ylim(-0.5, self.size - 0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal')
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.grid(True)
        self.fig.canvas.draw_idle()

    def show_array(self, _=None):
        with self.out:
            clear_output(wait=True)
            display(self.button)
            print("Memory Array (1s = dots, -1s = empty):\n")
            print(self.start)
            
    def EnergyValue(self, flatmatrix):
        return -(1/3) * np.dot(flatmatrix, np.dot(self.interaction, flatmatrix))

    def recall(self, iterations):
        state = self.start.flatten()
        for _ in range(iterations):
            indices = list(range(len(state)))
            random.shuffle(indices)
            for val in indices:
                delta_E = (2/3)*state[val] * np.dot(self.interaction[val], state)
                if delta_E <= 0:
                    state[val] *= -1
                else:
                    r = random.random()
                    if r <= np.exp(-1*abs(delta_E)/self.T):
                        state[val] *= -1
                                           
                self.EnergyArray.append(self.EnergyValue(state))
                
                
        self.start = state.reshape((self.size, self.size))


# In[5]:


if __name__ == "__main__":
    get_ipython().run_line_magic('matplotlib', 'widget')
    test = TempMemoryNet(10,5)
    test.interface()


# In[8]:


if __name__ == "__main__":

    # Load and plot
    W = LoadMemory()
    plt.figure(figsize=(6, 6))
    plt.imshow(W, cmap='bwr', interpolation='nearest')
    plt.colorbar(label='Interaction Strength')
    plt.title("Hebbian Interaction Map")
    plt.xlabel("Neuron j")
    plt.ylabel("Neuron i")
    plt.show()


# In[ ]:




