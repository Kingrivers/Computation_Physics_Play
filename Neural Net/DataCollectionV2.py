import numpy as np
import matplotlib.pyplot as plt
import random
from multiprocessing import Pool
import HotMemory

W = HotMemory.LoadMemory()
A_mem = np.loadtxt("memory_pattern_A.csv", delimiter=",", dtype=int)
B_mem = np.loadtxt("memory_pattern_B.csv", delimiter=",", dtype=int)
C_mem = np.loadtxt("memory_pattern_C.csv", delimiter=",", dtype=int)

# Parameters
num_trials = 60       # Number of times to repeat for averaging
num_perturbations = 60  # Number of perturbed bits per trial
temperature_range = np.arange(0.001, 12.001, 0.1)

def AveragePlotting(choice):
    if choice == 1:
        average_correct_A = []  # To store averaged results
        std_correct_A = []
        for T in temperature_range:
            trial_correct = []
            for trial in range(num_trials):
                # Generate new perturbation order each trial
                indices = [(i, j) for i in range(10) for j in range(10)]
                random.shuffle(indices)

                A_Perturbed = np.copy(A_mem)
                correct = 0
                for k in range(num_perturbations):
                    i, j = indices[k]
                    A_Perturbed[i][j] *= -1
                    ARec = HotMemory.DAQMemoryNet(W, T, A_Perturbed)
                    ARec.recall(3)
                    diff_score = np.sum(ARec.start == A_mem)
                    if diff_score == 100:
                        correct += 1
                trial_correct.append(correct)  # store each result

            avg = np.mean(trial_correct)
            std = np.std(trial_correct) / np.sqrt(num_trials)
            average_correct_A.append(avg)
            std_correct_A.append(std)    
            print(f"T = {T:.3f}, Average correct = {average_correct_A[-1]}")

        # Plotting
        plt.errorbar(
            temperature_range,
            average_correct_A,
            yerr=std_correct_A,        
            fmt='o-',                 
            capsize=3                  
        )
        plt.xlabel("Temperature")
        plt.ylabel("Average Correct Recoveries")
        plt.title(f"Average Recall Success vs Temperature\n({num_trials} trials, {num_perturbations} perturbations each)")
        plt.grid(True)
        plt.savefig(f"A-AvgRecall_{num_trials}trials_{num_perturbations}flips.png")
        plt.close()

    if choice == 2:
        average_correct_B = []  # To store averaged results
        std_correct_B = []
        for T in temperature_range:
            trial_correct = []
            for trial in range(num_trials):
                # Generate new perturbation order each trial
                indices = [(i, j) for i in range(10) for j in range(10)]
                random.shuffle(indices)

                B_Perturbed = np.copy(B_mem)
                correct = 0
                for k in range(num_perturbations):
                    i, j = indices[k]
                    B_Perturbed[i][j] *= -1
                    BRec = HotMemory.DAQMemoryNet(W, T, B_Perturbed)
                    BRec.recall(3)
                    diff_score = np.sum(BRec.start == B_mem)
                    if diff_score == 100:
                        correct += 1
                trial_correct.append(correct)  # store each result

            avg = np.mean(trial_correct)
            std = np.std(trial_correct) / np.sqrt(num_trials)
            average_correct_B.append(avg) 
            std_correct_B.append(std)    
            print(f"T = {T:.3f}, Average correct = {average_correct_B[-1]}")

        # Plotting
        plt.errorbar(
            temperature_range,
            average_correct_B,
            yerr=std_correct_B,        
            fmt='o-',                 
            capsize=3                  
        )
        plt.xlabel("Temperature")
        plt.ylabel("Average Correct Recoveries")
        plt.title(f"Average Recall Success vs Temperature\n({num_trials} trials, {num_perturbations} perturbations each)")
        plt.grid(True)
        plt.savefig(f"B-AvgRecall_{num_trials}trials_{num_perturbations}flips.png")
        plt.close()

    if choice == 3:
        average_correct_C = []  # To store averaged results
        std_correct_C = []
        for T in temperature_range:
            trial_correct = []
            for trial in range(num_trials):
                # Generate new perturbation order each trial
                indices = [(i, j) for i in range(10) for j in range(10)]
                random.shuffle(indices)

                C_Perturbed = np.copy(C_mem)
                correct = 0
                for k in range(num_perturbations):
                    i, j = indices[k]
                    C_Perturbed[i][j] *= -1
                    CRec = HotMemory.DAQMemoryNet(W, T, C_Perturbed)
                    CRec.recall(3)
                    diff_score = np.sum(CRec.start == C_mem)
                    if diff_score == 100:
                        correct += 1
                trial_correct.append(correct)  # store each result

            avg = np.mean(trial_correct)
            std = np.std(trial_correct) / np.sqrt(num_trials)
            average_correct_C.append(avg)
            std_correct_C.append(std)  
            print(f"T = {T:.3f}, Average correct = {average_correct_C[-1]}")

        # Plotting
        plt.errorbar(
            temperature_range,
            average_correct_C,
            yerr=std_correct_C,        
            fmt='o-',                 
            capsize=3                  
        )
        plt.xlabel("Temperature")
        plt.ylabel("Average Correct Recoveries")
        plt.title(f"Average Recall Success vs Temperature\n({num_trials} trials, {num_perturbations} perturbations each)")
        plt.grid(True)
        plt.savefig(f"C-AvgRecall_{num_trials}trials_{num_perturbations}flips.png") 
        plt.close()   

choice = [i for i in range(1,4)]
if __name__ == "__main__":
    Pool().map(AveragePlotting, choice)
