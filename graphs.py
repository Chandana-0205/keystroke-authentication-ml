import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("dataset/DSL-StrongPasswordData.csv")

user = "s002"

genuine = data[data["subject"] == user]

features = genuine.iloc[:,3:].values

# Average typing pattern
mean_pattern = np.mean(features, axis=0)

plt.figure(figsize=(10,5))
plt.plot(mean_pattern)

plt.title("Average Keystroke Timing Pattern")
plt.xlabel("Feature Index")
plt.ylabel("Time Interval")
plt.show()
