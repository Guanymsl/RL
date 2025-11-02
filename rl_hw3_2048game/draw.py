import matplotlib.pyplot as plt

bin = [0, 0, 0, 0, 0, 0, 1, 1, 17, 41, 38, 2]
labels = [f"{2**i}" for i in range(len(bin))]

plt.bar(labels, bin, color="orange")
plt.xlabel("Best Tile")
plt.ylabel("Frequency")
plt.title("Best Tile Distribution")
plt.show()
