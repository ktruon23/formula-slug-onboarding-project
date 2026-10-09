import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y, color = 'blue', linestyle = '--')
plt.title("My Matplotlib Plot")
plt.xlabel("X Axis Label")
plt.ylabel("Y Axis Label")
plt.savefig("plot.png", dpi=150)
