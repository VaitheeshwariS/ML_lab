import math
import matplotlib.pyplot as plt
data = [
    [45, 130, 210, "No"],
    [50, 140, 250, "Yes"],
    [35, 120, 180, "No"],
    [60, 150, 260, "Yes"],
    [40, 125, 190, "No"],
    [55, 145, 250, "Yes"],
    [48, 135, 220, "No"],
    [65, 160, 280, "Yes"]
]
new = [52, 142, 245]
k = 3

dist = []

for x in data:
    d = math.sqrt(
        (x[0] - new[0]) ** 2 +
        (x[1] - new[1]) ** 2 +
        (x[2] - new[2]) ** 2
    )
    dist.append((d, x))

dist.sort()

nearest = dist[:k]

print("K Nearest Neighbours:")

for d, x in nearest:
    print(f"Distance={d:.2f}->{x[3]}")

yes = sum(x[1][3] == "Yes" for x in nearest)
no = sum(x[1][3] == "No" for x in nearest)

print("\nYes=", yes)
print("No=", no)

prediction = "Yes" if yes > no else "No"

print("Prediction=", prediction)

# Simple 2D graph
yes_points = []
no_points = []

for x in data:
    if x[3] == "Yes":
        yes_points.append(x)
    else:
        no_points.append(x)

plt.scatter(
    [x[0] for x in yes_points],
    [x[1] for x in yes_points],
    color="green",
    marker="o",
    label="Yes"
)

plt.scatter(
    [x[0] for x in no_points],
    [x[1] for x in no_points],
    color="red",
    marker="x",
    label="No"
)

plt.scatter(
    new[0],
    new[1],
    color="blue",
    marker="*",
    s=250,
    label="New Patient"
)

plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("K-Nearest Neighbours (KNN)")
plt.legend()
plt.show()
