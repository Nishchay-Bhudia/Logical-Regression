import numpy as np
import matplotlib.pyplot as plt #needed for graphical representation

#sigmoid func

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#users input of training data

print("Enter X values (example: 1,2,3,4)")
x_input = input("X values: ")

print("Enter Y values (ONLY 0 or 1 example: 0,0,1,1)")
y_input = input("Y values: ")

x = np.array([float(i) for i in x_input.split(",")])
y = np.array([float(i) for i in y_input.split(",")])

#set initial parameters

w = 0.0
b = 0.0
learning_rate = 0.1
epochs = 1000
n = len(x)

loss_history = []

#training loop

for epoch in range(epochs):

    #Linear combination
    z = w * x + b

    # Apply sigmoid
    y_pred = sigmoid(z)

    #Binary Cross Entropy Loss
    loss = -np.mean(y*np.log(y_pred + 1e-9) + (1-y)*np.log(1-y_pred + 1e-9))
    loss_history.append(loss)

    #gradients
    dw = (1/n) * np.sum((y_pred - y) * x)
    db = (1/n) * np.sum(y_pred - y)

    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

print("\nTraining complete!")
print("Learned w:", w)
print("Learned b:", b)

#prediction of new data point

new_x = float(input("\nEnter new X value: "))
probability = sigmoid(w * new_x + b)

print("Probability of Class 1:", probability)

if probability >= 0.5:
    print("Predicted Class: 1")
else:
    print("Predicted Class: 0")

#matplotlib visualization

x_line = np.linspace(min(x)-1, max(x)+1, 100)
y_line = sigmoid(w * x_line + b)

plt.scatter(x, y, label="Training Data")
plt.plot(x_line, y_line, label="Logistic Curve")
plt.legend()
plt.title("Logistic Regression")
plt.show()

#matploitlib loss plot

plt.plot(loss_history)
plt.title("Loss Over Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
