# Logical Regression From Scratch

This project shows how a simple **Logical regression model** can be built from scratch using Python.  
It uses **NumPy** for the calculations and **Matplotlib** to show the results visually.

Instead of using machine learning libraries, the aim of this script is to help understand what is happening behind the scenes when a model learns from data.

---

## What the Script Does

The program works in a few simple steps:

1. **Enter training data**  
   You enter X values and Y values directly in the terminal.  
   The Y values should only be **0 or 1**, since this is a binary classification problem.

2. **Model training**  
   The script trains the model for 1000 iterations so it can learn the relationship between the X and Y values.

3. **Prediction**  
   After training, you can enter a new X value and the model will estimate the probability that it belongs to class **1**.

4. **Visualisation**  
   Two graphs are shown:
   - The logistic curve fitted to the training data
   - How the training loss changes over time

---

## How the Model Works

Logical regression uses a **sigmoid function** to turn a value into a probability between 0 and 1.

```
sigmoid(z) = 1 / (1 + e^(-z))
```

Where:

```
z = wx + b
```

- **w** is the weight (slope)  
- **b** is the bias (intercept)

During training, the model slowly adjusts these values so that its predictions match the real data as closely as possible.

To measure how well the model is doing, it uses something called **Binary Cross Entropy Loss**, which simply measures how far the predictions are from the correct answers.

---

## Requirements

You will need the following Python libraries:

- numpy  
- matplotlib  

Install them with:

```bash
pip install numpy matplotlib
```

---

## How to Run the Script

Run the file with Python:

```bash
git clone https://github.com/Nishchay-Bhudia/Logical-Regression.git
```

Example input:

```
X values: 1,2,3,4
Y values: 0,0,1,1
```

After the model finishes training, you can enter a new X value to see the prediction.

---

## Output

The program will show:

- The learned model parameters (w and b)
- The probability of class 1 for a new value
- The predicted class (0 or 1)
- A graph of the logistic regression curve
- A graph showing how the loss changes during training

---

## Purpose

This project was made for **learning and experimentation**.  
It is a simple way to see how Logical regression works without relying on large machine learning libraries.
