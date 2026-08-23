# Logistic Regression From Scratch

One Python script that learns to separate two classes from numbers you type in, with
the sigmoid, the loss and the gradient descent loop all written out in NumPy. The repo
name says "Logical", which is a typo I made early on. The algorithm is logistic regression.

## What it does

You enter x values and matching y labels that are each 0 or 1. The script trains for
1000 steps, prints the learned weight and bias, then asks for a new x and tells you the
probability that it belongs to class 1, plus the class it falls into at a 0.5 threshold.
It finishes with two plots: your data with the fitted S-curve through it, and the loss
over training.

## How it works

A straight line can output any number, but a probability has to sit between 0 and 1, so
the line's output is squashed through the sigmoid function first. Error is measured with
binary cross entropy, which punishes confident wrong answers far harder than uncertain
ones.

Then gradient descent: the weight and bias start at zero, and on every step the script
works out which direction each of them should move to make the error smaller, and moves
them a small distance that way. Repeat a thousand times and the S-curve slides and
steepens until it sits where the 0s stop and the 1s start. The learning rate of 0.1 sets
how big each of those steps is.

## Running it

```bash
pip install numpy matplotlib
python "logical regression.py"
```

## Current state

Works for what it is, with the limits you would expect from a teaching script. One
feature only. No input validation, so mismatched x and y lengths or labels that are not
0 or 1 will misbehave. No train/test split and no accuracy figure, since with a handful
of hand-typed points there is nothing meaningful to hold out. The `1e-9` inside the log
is there to stop the loss becoming infinite when the model gets fully confident.

## Tech

Python, NumPy, matplotlib. MIT licensed.
