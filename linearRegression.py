import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes


# Define function to calculate 'm' and 'b' values for equation 'y = mx + b'
def line_best_fit(x, y):
    gradient = (np.mean(x) * np.mean(y) - np.mean(x*y)) / (np.mean(x)**2 - np.mean(x**2))
    coef = np.mean(y) - gradient * np.mean(x)

    return gradient, coef


# Load Diabetes dataset and create x and y axis for graph
database = load_diabetes()
database_data = database.data[:, np.newaxis, 2]

# Set training and test data
dx_train = database_data[:-20]
dy_train = database.target[:-20]
dx_test = database_data[-20:]
dy_test = database.target[-20:]

# Call line_best_fit method to calculate equation variables
gradient, coef = line_best_fit(np.squeeze(dx_train), np.squeeze(dy_train))

# Determine x axis min and max
# Use x axis min and max to calculate y equation by substituting 'm' and 'b' 
x_axis = [np.min(dx_train), np.max(dx_train)]
y_axis = [(np.min(dx_train) * gradient + coef), (np.max(dx_train) * gradient + coef)]

# Plot training and test data
plt.scatter(dx_train, dy_train, c='r', label='Train Data')
plt.scatter(dx_test, dy_test, c='g', label='Test Data')

# Plot best fit line
plt.plot(x_axis, y_axis, c='b', label='Line Best Fit')

# Create legend and display scatter plot
legend = plt.legend()
plt.show()
