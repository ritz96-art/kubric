import requests
import pandas
import scipy
import numpy
import sys
import matplotlib.pyplot as plt


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def computeCost(X, y, theta):
    inner = np.power(((X @ theta.T) - y), 2) 
    return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iters):
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum((X @ theta.T - y) * X, axis=0)
        cost = computeCost(X, y, theta)
    return (theta, cost)


def predict_price(my_data) -> float:
    
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    X = my_data[:, 0].reshape(-1,1) # -1 tells numpy to figure out the dimension by itself
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate([ones, X],1)
    alpha = 0.0001
    iters = 1000
    theta = np.array([[1.0, 1.0]])
    y = my_data[:, 1].reshape(-1,1)

    g, cost = gradientDescent(X, y, theta, alpha, iters) 
    plt.scatter(my_data[:, 0].reshape(-1,1), y)
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim()) 
    y_vals = g[0][0] + g[0][1]* x_vals
    return y_vals
    ...


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
