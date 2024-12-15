import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
import random

random.seed(26)
np.random.seed(26)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------#
# Functions for the gradient descent method

def backtracking(x, f, grad_f, params_f = None):
    """
    Backtracking method to find the best alpha that satisfies the backtracking condition.

    inputs:
    x: ndarray. The actual iterate x_k.
    f: function. The function that we want to optimize.
    grad_f: function. The gradient of f(x).
    params_f: tuple. Tuple with the parameters of f.

    outputs:
    alpha: float. The value of alpha that satisfies the backtracking condition.
    """
    alpha = 1
    c = 0.8
    tau = 0.25
    # Calculating the gradient of f(x)
    grad_f_x = grad_f(x, params_f)
    # Calculating f(x)
    f_x = f(x, params_f)

    n = 0
    
    if grad_f_x.shape != (1,1):
        n = norm(grad_f_x,2)
    else:
        n = np.abs(grad_f_x)

    # Cheking the Sufficent Decrease Condition (Armijo Condition)
    # f(x - alpha*grad_f_x) <= f(x) - c*alpha*(grad_f_x^T)*grad_f_x
    # second Wolf condition (Curvature condition) is automatically satisfied
    while f((x - (alpha * grad_f_x).T).flatten(), params_f) > f_x - c * alpha * (n ** 2):
        alpha = tau * alpha
        
        if alpha < 1e-3:
            break
    return alpha

def GD(f, grad_f, params_f = None, x0=0, maxit=1000, back_flag=False, alpha=0.1, tolf=1e-5, tolx=1e-5):
    """
    Gradient Descent method

    inputs:
    f: function. Function to be minimized(optimize).
    grad_f: function. Gradient of f.
    params_f: tuple. Tuple with the parameters of f.
    x0: ndarray. Initial guess of theta to start and optimize.
    maxit: int. Maximum number of iterations.
    tolf: float. Tolerance for the of the algorithm. Convergence is reached when the norm(grad_f(x_k),2) < tolf*norm(grad_f(x0),2).
    tolx: float. Tolerance for the x value. Convergence is reached when norm(x_k - x_k-1,2) < tolx.
    back_flag: bool. If True, the algorithm uses backtracking to update alpha.
    alpha: float. Value of alpha. Only used if back_flag is False.

    outputs:
    x: ndarray. (Hystory) Array that contains the value of x(theta) for each iterate.
    k: int. Number of iterations needed to converge.
    f_val: ndarray. (Hystory) Array of the value of each f(x).
    grads: ndarray. (Hystory) Array of gradient values.
    err: ndarray. (Hystory) Array of error values (normalize gradient).
    converge: bool. True if the method converges.
    message: str. Message that explains why the method stopped.
    """
    # Trashold to check if the values are diverging
    div_chek = 1e10

    # Calculate the initial alpha value
    if back_flag:
        alpha = backtracking(x0, f, grad_f, params_f)

    # Setting the initial values
    x, f_val, grads, err = [], [], [], []
    x.append(x0)
    f_val.append(f(x0, params_f))
    grads.append(grad_f(x0, params_f))
    err.append(norm(grads[0],2))

    for k in range(1,maxit):
        # Update the x value iterativly and saves the last value
        x_k = x[k-1] - (alpha*grad_f(x[k-1], params_f).flatten())
        x.append(x_k)
        if (x[k] > div_chek).any():
            message = "Diverging 1"
            # print(message)
            return x,k,f_val,grads,err,False,message
        
        # Update alpha with backtracking
        if back_flag:
            alpha = backtracking(x[k], f, grad_f, params_f)
        
        # Adding the values to be returned
        f_val.append(f(x[k], params_f))
        grads.append(grad_f(x[k], params_f))
        err.append(norm(grads[k],2))
        
        if (grads[k] > div_chek).any():
            message = "Diverging 2"
            # print(message)
            return x,k,f_val,grads,err,False,message
        

        # Check the stop condition
        if norm(grad_f(x[k], params_f),2) < tolf * norm(grad_f(x0, params_f),2):
            message = "Stop for f tolerance"
            # print(message)
            return x,k,f_val,grads,err,True,message
        if norm(x[k] - x[k-1],2) < tolx:
            message = "Stop for x tolerance"
            # print(message)
            return x,k,f_val,grads,err,True,message
        
    message = "Reached maxit"
    # print(message)
    return x,maxit,f_val,grads,err,True,message


def exercise_GD(f, grad_f, params_f = None , x0 = None, true_sol = None, dimensions = 2, alphas=[0.001,0.01,0.1,0.2],maxit = 100, tolf=1e-5, tolx=1e-5):
    """
    Exercise for the Gradient Descent method.

    inputs:
    f: function. Function to be minimized.
    grad_f: function. Gradient of f.
    params_f: tuple. Tuple with the parameters of f.
    x0: ndarray. Initial guess of theta to start and optimize.
    true_sol: ndarray. True solution of the problem.
    dimensions: int. Number of dimensions of the problem, len of theta to optimize each iterration.

    """

    # Fixed parameters
    if x0 is None:
        x0 = np.zeros((dimensions, )).T
    kmax = maxit

    # Compute the solution with fixed alpha values
    x, k, fval, grads, err, converge, messages = [], [], [], [], [], [], []
    for alpha in alphas:
        x_i, k_i, fval_i, grads_i, err_i, converge_i, message_i = GD(f , grad_f, params_f, x0, maxit=kmax, back_flag=False, alpha=alpha, tolf=tolf, tolx=tolx)
        x.append(x_i)
        k.append(k_i)
        fval.append(fval_i)
        grads.append(grads_i)
        err.append(err_i)
        converge.append(converge_i)
        messages.append(message_i)

    # Compute the solution with backtracking
    x_back,k_back,fval_back,grads_back,err_back, converge_back, message_back = GD(f,grad_f,params_f,x0,maxit=kmax, back_flag=True, alpha=alpha, tolf=tolf, tolx=tolx)

    # Plots
    fig = plt.figure(figsize=(12,6))

    # Plot the norm of the gradient
    plt.subplot(1,2,1)
    plt.title("Norm of the gradient")
    for (i,alpha) in enumerate(alphas): 
        if converge[i]:
            line, = plt.plot(err[i],label=r"$\alpha$={}, $k$={}".format(alpha,k[i]))
            index_1_5 = len(err[i]) // 5
            x_position = index_1_5
            y_position = err[i][index_1_5]
            plt.text(x_position, y_position+(i/5), messages[i], fontsize=12, color=line.get_color())
    line, = plt.plot(err_back,label=r"$\alpha$=back, $k$={}".format(k_back))
    index_1_5 = len(err_back) // 5
    x_position = index_1_5
    y_position = err_back[index_1_5]
    plt.text(x_position, y_position, message_back, fontsize=12, color=line.get_color())
    plt.xlabel(r"$k$")
    plt.grid()
    plt.ylabel("$\\| \\nabla f(x^k) \\|_2$")
    plt.legend(loc="upper right")
    
    if true_sol is not None:
        plt.subplot(1,2,2)
        plt.title("Distance from true solution")
        for (i,alpha) in enumerate(alphas): 
            if converge[i]:
                plt.plot(norm(x[i] - true_sol.T,2,axis=1),label=r"$\alpha$={}, $k$={}".format(alpha,k[i]))
        plt.plot(norm(x_back - true_sol.T,2,axis=1),label=r"$\alpha$=back, $k$={}".format(k_back))
        plt.grid()
        plt.xlabel(r"$k$")
        plt.ylabel(r"$\left\| x_i - \mathbf{true\_sol}^T \right\|_2$")
        plt.legend(loc="upper right")
    
    plt.show()

    
def plot_step_length(x0, f, grad_f, params_f = None, alpha = None, maxit = 100):
    """
    Plot the function f and the iterates of the gradient descent method starting from x0.

    inputs:
    x0: ndarray. Initial guess. <-----------------
    f: function. Function to be minimized.
    grad_f: function. Gradient of f.
    params_f: tuple. Tuple with the parameters of f.
    alpha: float. Learning rate.
    maxit: int. Maximum number of iterations.
    
    """
    plt.figure(figsize=(20,5))
    colors = ['b','g','r','c','m','y']
    n_x=len(x0)
    for i in range(n_x):
        plt.subplot(1,n_x,i+1)
        xx = np.linspace(-3,3,maxit)
        plt.plot(xx,f(xx, params_f))
        plt.grid()

        # Calling the GD starting from x0
        if alpha is None:
            x,k,fval,grads,err, converge, message = GD(f,grad_f,params_f,np.array([x0[i]]),maxit=maxit,back_flag=True)
        else:
            x,k,fval,grads,err, converge, message = GD(f,grad_f,params_f,np.array([x0[i]]),maxit=maxit,alpha=alpha)
        
        x = np.array(x).flatten()
        if converge:
            plt.title(f"$x_0 = {x0[i]}$")
        else:
            plt.title(f"$x_0 = {x0[i]}$ - {message}")

        # in the case of divergence f may not have the same size of x, or vice versa
        if len(fval) != len(x):
            right_len = min(len(fval),len(x))
        else:
            right_len = len(fval)
        plt.scatter(x[:right_len],fval[:right_len],c=colors[i % len(colors)])
        for j in range(right_len-1):
            plt.plot([x[j],x[j+1]],[fval[j],fval[j+1]],c=colors[i % len(colors)])
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------#
# General functions

# split the data into training and testing sets
def split_data(X, Y, per_train=0.8):

    if type(X) is not np.ndarray and type(Y) is not np.ndarray:
        X = np.array(X)
        Y = np.array(Y)
     
    if len(X.shape) != 1:
        X = X.flatten()
        print("Flattening X", X.shape)

    N_train = int(X.shape[0] * per_train)

    N = X.shape[0]

    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:N_train]
    test_idx = idx[N_train:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    
    X_test = X[test_idx]
    Y_test = Y[test_idx]

    print(f"Train test split = {Y_train.size}, {Y_test.size}")

    return (X_train, Y_train), (X_test, Y_test)

def polynomial_features(X, degree):
    """
    Generate polynomial features for input data X up to a given degree.
    Vandermode matrix of X with degree k is the matrix of shape (n_samples, k+1) with the columns being [1, X, X^2, ..., X^k].
    
    Parameters:
    X (numpy.ndarray): Input data of shape (n_samples, n_features).
    degree (int): Degree of the polynomial features.
    
    Returns:
    numpy.ndarray: Polynomial features of shape (n_samples, n_polynomial_features).
    """
    return np.vstack([X**i for i in range(degree)])

def predict_poly(X_test, theta):
    """
    Predict the output of the model.
    
    Parameters:
    X_test (numpy.ndarray): Test data.
    theta (numpy.ndarray): Parameters of the model.
    
    Returns:
    numpy.ndarray: Predicted output.
    """
    degree = theta.shape[0]
    Phi_X = polynomial_features(X_test, degree)
    return Phi_X.T @ theta


def mse_error(X_test, Y_test, theta):
    """
    Compute the Mean Squared Error (MSE) for the given test data and parameters.
    
    Parameters:
    X_test (numpy.ndarray): Test input data of shape (n_samples,).
    Y_test (numpy.ndarray): True output data of shape (n_samples,).
    theta (numpy.ndarray): Polynomial regression parameters.
    
    Returns:
    float: Mean Squared Error (MSE).
    """
    N =len(Y_test)
    predictions = predict_poly(X_test, theta)
    error = np.mean((predictions - Y_test)**2)
    # residuals = predictions - Y_test
    # mse = (1 / N) * np.linalg.norm(residuals, 2) ** 2
    # mse2 = (1/N) * np.sum((predictions - Y_test)**2)
    # print("error = ",error, "and", mse2 , "and", mse)
    return error

#-------------------------------------------------------------------------------------------#
# Functions for minimize the polynomial regression

def loss_mse_poly(theta, params):
    """
    Compute the Mean Square Error (MSE) loss.
    
    Parameters:
    theta (numpy.ndarray): Parameters of the model.
    X (numpy.ndarray): Input data.  
    Y (numpy.ndarray): True labels.
    
    Returns:
    float: MSE loss.
    """
    X, Y = params[0], params[1]
    degree = theta.shape[0]
    Phi_X = polynomial_features(X, degree)
    N = Y.shape[0]
    predictions = Phi_X.T @ theta
    loss = (1 / N) * np.linalg.norm(predictions - Y, 2)**2
    return loss

def grad_loss_mse_poly(theta, params):
    """
    Compute the gradient of the Mean Square Error (MSE) loss.
    
    Parameters:
    theta (numpy.ndarray): Parameters of the model.
    X (numpy.ndarray): Input data.
    Y (numpy.ndarray): True labels.
    
    Returns:
    numpy.ndarray: Gradient of the MSE loss with respect to theta.
    """
    X, Y = params[0], params[1]
    degree = theta.shape[0]
    Phi_X = polynomial_features(X, degree)
    N = Y.shape[0]
    predictions = Phi_X.T @ theta
    gradient = (2 / N) * Phi_X @ (predictions - Y)
    return gradient


#-------------------------------------------------------------------------------------------#
# Functions for the polynomial regression with stochastic gradient descent

def SGD_poly(loss, grad_loss, params_f, degree_poly = 2, theta0 = None, alpha = 0.1, batch_size = None, n_epochs = 100):
    """
    Stochastic Gradient Descent method

    inputs:
    loss: function. Loss function to be minimized.
    grad_loss: function. Gradient of the loss function.
    D: tuple. Tuple with the data (X, y).
    theta0: ndarray. Initial guess of the weights.
    alpha: float. Learning rate.
    batch_size: int. Size of the batch.
    n_epochs: int. Number of epochs.

    outputs:
    theta_history: ndarray. Array that contains the value of theta for each epoch.
    loss_history: ndarray. Array of the value of each loss.
    grad_norm_history: ndarray. Array of gradient norms.

    """
    # Unpack the data
    if len(params_f) == 3:
        X, y, lam = params_f  
    else:
        X, y = params_f  
    N = X.shape[0] # We assume both X and Y has shape (N, )
    idx = np.arange(0, N) # This is required for the shuffling

    # Initialization initial weights
    if theta0 is None:
        theta0 = np.zeros((degree_poly, ))
    d = theta0.shape[0] # While theta0 has shape (d, )

    if (d != degree_poly):
        raise ValueError(f"theta0 has shape {theta0.shape} but degree_poly is {degree_poly}")

    # If batch_size is not given, we use 1/3 of the data
    if batch_size is None:
        batch_size = (params_f[0].shape[0])/3

    
    # Initialization of history vectors
    theta_history = np.zeros((n_epochs, degree_poly))  # Save parameters at each epoch
    loss_history = np.zeros((n_epochs, ))  # Save loss values at each epoch
    grad_norm_history = np.zeros((n_epochs, ))  # Save gradient norms at each epoch
    
    # Initialize weights
    theta = theta0
    for epoch in range(n_epochs):
        # Shuffle the data at the beginning of each epoch
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Initialize a vector that saves the gradient of the loss at each iteration
        grad_loss_vec = []

        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            X_batch = X[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]
            
            # GD step
            # we are computing GD(f_loss, grad_loss, theta, maxit = 1, alpha = fixed)
            # only problem is passing the X_batch and y_batch, so it seampler to just pass to use this way
            # Compute the gradient of the loss
            if len(params_f) == 3:
                params_f = (X_batch, y_batch, lam)
            else:
                params_f = (X_batch, y_batch)

            gradient = grad_loss(theta, params_f)
            grad_loss_vec.append(np.linalg.norm(gradient, 2))
            # Update weights
            theta = theta - alpha * gradient

        # Save the updated values
        theta_history[epoch] = theta
        if len(params_f) == 3:
            params_f = (X, y, lam)
        else:
            params_f = (X, y)
        loss_history[epoch] = loss(theta, params_f)
        grad_norm_history[epoch] = np.mean(grad_loss_vec)
    
    return theta_history, loss_history, grad_norm_history



def plot_loss(loss_history, grad_norm_history):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(loss_history)
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("MSE poly")
    ax[0].grid(True)

    ax[1].plot(grad_norm_history, color="red")
    ax[1].set_title("Gradient Norm")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Gradient Norm")
    ax[1].grid(True)
    plt.show()

def plot_poly_regression(X_train, Y_train, X_test, Y_test, thetas_history_different_k, title = "Model Prediction", VS = False, lamds = None):
    
    if type(thetas_history_different_k) is not list and type(thetas_history_different_k) is not tuple:
        thetas_history_different_k = [thetas_history_different_k]
    
    plt.figure(figsize=(10, 5))
    colors = ['k','r','c','m','y','g','b']
    plt.scatter(X_train, Y_train, label="Train data")
    plt.scatter(X_test, Y_test, label="Test data")

    if VS == False:
        index_c = 0
        for i,thetas in enumerate(thetas_history_different_k):
            if i%3 == 0:
                x_range = np.linspace(X_train.min(), X_train.max(), 100)
                y_range = predict_poly(x_range, thetas)
                degree_poly = thetas.shape[0]
                plt.plot(x_range, y_range, color=colors[index_c%6], label=f"Model {i}, k = {degree_poly}")
                index_c +=1
        plt.grid()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Model " + title)
        plt.legend()
    else:
        for i,thetas in enumerate(thetas_history_different_k):
            x_range = np.linspace(X_train.min(), X_train.max(), 100)
            y_range = predict_poly(x_range, thetas)
            degree_poly = thetas.shape[0]
            if lamds is not None:
                plt.plot(x_range, y_range, color=colors[i], label=f"Model {i}, k = {degree_poly}, lam = {lamds[i]}")
            else:
                plt.plot(x_range, y_range, color=colors[i], label=f"Model {i}, k = {degree_poly}")
        plt.grid()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Model " + title)
        plt.legend()

    plt.show()


def plot_train_test_error_respect_to_lambda(X_train, Y_train, X_test, Y_test, theta_histories_lambda, lamds, title):
    """
    Plot the train and test error for the given thetas history.
    
    Parameters:
    X_train (numpy.ndarray): Train input data.
    Y_train (numpy.ndarray): Train true labels.
    X_test (numpy.ndarray): Test input data.
    Y_test (numpy.ndarray): Test true labels.
    theta_histories_lambda (list): List of list of thetas history changing for lambda.
    lamds (list): List of lambda values.
    title (str): Title for the plot.
    """
    if type(theta_histories_lambda) is not list and type(theta_histories_lambda) is not tuple:
        theta_histories_lambda = [theta_histories_lambda]
    
    train_errors = []
    test_errors = []

    for thetas in theta_histories_lambda:
        train_error = mse_error(X_train, Y_train, thetas) 
        test_error = mse_error(X_test, Y_test, thetas)
        train_errors.append(train_error)
        test_errors.append(test_error)

    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.plot(lamds, train_errors, label="Train Error", marker="o")
    plt.plot(lamds, test_errors, label="Test Error", marker="o")
    plt.xlabel("$\lambda$")
    plt.ylabel("$MSE$")
    plt.title(title)
    plt.legend()
    plt.suptitle("Train and Test Error changing $\lambda$", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_train_test_error_respect_to_k(X_train, Y_train, X_test, Y_test, thetas_histories_different_k, titles):
    """
    Plot the train and test error for the given thetas history.
    
    Parameters:
    X_train (numpy.ndarray): Train input data.
    Y_train (numpy.ndarray): Train true labels.
    X_test (numpy.ndarray): Test input data.
    Y_test (numpy.ndarray): Test true labels.
    thetas_histories_different_k (list): List of list of thetas history changing for K.
    titles (list): List of titles for the plots.
    """
    if type(thetas_histories_different_k[0]) is not list and type(thetas_histories_different_k[0]) is not tuple:
        thetas_histories_different_k = [thetas_histories_different_k]
    
    num_plots = len(titles)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 5 * num_plots))
    outputs = {}

    for ax, thetas_history_different_k, title in zip(axes, thetas_histories_different_k, titles):
        train_errors = []
        test_errors = []
        degrees_k = []
        out = []
        outputs[title] = out
        for thetas in thetas_history_different_k:
            train_error = mse_error(X_train, Y_train, thetas)
            test_error = mse_error(X_test, Y_test, thetas)
            train_errors.append(train_error)
            test_errors.append(test_error)
            out.append((train_error, test_error, thetas.shape[0]))
            degrees_k.append(thetas.shape[0])

        ax.grid()
        ax.plot(degrees_k, train_errors, label="Train Error")
        ax.plot(degrees_k, test_errors, label="Test Error")
        ax.set_xlabel("$K$")
        ax.set_ylabel("$MSE$")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Train and Test Error changing K", fontsize=16)
    plt.tight_layout()
    plt.show()

    # for key in outputs.keys():
    #     print(key)
    #     for i in range(len(outputs[key])):
    #         print(f"K={outputs[key][i][2]}: Train Error={outputs[key][i][0]}, Test Error={outputs[key][i][1]}")


#-----------------------------------------------------------------------------------------------------------------------------------------------------------#
# MLE and MAP functions for GD, SGD and normal equations

def cholesky(A,b):
    """
    Solve the system Ax = b using Cholesky decomposition.
    we start by solving Ly = b 
    then L^T x = y.
    """
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L,b)
    x = np.linalg.solve(L.T, y)
    return x

def loss_MLE(theta,params_f): 
    X, Y = params_f
    Phi_X = polynomial_features(X, len(theta)).T 
    
    return (np.linalg.norm(Phi_X @ theta - Y,2)**2)/2

def grad_loss_MLE(theta,params_f): 
    X, Y = params_f
    Phi_X = polynomial_features(X, len(theta)).T
    return Phi_X.T @ (Phi_X @ theta - Y)

def MLE(D,degree_poly):
    X,Y = D
    K = degree_poly
    Phi_X = polynomial_features(X, K)
    # np.linalg.inv(Phi_X.T @ Phi_X) @ Phi_X.T @ Y
    A = Phi_X @ Phi_X.T
    b = Phi_X @ Y
    return cholesky(A,b)

def loss_MAP(theta,params_f):
    X, Y, lam = params_f
    Phi_X = polynomial_features(X, len(theta)).T
    return (np.linalg.norm(Phi_X @ theta - Y,2)**2 + lam * np.linalg.norm(theta, 2) ** 2)/2 

def grad_loss_MAP(theta,params_f): 
    X, Y, lam = params_f
    Phi_X = polynomial_features(X, len(theta)).T
    return Phi_X.T @ (Phi_X @ theta - Y) + lam * theta

def MAP(D, degree_poly, l):
    X,Y = D
    K = degree_poly
    #np.linalg.inv(Phi_X.T @ Phi_X + l*np.identity(K)) @ Phi_X.T @ Y
    Phi_X = polynomial_features(X, K)
    A = Phi_X @ Phi_X.T + l*np.identity(K)
    b = Phi_X @ Y
    return cholesky(A,b)


#-------------------------------------------------------------------------------------------#

# Functions to compute the weights for different polynomial degrees

def compute_weights( method, approach, kk, data, lam = None, loss = None, grad_loss = None, batch_size = 50, n_epochs = 500, alpha = 0.01, kmax = 500):
    """
    Compute the weights for different polynomial degrees

    input:
    method: str, method to use to compute the weights
    - GD: Gradient Descent
    - SGD: Stochastic Gradient Descent
    - Normal: Normal equations
    approach: str, approach to use for Normal method
    - MLE: Maximum Likelihood Estimation
    - MAP: Maximum A Posteriori
    kk: list of int, polynomial degrees to try
    data: tuple of X and Y
    lam: float, regularization parameter
    loss: function, loss function to use
    grad_loss: function, gradient of the loss function
    batch_size: int, batch size for SGD
    n_epochs: int, number of epochs for SGD
    alpha: float, learning rate for SGD
    kmax: int, maximum number of iterations for GD

    output:
    thetas_history_different_k: list of arrays, weights for each polynomial degree
    converging_kk_history: list of int, polynomial degrees for which the optimization converged

    """
    if type(kk) is not tuple and type(kk) is not list:
        kk = [kk]
    
    X,Y = data
    thetas_history_different_k = []
    converging_kk_history = []
    for k in kk:# (tqdm(kk,desc='Optimize'))
        #initial guess of theta
        theta_0 = np.zeros((k,))
        degree_poly = k
            
        if method == "GD":
            # Gradient descent with backtracking
            if lam is None:
                params_f = (X,Y)
            else:
                params_f = (X,Y,lam)
            x,maxit,f_val,grads,err,converge, message = GD(loss, grad_loss,params_f, x0=theta_0, maxit=kmax, back_flag=True)
            if converge:
                thetas_history_different_k.append(x[-1])
                converging_kk_history.append(k)
            else:
                print(f"Convergence not reached for k={k}: {message}")
        elif method == "SGD":
            # Stochastic gradient descent with alpha defoult = 0.01
            if lam is None:
                params_f = (X,Y)
            else:
                params_f = (X,Y,lam)
            theta_history, loss_history, grad_norm_history = SGD_poly(loss, grad_loss, params_f= params_f,degree_poly = degree_poly, theta0=theta_0, alpha=alpha, batch_size=batch_size, n_epochs=n_epochs)
            thetas_history_different_k.append(theta_history[-1])
            converging_kk_history.append(k)
        elif method == "Normal":
            # Normal equations using lambda
            if approach == 'MLE':
                theta_MLE = MLE(data,k)
                thetas_history_different_k.append(theta_MLE)
                converging_kk_history.append(k)
            elif approach == 'MAP':
                theta_MAP = MAP(data,k, l=lam)
                thetas_history_different_k.append(theta_MAP)
                converging_kk_history.append(k)
            else:
                raise ValueError(f'Unknown approach')
        else:
            raise ValueError(f'Unknown value for method') 

    return thetas_history_different_k, converging_kk_history