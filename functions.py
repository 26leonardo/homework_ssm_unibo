import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
import random

random.seed(26)
np.random.seed(26)

def backtracking(x, f, grad_f, params_f = None):
    """
    inputs:
    x: ndarray. The actual iterate x_k.
    f: function. The function that we want to optimize.
    grad_f: function. The gradient of f(x).

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
    x0: ndarray. Initial guess.
    maxit: int. Maximum number of iterations.
    tolf: float. Tolerance for the of the algorithm. Convergence is reached when the norm(grad_f(x_k),2) < tolf*norm(grad_f(x0),2).
    tolx: float. Tolerance for the x value. Convergence is reached when norm(x_k - x_k-1,2) < tolx.
    back_flag: bool. If True, the algorithm uses backtracking to update alpha.
    alpha: float. Value of alpha. Only used if back_flag is False.

    outputs:
    x: ndarray. Array that contains the value of x for each iterate.
    k: int. Number of iterations needed to converge.
    f_val: ndarray. Array of the value of each f(x).
    grads: ndarray. Array of gradient values.
    err: ndarray. Array of error values (normalize gradient).
    converge: bool. True if the method converges.
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


def exercise_f(f, grad_f, params_f = None , x0 = None, true_sol = None, dimensions = 2, alphas=[0.001,0.01,0.1,0.2],maxit = 100, tolf=1e-5, tolx=1e-5):
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
    plt.ylabel("$\\| \\nabla f(x^k) \\|_2$")
    plt.legend(loc="upper right")
    
    if true_sol is not None:
        plt.subplot(1,2,2)
        plt.title("Distance from true solution")
        for (i,alpha) in enumerate(alphas): 
            if converge[i]:
                plt.plot(norm(x[i] - true_sol.T,2,axis=1),label=r"$\alpha$={}, $k$={}".format(alpha,k[i]))
        plt.plot(norm(x_back - true_sol.T,2,axis=1),label=r"$\alpha$=back, $k$={}".format(k_back))
        plt.xlabel(r"$k$")
        plt.ylabel(r"$\left\| x_i - \mathbf{true\_sol}^T \right\|_2$")
        plt.legend(loc="upper right")
    
    plt.show()

    
def plot_f(x0, f, grad_f, params_f = None, alpha = None, maxit = 100):
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
        xx = np.linspace(-2,2,maxit)
        plt.plot(xx,f(xx, params_f))

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


    # split the data into training and testing sets
def split_data(X, Y, per_train=0.8):

    if type(X) is not np.ndarray or type(Y) is not np.ndarray:
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
    X, y = params_f  # Unpack the data
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
            params_f = (X_batch, y_batch)
            gradient = grad_loss(theta, params_f)
            grad_loss_vec.append(np.linalg.norm(gradient, 2))
            # Update weights
            theta = theta - alpha * gradient

        # Save the updated values
        theta_history[epoch] = theta
        params_f = (X, y)
        loss_history[epoch] = loss(theta, params_f)
        grad_norm_history[epoch] = np.mean(grad_loss_vec)
    
    return theta_history, loss_history, grad_norm_history

def polynomial_features(X, degree):
    """
    Generate polynomial features for input data X up to a given degree.
    
    Parameters:
    X (numpy.ndarray): Input data of shape (n_samples, n_features).
    degree (int): Degree of the polynomial features.
    
    Returns:
    numpy.ndarray: Polynomial features of shape (n_samples, n_polynomial_features).
    """
    return np.vstack([X**i for i in range(degree)])

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
    N = len(Y_test)
    predictions = predict_poly(X_test, theta)
    error = np.mean((predictions - Y_test)**2)
    return error


def plot_loss(loss_history, grad_norm_history):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(loss_history)
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("MSE poly")

    ax[1].plot(grad_norm_history, color="red")
    ax[1].set_title("Gradient Norm")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Gradient Norm")

    plt.show()

def plot_poly_regression(X_train, Y_train, X_test, Y_test, theta):
    plt.figure(figsize=(10, 5))

    x_range = np.linspace(X_train.min(), X_train.max(), 100)
    y_range = predict_poly(x_range, theta)

    plt.scatter(X_train, Y_train, label="Train data")
    plt.scatter(X_test, Y_test, label="Test data")
    plt.plot(x_range, y_range, color="red", label="Model")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Model Prediction")
    plt.legend()

    plt.show()