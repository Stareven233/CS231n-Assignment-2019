import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    # scores -= np.max(scores)  # 提高计算中的数值稳定性，防止溢出

    for i in range(X.shape[0]):
        exp_sum = np.sum(np.exp(scores[i]))
        loss += -scores[i, y[i]] + np.log(exp_sum)
        for j in range(W.shape[1]):
            if j == y[i]:
                dW[:, j] -= X[i]
            dW[:, j] += X[i] * np.exp(scores[i, j])/exp_sum
        # softmax梯度推导：https://blog.csdn.net/qq_27261889/article/details/82915598
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss = loss/X.shape[0] + reg * np.sum(W ** 2)
    dW = dW/X.shape[0] + 2 * reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_img = X.shape[0]
    scores = X.dot(W)
    scores -= np.max(scores, axis=1)[:, np.newaxis]
    # 同样是为了数值稳定
    # 相当于np.max(scores, axis=1, keepdims=True)，否则max完size为(500,)少了一维
    exp_sum = np.sum(np.exp(scores), axis=1)
    loss = np.log(exp_sum) - scores[range(num_img), y]
    # 要求range(num_img)与y长度一致，其中元素对应组合的元组作为scores下标
    loss = np.sum(loss)/num_img
    # 因为上面log完loss为(500, )，要再次累加
    loss += reg * np.sum(W ** 2)

    dW = np.exp(scores) / exp_sum.reshape(-1, 1)
    # (N, C) / (N, 1)
    dW[range(num_img), y] -= 1
    dW = X.T @ dW
    # 相当于X.T.dot(W)
    dW /= num_img
    dW += 2 * reg * W
    pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW
