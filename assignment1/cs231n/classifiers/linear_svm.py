from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    # 图片的总类别数
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        # 普通的矩阵乘法，1张图片每个类别的分数，行向量
        correct_class_score = scores[y[i]]
        # 正确的类别对应的分数
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            # loss为所有不正确分类的得分减去正确分类得分并加上阈值
            # 1为阈值，要求正确分类的分数要比其他不正确分类的高至少1，否则作为loss
            if margin > 0:
                loss += margin
                dW[:, y[i]] += -X[i].T  # 正确分类项的梯度，累加每个不正确项导致的-X[i]
                dW[:, j] += X[i].T  # 不正确分类项的梯度，梯度为X[i]；j在变化，每次循环都更新一个不同的j
                # X[i]不转置也可，会自动转置后计算
                # 参考https://zhuanlan.zhihu.com/p/21478575 (梯度公式是对Loss公式求导而来)
                # todo 不正确分类项(公式2)求导公式推导存疑

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # 损失再算上正则化项：reg*sum(theta^2)，此处theta即为W矩阵
    dW += 2 * reg * W
    # 好像就是简单地对loss公式求导，因此算上正则项为 reg*(2 * theta)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    num_train = X.shape[0]

    scores_y = scores[range(num_train), y].reshape(num_train, 1)
    # print("scores_y", scores_y.shape)  # scores_y (500, 1)
    margins = np.maximum(0, scores-scores_y+1)
    # print("margins", margins.shape)  # margins (500, 10)
    margins[range(num_train), y] = 0
    # 因为正确分类不需要参与loss计算，此处置0方便后面sum
    loss += np.sum(margins)/num_train
    loss += reg * np.sum(W ** 2)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    margins[margins > 0] = 1
    # 重复利用上面的margins，而它在上面已经将所有<=0的都设为了0
    margins[range(num_train), y] = -np.sum(margins, axis=1)
    # 按行计算不正确分类的个数总和，并赋给正确分类所在位置
    dW += np.dot(X.T, margins)
    # X.T：(D, N)， margins：(N, C), C=10
    dW += 2 * reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
