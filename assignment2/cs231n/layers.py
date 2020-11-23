from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    new_x = x.reshape(x.shape[0], -1)
    out = new_x@w + b
    # todo 此处需保留原始的x用于cache中，原因暂时未知
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout @ w.T
    dx = dx.reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T @ dout
    db = dout.sum(axis=0)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.clip(x, 0, None)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * (x>0)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        sample_mean = np.mean(x, axis=0)
        sample_var = np.mean((x - sample_mean)**2, axis=0)
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = x_hat*gamma + beta

        cache = (x, sample_mean, sample_var, x_hat, eps, gamma, )
        # cache难顶，要到了backword才知道
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x = (x - running_mean) / np.sqrt(running_var + eps)
        out = x*gamma + beta
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, mean, var, x_hat, eps, gamma = cache
    N = x.shape[0]

    dbeta = np.sum(dout, axis=0)
    d_tmp = dout

    # out = x_hat*gamma + beta:
    dgamma = np.sum(x_hat * d_tmp, axis=0)
    # gamma与Xi逐元素对应相乘，跟beta一样要在列上累加
    dx_hat = gamma * d_tmp

    # x_hat = (x - sample_mean) / np.sqrt(sample_var + eps):
    ue = x - mean
    sita = np.sqrt(var + eps)
    d_ue = (1 / sita) * dx_hat
    d_sita = np.sum((-ue / sita**2) * dx_hat, axis=0)
    # 分别为分子分母两部分

    # (x - sample_mean): 
    dx1 = d_ue
    dmean1 = -np.sum(d_ue, axis=0)

    # np.sqrt(sample_var + eps):
    dvar = 0.5 * (var + eps)**(-0.5) * d_sita
    # 形如 (a+b)**0.5 = z 对a的反向传播

    # sample_var = np.mean((x - sample_mean)**2, axis=0):
    dx_square = (1 / N) * dvar
    dx_square = 2 * (x - mean) * dx_square

    # x_mean = x - sample_mean:
    dx2 = dx_square
    dmean2 = -np.sum(dx_square, axis=0)
    # 两个dmean没有在axis=0计算sum，结果dx相对值直接为1.0

    dx3 = (1 / N) * dmean1
    dx4 = (1 / N) * dmean2
    # X由许多独立的样本Xi组成
    # sample_mean = np.mean(x, axis=0) 实际上就是 ```mean = 1/N * Sigma_i=0_to_N(Xi)```
    # 故对其中每个Xi都可求导得 dXi = 1/N * dmean
    # 既然每个dXi结果一致，那么对于dX同样是 1/N * dmean (靠广播加到每行Xi上)，
    # 注意这不代表dX每行的值都一样，dmean反向传来的梯度只是其中一部分
    # 总之，mean这类的反向梯度计算直接用 1/N * dmean
    # 参考 https://blog.csdn.net/LittleGreyWing/article/details/106967647

    dx = dx1 + dx2 + dx3 + dx4
    # 由计算图可知X一共有4个入口(包括两个在不同地方用于计算mean)，最后的结果都要加上
    # 参考 https://blog.csdn.net/SpicyCoder/article/details/97796858
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, mean, var, x_hat, eps, gamma = cache
    N = x.shape[0]

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_hat * dout, axis=0)

    # dx = -gamma*dout * ((x - mean)**2 * (var + eps)**(-0.5) + np.sqrt(var)) / (N * var)
    # 自己推导的，通过逐步推出链式法则各个中间部分的导数再相乘得到 dy/dx

    # dx = (1. / N) * gamma * (var + eps) ** (-1. / 2.) * (N * dout - np.sum(dout, axis=0) - (x - mean) * (var + eps) ** (-1.0) * np.sum(dout * (x - mean), axis=0))
    
    # dx_hat = dout * gamma
    # dsigma = -0.5 * np.sum(dx_hat * (x - mean), axis=0) * np.power(var + eps, -1.5)
    # dmu = -np.sum(dx_hat / np.sqrt(var + eps), axis=0) - 2 * dsigma * np.sum(x - mean, axis=0) / N
    # dx = dx_hat / np.sqrt(var + eps) + 2.0 * dsigma * (x - mean) / N + dmu / N
    # 查到的很多都是这种做法，感觉完全没有差别，也不符合一行的要求

    dx = gamma * (dout - (dbeta + x_hat * dgamma)/N) * (1 / np.sqrt(var + eps))
    # 来自 https://github.com/bingcheng1998/CS231n-2020-spring-assignment-solution/blob/main/assignment2/cs231n/layers.py#L259

    # 题目要求 implementation fits on a single 80-character line，而且要略快于之前的
    # 然而上面三种仅最后一种比较符合，而且速度也都非常不稳定，实在是不会了

    # inv_sigma = 1. / np.sqrt(var + eps)
    # dxhat = dout * gamma
    # dx = (1. / N) * inv_sigma * (N * dxhat - np.sum(dxhat, axis=0) - x_hat * np.sum(dxhat * x_hat, axis=0))
    # 也是完全不行

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # laynorm 是对每一行数据单独进行标准化
    # sample_mean = x.mean(1).reshape(-1, 1)
    # sample_var = x.var(1).reshape(-1, 1)
    # 借助numpy现成的函数
    sample_mean = np.mean(x, axis=1, keepdims=True)
    sample_var = np.mean((x - sample_mean)**2, axis=1, keepdims=True)
    # 需要keepdims=True，否则报错ValueError: operands could not be broadcast together with shapes (4,3) (4,) 
    # 因为 axis=1 使结果shape为向量(n, )，但向量似乎默认行广播，与下方列广播需求不符
    x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
    out = x_hat*gamma + beta

    cache = (x, sample_mean, sample_var, x_hat, eps, gamma, )
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # ----------------------矩阵与向量--------------------------------
    # row = np.random.randint(1, 4, size=(3, ))
    # # 单行矩阵
    # d = np.random.randint(1, 6, size=(3, ))
    # # 向量
    # print('row,d: ', f'{row},{d}', '\n')
    # print(row * d, '\n')
    # # 矩阵与向量，此时*为点乘
    # print(row * d.reshape(-1, 1), '\n')
    # # 矩阵与矩阵，此时*为矩阵乘
    # ---------------------------------------------------------------

    x, mean, var, x_hat, eps, gamma = cache
    N = x.shape[0]

    dbeta = np.sum(dout, axis=0)
    d_tmp = dout

    dgamma = np.sum(x_hat * d_tmp, axis=0)
    dx_hat = gamma * d_tmp

    # ------------------------失败的尝试-------------------------------
    # ue = x - mean
    # sita = np.sqrt(var + eps)
    # d_ue = (1 / sita) * dx_hat
    # d_sita = np.sum((-ue / sita**2) * dx_hat, axis=1, keepdims=False)
    # # 与batchnorm不同之处：axis=1, keepdims

    # dx1 = d_ue
    # dmean1 = -np.sum(d_ue, axis=1, keepdims=True)
    # # 与batchnorm不同之处：axis=1, keepdims

    # dvar = 0.5 * (var.squeeze() + eps)**(-0.5) * d_sita

    # dx_square = (1 / N) * dvar
    # dx_square = 2 * ((x - mean).T * dx_square).T

    # dx2 = dx_square
    # dmean2 = -np.sum(dx_square, axis=1, keepdims=True)
    # # 与batchnorm不同之处：axis=1, keepdims

    # dx3 = (1 / N) * dmean1
    # dx4 = (1 / N) * dmean2

    # dx = dx1 + dx2 + dx3 + dx4

    # 结果还是捣鼓不出来，这tm确定 slightly modifying 就好？？
    # ---------------------------------------------------------------
    
    # dx_hat = dout * gamma  # 乘法运算操作数顺序可以更换
    # dsigma = -0.5 * np.sum(dx_hat * (x - mean), axis=1) * np.power(var.squeeze() + eps, -1.5)
    # dmu = -np.sum(dx_hat / np.sqrt(var + eps), axis=1).squeeze() - 2 * dsigma * np.sum(x - mean, axis=1) / N
    # dx = dx_hat / np.sqrt(var + eps) + 2.0 * dsigma.reshape(-1, 1) * (x - mean) / N + dmu.reshape(-1, 1) / N
    # 这份网上的代码也还是不行，dx的误差仍旧有0.5，吐了
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    s, pad = conv_param['stride'], conv_param['pad']
    hh, ww = w.shape[2], w.shape[3]
    newx = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad), ), 'constant', constant_values=0)

    newh = 1 + (x.shape[2] + 2 * pad - hh) // s
    neww = 1 + (x.shape[3] + 2 * pad - ww) // s
    f_num = w.shape[0]
    out = np.zeros(shape=(x.shape[0], f_num, newh, neww))

    for n in range(x.shape[0]):
      for f in range(f_num):
        for i in range(newh):
          for j in range(neww):
            out[n, f, i, j] = np.sum(newx[n, :, i*s : i*s+hh, j*s : j*s+ww] * w[f, :, :, :]) + b[f]
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

      - x: Input data of shape (N, C, H, W)
      - w: Filter weights of shape (F, C, HH, WW)
      - b: Biases, of shape (F,)
      - conv_param: A dictionary with the following keys:
      --> dout: shape(N, F, H' W')

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache
    _, _, HH, WW = w.shape
    N, F, H_out, W_out = dout.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad), ), mode='constant', constant_values=0)
    # 这个x_pad才是前向传播时真正参与卷积的x

    db = np.sum(dout, axis=(0, 2, 3))
    dx_pad = np.zeros_like(x_pad)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)

    # 计算公式参考：https://blog.csdn.net/weixin_44538273/article/details/86678435
    # 代码参考：https://blog.csdn.net/LittleGreyWing/article/details/106967647 CNN节相应出黑字部分

    # 原本看了上述公式是打算直接照着卷积，然而发现shape不匹配，很迷惑
    # dw, _ = conv_forward_naive(x, dout.swapaxes(0, 1), np.zeros_like(b), {'stride': 1, 'pad': 0})
    # dx, _ = conv_forward_naive(np.pad(dout), w[::-1, ...])

    # 参考了网上代码后得到的结果
    for n in range(N):
      for f in range(F):
        for j in range(H_out):
          for k in range(W_out):
            dw[f] += x_pad[n, :, j*stride: j*stride+HH, k*stride: k*stride+WW] * dout[n, f, j, k]
            # x*w=z --> dL/dw = dL/dz * dz/dw
            # 前向卷积时：x_jk * w_00 + x_jk+1 * w_01 + ... = out_jk
            # 故：dw_00 = dout_jk * x_jk
            # 同理：dw_01 = dout_jk * x_jk+1 ...
            # 可得；dw = dout_jk * x[j:j+hh, k:k+ww]
            # 再考虑整个输出，所有样本，就有了上式：
            #   dw[f, :, :, :] = Sigma_j=0-h_k=0-w{ dout[n, f, j, k] * x_pad[n, :, j:j+hh, k: k+ww] }
            # 这个本质跟上述CSDN连接(计算公式)中卷积计算dw的图是一致的

            dx_pad[n, :, j*stride: j*stride+HH, k*stride: k*stride+WW] += w[f] * dout[n, f, j, k]
            # x*w=z --> dL/dw = dL/dz * dz/dx
            # 跟上述求dw一样的道理，都可以由 x_jk * w_00 + x_jk+1 * w_01 + ... = out_jk 推出
            # 感觉跟一般的反向传播一样，只是卷积中每个x元素的导数要将它所有参与计算的所有out元素的导数考虑进来(加起来)


    dx = dx_pad[:, :, pad: -pad, pad: -pad]
    # 最后去掉两边的pad就得到了原始的x

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param.values()
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    out = np.zeros(shape=(N, C, H_out, W_out))

    for n in range(N):
      for c in range(C):
        for i in range(H_out):
          for j in range(W_out):
            out[n, c, i, j] = np.max(x[n, c, i*stride : i*stride+pool_height, j*stride : j*stride+pool_width])

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param.values()
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    dx = np.zeros_like(x)

    # for n in range(N):
    #   for c in range(C):
    for i in range(H_out):
      for j in range(W_out):
        # mask = np.argmax(x[:, :, i*stride: i*stride+pool_height, j*stride: j*stride+pool_width], axis=(2, 3))
        # argmax不行，只能指定一个轴
        x_field = x[:, :, i*stride : i*stride+pool_height, j*stride : j*stride+pool_width]
        x_field_max = np.max(x_field, axis=(2, 3), keepdims=True)
        dx[:, :, i*stride: i*stride+pool_height, j*stride: j*stride+pool_width] = (x_field == x_field_max) * dout[:, :, i, j][:, :, None, None]
        # None用于在矩阵指定位置添加一维。如data的shape为(3, 3)，则data[:, None]的shape是(3,1,3)，data(:, :, None)的shape是(3, 3, 1)。
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

      # 实际上↑↑↑ running_mean 与 running_var 应该也是 shape(C,)

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    x_new = x.transpose(0, 2, 3, 1).reshape(N*H*W , C)
    out, cache = batchnorm_forward(x_new, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # x.transpose(): 相当于转置，与x.T一样
    # x.transpose(0, 2, 3, 1): 将1轴的换到最后一轴，原本的(N, C, H, W)将变为(N, H, W, C)

    # reshape之前交换C轴到最后，才能保证reshape后C轴上数据一致...
    # 想像图片是三层的饼干，N个饼干并排放在桌子上，mean与var形状是单个饼干从一面笔直穿过另一面取出的一条，即(1, 1, C)
    # 依据来源于题目：
    # If the feature map was produced using convolutions, then we expect every feature channel's statistics 
    # e.g. mean, variance to be relatively consistent both between different images, and different locations within the same image
    #  -- after all, every feature channel is produced by the same convolutional filter
    # 即卷积后的数据视为多层饼干，可假定每层上单个饼干的不同部分甚至各个饼干间都是相似的，因为他们都是同一卷积核计算结果

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    dout_new = dout.transpose(0, 2, 3, 1).reshape(N*H*W, C)
    dx, dgamma, dbeta = batchnorm_backward(dout_new, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
