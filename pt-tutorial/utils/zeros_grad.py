有两种方式直接把模型的参数梯度设成0：

model.zero_grad()
optimizer.zero_grad()  # 当optimizer=optim.Optimizer(model.parameters())时，两者等效


如果想要把某一Variable的梯度置为0，只需用以下语句：
Variable.grad.data.zero_()




https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

Why do we need to call zero_grad() in PyTorch? [duplicate]
Leaving the gradients in place before calling .step() is useful in case you'd like to accumulate the gradient across multiple batches (as others have mentioned).
It's also useful for after calling .step() in case you'd like to implement momentum for SGD, and various other methods may depend on the values from the previous update's gradient.


16

In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. This is convenient while training RNNs. So, the default action is to accumulate the gradients on every loss.backward() call.

Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly. Else the gradient would point in some other directions than the intended direction towards the minimum (or maximum, in case of maximization objectives).

Here is a simple example:

import torch
from torch.autograd import Variable
import torch.optim as optim

def linear_model(x, W, b):
    return torch.matmul(x, W) + b

data, targets = ...

W = Variable(torch.randn(4, 3), requires_grad=True)
b = Variable(torch.randn(3), requires_grad=True)

optimizer = optim.Adam([W, b])

for sample, target in zip(data, targets):
    # clear out the gradients of all Variables
    # in this optimizer (i.e. W, b)
    optimizer.zero_grad()
    output = linear_model(sample, W, b)
    loss = (output - target) ** 2
    loss.backward()
    optimizer.step()
Alternatively, if you're doing a vanilla gradient descent then

W = Variable(torch.randn(4, 3), requires_grad=True)
b = Variable(torch.randn(3), requires_grad=True)

for sample, target in zip(data, targets):
    # clear out the gradients of Variables
    # (i.e. W, b)
    W.grad.data.zero_()
    b.grad.data.zero_()

    output = linear_model(sample, W, b)
    loss = (output - target) ** 2
    loss.backward()

    W -= learning_rate * W.grad.data
    b -= learning_rate * b.grad.data
