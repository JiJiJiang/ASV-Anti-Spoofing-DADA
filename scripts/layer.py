import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.parameter import Parameter


class GradReversalFunction(Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha

        return input
    

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.alpha

        # None is to the grad to constant alpha
        return grad_input, None


class GradReversalLayer(nn.Module):
    def __init__(self, alpha):
        super(GradReversalLayer, self).__init__()
        self.alpha = alpha
    

    def forward(self, input, alpha=1.0):
        #return GradReversalFunction.apply(input, alpha)
        return GradReversalFunction.apply(input, self.alpha)
    

    def extra_repr(self):
        return 'alpha={}'.format(self.alpha)
