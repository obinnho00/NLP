from typing import Callable, Iterable, Tuple
import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        # Validate and set hyperparameters
        # Learning rate, betas, and epsilon are validated to be within their respective ranges.
        # These parameters are crucial for the stability and convergence of the optimizer.
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                # The first and second moments are initialized as zeros. These moments are used to
                # adaptively change the learning rates for each parameter.
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Update first and second moments
                # The exponential moving averages are updated using the current gradient.
                # This allows the optimizer to be more sensitive to recent gradients.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute the denominator for parameter update
                # The square root of the second moment is used to normalize the gradient, ensuring
                # that the step size is not too large.
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Compute the step size
                # The step size is adjusted for bias correction to account for the fact that the
                # first and second moments are initialized as zeros.
                step_size = group['lr']
                if group['correct_bias']:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # Apply weight decay
                # Weight decay is applied directly to the weights before the gradient update.
                # This is a key difference between AdamW and the standard Adam optimizer.
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

                # Update parameters
                # The parameters are updated using the computed step size and the normalized gradient.
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
