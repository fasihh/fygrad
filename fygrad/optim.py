from typing import List
from fygrad.data import xp
from fygrad.node import Node


class Optimizer:
    def __init__(self, params: List[Node], lr=0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params: List[Node], lr=0.01):
        super().__init__(params, lr)

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            p.value.data -= self.lr * p.grad.data


class Adam(Optimizer):
    def __init__(
        self,
        params: List[Node],
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-6,
    ):
        super().__init__(params, lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [None] * len(self.params)
        self.v = [None] * len(self.params)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            xpv = xp(p.device)

            if self.m[i] is None:
                self.m[i] = xpv.zeros_like(g)
                self.v[i] = xpv.zeros_like(g)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            p.value.data -= self.lr * m_hat / (xpv.sqrt(v_hat) + self.eps)
