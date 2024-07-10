import torch
import numpy as np
import math
import torch.nn as nn


class Qnet(nn.Module):
    def __init__(self, n_dims=6, n_out=1, n_hid=32, layer=nn.Linear, relu=True):
        super(Qnet, self).__init__()
        self.net = nn.Sequential(
            layer(n_dims, n_hid),
            nn.ReLU() if relu else nn.SiLU(n_hid),
            layer(n_hid, n_hid),
            nn.ReLU() if relu else nn.SiLU(n_hid),
            layer(n_hid, n_out)
        )

    def forward(self, x_t, s_t):
        h = torch.cat((x_t, s_t), 1)
        h = self.net(h)
        return h


class AMLoss:
    def __init__(self, s, q_t, config, device):
        self.u0 = 0.5
        self.batch_size = config.train.batch_size
        self.s = s
        self.q_t = q_t
        self.device = device
        self.Q = Qnet()
        self.Q.load_state_dict(torch.load("drive/MyDrive/Diploma/Action_matching/Q_function_neg.pt"))
        self.Q.to(device)
        self.Q.eval()

    def sample_t(self, n):
        u = (self.u0 + np.sqrt(2)*np.arange(n)) % 1
        return torch.tensor(u).view(-1,1).to(self.device)

    def add_noise_random(self, q_t, t):
        x_t, t = q_t(t)
        choice = np.random.choice([0, 1])
        if choice:
            x_t += torch.zeros_like(x_t, device=x_t.device).normal_(0, 100)
            return x_t, t
        else:
            return x_t, t
    
    def eval_loss(self):
        q_t, s = self.q_t, self.s
        bs = self.batch_size
        
        t = self.sample_t(bs)
        x_t, t = self.add_noise_random(q_t, t)
        x_t.requires_grad, t.requires_grad = True, True
        s_t = s(t, x_t)
        assert (2 == s_t.dim())
        dsdt, dsdx = torch.autograd.grad(s_t.sum(), [t, x_t], create_graph=True, retain_graph=True)
        x_t.requires_grad, t.requires_grad = False, False
        
        loss = 0.5*(dsdx**2).sum(1, keepdim=True) + dsdt.sum(1, keepdim=True)
        loss = loss.squeeze()
        
        t_0 = torch.zeros([bs, 1], device=self.device)
        x_0, _ = q_t(t_0)
        loss = loss + s(t_0,x_0).squeeze()
        t_1 = torch.ones([bs, 1], device=self.device)
        x_1, _ = q_t(t_1)
        loss = loss - s(t_1,x_1).squeeze()
        loss = loss - torch.clamp(self.Q(x_t, dsdx), min=-2000, max=2000)
        return loss.mean()


class SMLoss:
    def __init__(self, s, q_t, config, device, sliced=True):
        self.u0 = 0.5
        self.batch_size = config.train.batch_size
        self.s = s
        self.q_t = q_t
        self.device = device
        self.div = div
        if sliced:
            self.div = divH

    def sample_t(self, n):
        u = (self.u0 + np.sqrt(2)*np.arange(n)) % 1
        return torch.tensor(u).view(-1,1).to(self.device)
    
    def eval_loss(self):
        q_t, s = self.q_t, self.s
        bs = self.batch_size
        
        t = self.sample_t(bs)
        x_t, t = q_t(t)
        dsdx = self.div(s, t, x_t, create_graph=True)
        
        loss = 0.5*(s(t, x_t)**2).sum(1, keepdim=True) + dsdx
        loss = loss.squeeze()
        return loss.mean()

def div(v, t, x, create_graph=False):
    f = lambda x: v(t, x).sum(0)
    J = torch.autograd.functional.jacobian(f, x, create_graph=create_graph).swapaxes(0,1)
    return J.diagonal(dim1=1,dim2=2).sum(1, keepdim=True)

def divH(v, t, x, create_graph=False):
    eps = torch.randint_like(x, low=0, high=2).float() * 2 - 1.
    x.requires_grad = True
    dxdt = v(t, x)
    div = (eps*torch.autograd.grad(dxdt, x, grad_outputs=eps, create_graph=create_graph)[0]).sum(1)
    x.requires_grad = False
    return div
    