import torch
import torch.nn as nn
import lorenz95 as l95
from torch.nn import Module
from torch.distributions.multivariate_normal import MultivariateNormal


def requires(test, s=""):
    """
    raise an error with text str is the test fails
    test: Bool
    s: String
    return : None
    """
    if (not test):
        raise NameError(s)


def vec_to_inds(x_dim, vec_dim):
    """
    Computes the indices of scale_tril coeffs,
    scale_tril is filled main diagonal first

    x_dim: dimension of the random variable
    vec_dim: dimension of the vector containing the coeffs of loc and scale_tril
    """
    ldiag, d, c = x_dim, 0, 0  # diag length, diag index, column index
    inds = [[], []]  # list of line and column indexes
    for i in range(vec_dim - x_dim):  # loop over the non-mean coeff
        inds[0].append(c+d)  # line index
        inds[1].append(c)  # column index
        if c == ldiag-1:  # the current diag end is reached
            ldiag += -1  # the diag length is decremented
            c = 0  # the column index is reinitialized
            d += 1  # the diag index is incremented
        else:  # otherwize, only the column index is incremented
            c += 1
    return inds


class FcZero(Module):
    """
    Fully connected neural network with ReZero trick
    """
    def __init__(self, layers):
        """
        layers: the list of the layers dimensions
        """
        Module.__init__(self)
        n = len(layers)
        self.lins = nn.ModuleList(
            [nn.Linear(d0, d1) for d0, d1 in zip(layers[:-1], layers[1:])])
        self.acts = nn.ModuleList([nn.LeakyReLU()]*(n-1))
        self.alphas = torch.nn.Parameter(torch.zeros(n-1))

    def forward(self, h, t):
        for lin, act, alpha in zip(self.lins[:-1], self.acts, self.alphas):
            h = h + alpha*act(lin(h))
        return self.lins[-1](h)


class Gaussian(MultivariateNormal):
    """
    A Gaussian pdf inheriting pytorch's Gaussian pdfs
    """

    def __init__(self, *args):
        """
        args is either a (loc, scale_tril) or a (x_dim, vec)
        """

        # args is a (x_dim, vec)
        if isinstance(args[0], int):
            x_dim, vec = args
            vec_dim = vec.size(-1)
            if vec_dim == x_dim + 1:
                loc = vec[:, :x_dim]
                scale_tril = torch.eye(x_dim)\
                                  .reshape((1, x_dim, x_dim))\
                                  .repeat(vec.size(0), 1, 1)
                scale_tril = torch.exp(vec[:, x_dim])\
                                  .view(vec.size(0), 1, 1)*scale_tril
            else:
                inds = vec_to_inds(x_dim, vec_dim)
                loc = vec[:, :x_dim]
                lbda = torch.cat(
                    (torch.exp(vec[:, x_dim:2*x_dim]),  # ensures positive diag
                     vec[:, 2*x_dim:]), 1)
                scale_tril = torch.zeros(vec.size(0), x_dim, x_dim)
                scale_tril[:, inds[0], inds[1]] = lbda
            MultivariateNormal.__init__(self, loc=loc, scale_tril=scale_tril)

        # args is a loc, scale_tril
        else:
            MultivariateNormal.__init__(self, loc=args[0], scale_tril=args[1])

        self.dim = self.event_shape[0]  # the RV dimension USELESS

    def margin(self):
        """
        If self is a pdf over (x0, x1),
        return is the pair of marginals pdfs of x0 and x1
        """

        n01 = self.mean.size(-1)
        requires(n01 % 2 == 0, "the RV dim is not pair")
        n = n01 // 2
        """
        The Multivariate-Normal class may not accept rectangular scale_tril
        so the covariance of x1 should be computed,
        then a Cholesky decomposition is performed to get L1, its sqrt.
        """
        loc01 = self.mean
        loc0, loc1 = loc01[:, :n], loc01[:, n:]
        L01 = self.scale_tril
        L0 = L01[:, :n, :n]
        C01 = self.covariance_matrix
        C1 = C01[:, n:, n:]
        L1 = torch.cholesky(C1)
        return Gaussian(loc0, L0), Gaussian(loc1, L1)


class Id():
    """
    Dumb Id class behaving like a id function
    """
    def __init__(self):
        pass

    def __call__(self, x, t):
        return x


class Lorenz_cpdf():
    """
    A Gaussian cpdf having lorenz95 as mean and a cst cov matrix
    """
    def __init__(self, sigma_Q=0.1, Ndt=1, dt=0.05):
        self.Ndt = Ndt
        self.dt = dt
        self.sigma_Q = sigma_Q

    def __call__(self, x, t):
        """
        Making Lambda_Q here allows to use x's dimensions
        but may be bad for performance
        """
        Lambda_Q = self.sigma_Q*torch.eye(x.size(-1))\
                                     .expand(x.size(0), -1, -1)
        return Gaussian(l95.M(x, self.Ndt, self.dt), Lambda_Q)


class Id_mu_cpdf():
    """
    A Gaussian cpdf having id as mean and a cst cov matrix
    """
    def __init__(self, sigma_R=1):
        self.sigma_R = sigma_R

    def __call__(self, x, t):
        """
        Making Lambda_R here allows to use x's dimensions
        but may be bad for performance
        """
        Lambda_R = self.sigma_R*torch.eye(x.size(-1))\
                                     .expand(x.size(0), -1, -1)
        return Gaussian(x, Lambda_R)


class FcZero_mu_cst_Lambda_cpdf(Module):
    """
    A Gaussian cpdf having a FcZero net as mean
    and a cst matrix as cov
    """
    def __init__(self, layers, vec_dim):
        Module.__init__(self)
        x_dim = layers[-1]
        self.loc = FcZero(layers)
        self.scale_vec = nn.Parameter(torch.zeros(vec_dim - x_dim))

    def forward(self, x, t):
        loc_ = self.loc(x, t)
        vec = torch.cat((loc_, self.scale_vec.expand(x.size(0), -1)), 1)
        return Gaussian(loc_.size(-1), vec)


class Id_mu_cst_Lambda_cpdf(Module):
    """
    A Gaussian cpdf having id as mean
    and a learnable cst matrix as cov
    """
    def __init__(self, x_dim, vec_dim):
        """
        n: mu dimension
        ndiag: number of learnable
        """
        Module.__init__(self)
        self.scale_vec = nn.Parameter(torch.zeros(vec_dim - x_dim))

    def forward(self, x, t):
        vec = torch.cat((x, self.scale_vec.expand(x.size(0), -1)), 1)
        return Gaussian(x.size(-1), vec)


class Lorenz_mu_cst_Lambda_cpdf(Module):
    """
    A Gaussian cpdf having l95 as mean
    and a learnable cst matrix as cov
    """
    def __init__(self, x_dim, vec_dim, Ndt=1, dt=0.05):
        """
        n: mu dimension
        ndiag: number of learnable
        """
        Module.__init__(self)
        self.scale_vec = nn.Parameter(torch.zeros(vec_dim - x_dim))
        self.Ndt = Ndt
        self.dt = dt

    def forward(self, x, t):
        vec = torch.cat((l95.M(x, self.Ndt, self.dt),
                         self.scale_vec.expand(x.size(0), -1)), 1)
        return Gaussian(x.size(-1), vec)


class FcZero_cpdf(Module):
    """
    A Gaussian cpdf from a FcZero net,
    the net outputs a vec that is transformed into a Gaussian
    """

    def __init__(self, dim, layers):
        Module.__init__(self)
        self.f = FcZero(layers)
        self.dim = dim

    def forward(self, x, t):
        return Gaussian(self.dim, self.f(x, t))


class RK_mu_cst_Lambda_cpdf(Module):
    def __init__(self, x_dim, vec_dim, window=(-2, -1, 0, 1), N=1, dt=0.05):
        Module.__init__(self)
        self.x_dim = x_dim
        self.vec_dim = vec_dim
        self.scale_vec = nn.Parameter(torch.zeros(vec_dim - x_dim))
        self.N = N
        self.dt = dt
        self.window = window
        self.diameter = len(window)
        self.lin = nn.Linear(in_features=self.diameter,
                             out_features=1,
                             bias=True)
        self.bil = nn.Bilinear(in1_features=self.diameter,
                               in2_features=self.diameter,
                               out_features=1,
                               bias=False)

    def EDO(self, x):
        v = torch.cat(
            [torch.roll(x.unsqueeze(1), i, 2) for i in self.window], 1)
        v = torch.transpose(v, 1, 2)
        v_flat = v.reshape(-1, self.diameter)
        dx = self.lin(v_flat) + self.bil(v_flat, v_flat)
        return dx.view(x.size(0), x.size(1))

    def RK(self, x):
        for _ in range(self.N):
            k1 = self.EDO(x)
            k2 = self.EDO(x + 0.5*self.dt*k1)
            k3 = self.EDO(x + 0.5*self.dt*k2)
            k4 = self.EDO(x + self.dt*k3)
            x = x + (self.dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
        return x

    def forward(self, x, t):
        y = self.RK(x)
        vec = torch.cat((y, self.scale_vec.expand(x.size(0), -1)), 1)
        return Gaussian(x.size(-1), vec)


# Auxiliary losses
def loss_aux(key, q01b, q01a, x01):
    """
    key: string, the loss name
    q01b: prior pdf over 2 consecutive states
    q01a: posterior pdf over 2 consecutive states
    x01: true state
    """
    n01 = q01b.dim
    requires(n01 % 2 == 0, "the RV dim is not pair")
    n = n01 // 2
    if key == "logpdf_01b":
        return -torch.mean(q01b.log_prob(x01))
    elif key == "logpdf_01a":
        return -torch.mean(q01a.log_prob(x01))
    elif key == "rmse_0b":
        return torch.mean(
            torch.norm(
                x01[:, :n]-q01b.mean[:, :n], dim=1)/(n**0.5))
    elif key == "rmse_1b":
        return torch.mean(
            torch.norm(
                x01[:, n:]-q01b.mean[:, n:], dim=1)/(n**0.5))
    elif key == "rmse_0a":
        return torch.mean(
            torch.norm(
                x01[:, :n]-q01a.mean[:, :n], dim=1)/(n**0.5))
    elif key == "rmse_1a":
        return torch.mean(
            torch.norm(
                x01[:, n:]-q01a.mean[:, n:], dim=1)/(n**0.5))
    else:
        raise NameError(key + " is not defined in loss_aux")


def print_last(**kwargs):
    """
    prints the output dict last values
    """
    for key, val in kwargs.items():
        if isinstance(val, type([])):
            if val == []:
                s = ""
            else:
                s = '{:.2e}'.format(val[-1])
        else:
            s = str(val)
        print(key + " = " + s)


def save_dict(prefix, **kwargs):
    """
    saves the output dict
    """
    for key, val in kwargs.items():
        torch.save(val, prefix + key + ".pt")
