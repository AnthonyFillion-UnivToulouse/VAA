"""
This script contains functions that perform DAN and ODS joint learning
"""
import torch
import torch.nn as nn
import aux
from os import path, mkdir

from torch.distributions.kl import _kl_multivariatenormal_multivariatenormal as kl
from aux import loss_aux, print_last, save_dict

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class DANxODS(nn.Module):
    """
    A Data Assimilation Network and an Observed Dynamical System
    """

    def __init__(self, h0a, b, a, c, q0a, f, d, J, elboname="elbo2",
                 inflation=1.0, inflobs=1.0, inflsample=1.0):
        """
        h0a: mem Tensor(*, h_dim)
        b: mem Tensor(*, h_dim), time int ->
           mem Tensor(*, h_dim)
        a: mem Tensor(*, h_dim), obs Tensor(*, y_dim), time int ->
           mem Tensor(*, h_dim)
        c: mem Tensor(*, h_dim), time int ->
           Gaussian(*, 2*x_dim)
        q0a: Gaussian(*, x_dim)
        f: state Tensor(*, x_dim), time int ->
           state Gaussian(*, x_dim)
        d: state Tensor(*, x_dim), time in t ->
           obs Gaussian(*, y_dim)
        J : int
        return: DANxODS
        """
        nn.Module.__init__(self)
        self.b = b  # propagater
        self.a = a  # analyzer
        self.c = c  # procoder
        self.f = f  # forecaster
        self.d = d  # observer
        self.J = J  # reparameterization trick nb of samples
        self.elboname = elboname
        self.inflation = inflation
        self.inflobs = inflobs
        self.inflsample = inflsample

    def forward(self, y1, h0a, m0, t):
        """
        Applies the net, computes the ELBO,
        computes the next cycle DANxODS inner state

        y1: obs Tensor(I, y_dim)
        t: time int
        return: ELBO: float,
                q01b: prior Gaussian(I, 2*x_dim),
                q01a: posterior Gaussian(I, 2*x_dim)
                h1a: posterior memory Tensor(I, h_dim)
                q1: next posterior state pdf Gaussian(I, x_dim)
        """

        # Apply the net
        h1b = self.b(h0a, t)
        q01b = self.c(h1b, t)
        q0b, q1b = q01b.margin()
        h1a = self.a(torch.cat((h1b, y1), 1), t)
        q01a = self.c(h1a, t)
        q0a, q1a = q01a.margin()

        if (self.elboname == "elbo2"):
            # Computes the ELBO
            """
            kl may not cast a function over (x0) into
            a function over (x0, x1).
            Thus the following decomposition is used instead
            kl(p01,q0) = h(p0) - h(p01) + kl(p0, q0)
            """
            Da = torch.mean(q0a.entropy() - q01a.entropy() + kl(q0a, m0))
            Db = torch.mean(kl(q01a, q01b))
            # Reparameterization trick
            x01 = q01a.rsample(sample_shape=torch.Size([self.J]))
            x01 = x01.view(-1, x01.size(2))  # flatten the 2 batch dimension
            x0, x1 = x01[:, :q0a.dim], x01[:, q0a.dim:]
            # Expanding yt with the same recipe as x to be sure
            y1 = y1.unsqueeze(0).repeat(self.J, 1, 1).view(-1, y1.size(1))
            log_d = -torch.mean(self.d(x1, t).log_prob(y1))
            log_f = -torch.mean(self.f(x0, t).log_prob(x1))
            elbo = Da + self.inflsample*log_f + self.inflsample*self.inflobs*2*log_d + Db

        elif self.elboname == "elbo3":
            D1 = torch.mean(q0b.entropy() - q01b.entropy() + kl(q0b, m0))
            D2 = torch.mean(kl(q01a, q01b))
            x1a = q1a.rsample(sample_shape=torch.Size([self.J]))
            x1a = x1a.view(-1, x1a.size(2))
            y1 = y1.unsqueeze(0).repeat(self.J, 1, 1).view(-1, y1.size(1))
            log_d = -torch.mean(self.d(x1a, t).log_prob(y1))
            x01b = q01b.rsample(sample_shape=torch.Size([self.J]))
            x01b = x01b.view(-1, x01b.size(2))
            x0b, x1b = x01b[:, :q0b.dim], x01b[:, q0b.dim:]
            log_f = -torch.mean(self.f(x0b, t).log_prob(x1b))
            elbo = self.inflsample*log_f + D1 + self.inflation*(self.inflsample*self.inflobs*log_d + D2)

        elif self.elboname == "elbo4":
            Hb = torch.mean(q0b.entropy() - q01b.entropy() + kl(q0b, m0))
            Ha = torch.mean(q0a.entropy() - q01a.entropy() + kl(q0a, q0b))
            x01a = q01a.rsample(sample_shape=torch.Size([self.J]))
            x01a = x01a.view(-1, x01a.size(2))
            x0a, x1a = x01a[:, :q0a.dim], x01a[:, q0a.dim:]
            y1 = y1.unsqueeze(0).repeat(self.J, 1, 1).view(-1, y1.size(1))
            x01b = q01b.rsample(sample_shape=torch.Size([self.J]))
            x01b = x01b.view(-1, x01b.size(2))
            x0b, x1b = x01b[:, :q0b.dim], x01b[:, q0b.dim:]
            logb_f = -torch.mean(self.f(x0b, t).log_prob(x1b))
            loga_f = -torch.mean(self.f(x0a, t).log_prob(x1a))
            loga_d = -torch.mean(self.d(x1a, t).log_prob(y1))
            elbo = Hb + self.inflsample*logb_f +\
                   self.inflation*(Ha + self.inflsample*loga_f + self.inflsample*self.inflobs*loga_d)

        elif self.elboname == "elbo5":
            Hb = torch.mean(q0b.entropy() - q01b.entropy() + kl(q0b, m0))
            Ha = torch.mean(q0a.entropy() - q01a.entropy() + kl(q0a, m0))
            x01a = q01a.rsample(sample_shape=torch.Size([self.J]))
            x01a = x01a.view(-1, x01a.size(2))
            x0a, x1a = x01a[:, :q0a.dim], x01a[:, q0a.dim:]
            y1 = y1.unsqueeze(0).repeat(self.J, 1, 1).view(-1, y1.size(1))
            x01b = q01b.rsample(sample_shape=torch.Size([self.J]))
            x01b = x01b.view(-1, x01b.size(2))
            x0b, x1b = x01b[:, :q0b.dim], x01b[:, q0b.dim:]
            logb_f = -torch.mean(self.f(x0b, t).log_prob(x1b))
            loga_f = -torch.mean(self.f(x0a, t).log_prob(x1a))
            loga_d = -torch.mean(self.d(x1a, t).log_prob(y1))
            elbo = Hb + self.inflsample*logb_f +\
                self.inflation*(Ha + self.inflsample*loga_f + self.inflsample*self.inflobs*loga_d)

        return elbo, q01b, q01a, h1a, q1a


def train_test(net, check, T, h0a, m0, x0, f_truth, d_truth, I, direxp,
               outputs, *args):
    """
    Trains the DANxODS, save and print outputs

    net: DANxODS, the network
    check: int, checkpoint number of steps
    T: int, training time length
    x0: Tensor(I, x_dim), initial truth
    f_truth: Tensor(*, x_dim) -> Gaussian(*, x_dim), forecaster
    d_truth: Tensor(*, x_dim) -> Gaussian(*, y_dim), observer
    I: int, batch size
    direxp: Str, experiment directory
    outputs: Dict{Str: [float]}, dict of computed losses
    *args: optimizer => train mode, () => test mode
    """

    l_args = len(args)
    aux.requires((l_args == 0) or (l_args == 1), "train_test bad *args")
    mode = "test_"
    if l_args == 1:
        mode = "train_"
        optimizer = args[0]
    keys = outputs.keys()

    # Outputs ## TODO

    for t in range(1, T+1):

        # Generates data
        x1 = torch.squeeze(f_truth(x0, t).rsample(sample_shape=[1]))
        y1 = torch.squeeze(d_truth(x1, t).rsample(sample_shape=[1]))

        # Optimization if train mode
        if l_args == 1:
            optimizer.zero_grad()
        elbo, q01b, q01a, h1a, q1a = net(y1, h0a, m0, t)
        if l_args == 1:
            elbo.backward()
            optimizer.step()

        # update the net inner state (version très prudente)
        h0a = h1a.detach().clone()
        m0 = aux.Gaussian(q1a.mean.detach().clone(),
                          q1a.scale_tril.detach().clone())

        # Outputs
        with torch.no_grad():
            x01 = torch.cat((x0, x1), 1)
            outputs["elbo"].append(elbo.item())
            for key in keys:
                if key != "elbo":
                    outputs[key].append(loss_aux(key, q01b, q01a, x01).item())

        # Prepare next cycle
        x0 = x1.clone()

        # Checkpoint
        # (warning: if the function is hard terminated,
        # the last checkpoint may be far away)
        if (t % check == 1) or (t == T):
            print("")
            print_last(**{str(mode)+"t": t}, **outputs)
            save_dict(direxp + mode,
                      outputs=outputs,
                      net=net.state_dict(),
                      h0a=h0a,
                      m0=m0,
                      x0=x0)
            if l_args == 1:
                save_dict(direxp + mode,
                          optimizer=optimizer.state_dict())


def experiment(**kwargs):
    """
    Instantiate the experiment parameters and repeat trainings and testings
    """

    direxp = kwargs.get("direxp", "")
    if not path.exists(direxp):
        mkdir(direxp)
    check = kwargs.get("check", 2)

    x_dim = kwargs.get("x_dim", 40)
    xx_dim = x_dim*2
    y_dim = x_dim
    h_dim = kwargs.get("h_dim", 20*xx_dim)
    T = kwargs.get("T", 100)
    I = kwargs.get("I", 512)
    J = kwargs.get("J", 8)
    deep = kwargs.get("deep", 5)

    elboname = kwargs.get("elboname", "elbo2")
    inflation = kwargs.get("inflation", 1.0)
    inflobs = kwargs.get("inflobs", 1.0)
    inflsample = kwargs.get("inflsample", 1.0)
    h0a = kwargs.get("h0a", torch.zeros(h_dim).expand(I, -1))
    b_class = kwargs.get("b_class", aux.FcZero)
    b_args = kwargs.get("b_args", (deep*[h_dim],))
    b = b_class(*b_args)
    a_class = kwargs.get("a_class", aux.FcZero)
    a_args = kwargs.get("a_args", ((deep-1)*[h_dim+y_dim] + [h_dim],))
    a = a_class(*a_args)
    c_class = kwargs.get("c_class", aux.FcZero_cpdf)
    c_args = kwargs.get("c_args", (xx_dim, [h_dim, xx_dim*(xx_dim+3)//2]))
    c = c_class(*c_args)
    q0a = aux.Gaussian(torch.zeros(x_dim).expand(I, -1),
                       torch.eye(x_dim).expand(I, -1, -1))
    f_class = kwargs.get("f_class", aux.FcZero_mu_cst_Lambda_cpdf)
    f_args = kwargs.get("f_args", (deep*[x_dim],))
    f = f_class(*f_args)
    d_class = kwargs.get("d_class", aux.FcZero_mu_cst_Lambda_cpdf)
    d_args = kwargs.get("d_args", ((deep-1)*[x_dim] + [y_dim],))
    d = d_class(*d_args)
    net = DANxODS(h0a, b, a, c, q0a, f, d, J, elboname,
                  inflation, inflobs, inflsample)
    h0a = kwargs.get("h0a",
                     torch.zeros(I, h_dim))
    m0 = kwargs.get("m0",
                    aux.Gaussian(x_dim, torch.zeros(I, x_dim*(x_dim+3)//2)))
    x0 = kwargs.get("x0",
                    3*torch.ones(I, x_dim) + torch.randn(I, x_dim))
    f_truth_class = kwargs.get("f_truth_class", aux.Lorenz_cpdf)
    f_truth_args = kwargs.get("f_truth", ())
    f_truth = f_truth_class(*f_truth_args)
    d_truth_class = kwargs.get("d_truth_class", aux.Id_mu_cpdf)
    d_truth_args = kwargs.get("d_truth_args", ())
    d_truth = d_truth_class(*d_truth_args)
    optimizer_class = kwargs.get("optimizer_class", torch.optim.Adam)
    optimizer_kwargs = kwargs.get("optimizer_kwargs", {"lr": 10**-4})
    optimizer = optimizer_class(net.parameters(), **optimizer_kwargs)
    default_keys = ["elbo",
                    "logpdf_01b", "logpdf_01a",
                    "rmse_0b", "rmse_1b", "rmse_0a", "rmse_1a"]
    train_outputs = kwargs.get("train_outputs",
                               {key: [] for key in default_keys})
    test_outputs = kwargs.get("test_outputs",
                              {key: [] for key in default_keys})
    resume_training = kwargs.get("resume_training", False)
    modes = kwargs.get("modes", 1)
    seeds = kwargs.get("seeds", range(1, len(modes)+1))

    r = 1
    for mode, seed in zip(modes, seeds):
        print("### repetition "+str(r)+" ###")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if mode == "train":
            # TODO: control random seeds
            if resume_training:
                net.load_state_dict(torch.load(direxp+"train_net.pt"))
                optimizer.load_state_dict(torch.load(direxp+"train_optimizer.pt"))
                # train_outputs = torch.load(direxp+"train_outputs.pt")
                h0a = torch.load(direxp+"train_h0a.pt")
                m0 = torch.load(direxp+"train_m0.pt")
                x0 = torch.load(direxp+"train_x0.pt")
            resume_training = True
            train_test(net, check, T, h0a, m0, x0,
                       f_truth, d_truth, I, direxp,
                       train_outputs, optimizer)
            # TODO : reset inner state and x0
        if mode == "test":
            net.load_state_dict(torch.load(direxp+"train_net.pt"))
            h0a = torch.zeros(I, h_dim)
            m0 = aux.Gaussian(x_dim, torch.zeros(I, x_dim*(x_dim+3)//2))
            x0 = 3*torch.ones(I, x_dim) + torch.randn(I, x_dim)
            with torch.no_grad():
                train_test(net, check, T, h0a, m0, x0,
                           f_truth, d_truth, I, direxp,
                           test_outputs)
        # note that outputs are modified in-place because they are dict
        r += 1
