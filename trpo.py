import numpy as np

import torch
from torch.autograd import Variable
from utils import *
import scipy.optimize

#################################################
#
# TRPO STEP
#
#################################################

class trpo_step():
    def __init__(self, model, get_loss, get_kl, max_kl, damping, trpo_functions, log):
        super(trpo_step, self).__init__()

        self.model = model
        self.get_loss = get_loss
        self.log = log
        """log = [actions, states, advantages, fixed_log_prob]"""
        self.loss = self.get_loss(self.log[0], self.log[1], self.log[2], self.log[3])
        #self.grads = torch.autograd.grad(self.get_loss, model.parameters(), retain_graph=True)
        #self.loss_grad = torch.cat([grad.view(-1) for grad in self.grads]).data


        self.grads = torch.autograd.grad(self.loss, self.model.parameters(), retain_graph=True)
        self.loss_grad = torch.cat([grad.view(-1) for grad in self.grads]).data

        self.get_kl = get_kl
        self.max_kl = max_kl
        self.damping = damping
        self.trpo_func = trpo_functions

    def Fvp(self, v):

        kl = self.get_kl
        kl = kl.mean()

        grads = torch.autograd.grad(kl, self.model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * torch.autograd.Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, self.model.parameters(), retain_graph=True)
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * self.damping

    def conjugate_gradients(self, Avp, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = Avp(p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def linesearch(self, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
        #fval = f(True).data
        # fval = self.get_loss(True).data
        fval = self.get_loss(self.log[0], self.log[1], self.log[2], self.log[3], True).data
        print("fval before", fval.item())
        for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep
            self.trpo_func.set_flat_params_to(self.model, xnew)
            #newfval = self.get_loss(True).data
            newfval = self.get_loss(self.log[0], self.log[1], self.log[2], self.log[3], True).data
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                print("fval after", newfval.item())
                return True, xnew
        return False, x

    def eval(self):

        """samin: """
        # gives similar output up-until self.conjugate_gradients() step
        stepdir = self.conjugate_gradients(self.Fvp, -self.loss_grad, 10)

        shs = 0.5 * (stepdir * self.Fvp(stepdir)).sum(0, keepdim=True)

        lm = torch.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm[0]

        neggdotstepdir = (-self.loss_grad * stepdir).sum(0, keepdim=True)
        print(("lagrange multiplier:", lm[0], "grad_norm:", self.loss_grad.norm()))

        prev_params = self.trpo_func.get_flat_params_from(self.model)

        """ PROBLEM: doesnt give similar result"""
        success, new_params = self.linesearch(prev_params, fullstep,
                                         neggdotstepdir / lm[0])
        self.trpo_func.set_flat_params_to(self.model, new_params)

        return self.loss

##################################################
#
# UPDATE PARAMETERS
#
##################################################


class update_params():
    def __init__(self, batch, value_net, policy_net, args, trpo_functions):
        super(update_params, self).__init__()

        self.batch = batch
        self.value_net = value_net
        self.policy_net = policy_net
        self.args = args
        # important that we declare them as functions ()
        self.trpo_func = trpo_functions()

        self.state_trac = []
        self.target_trac = []

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(self, flat_params):
        states = self.state_trac
        targets = self.target_trac
        self.trpo_func.set_flat_params_to(self.value_net, torch.Tensor(flat_params))
        for param in self.value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = self.value_net(torch.autograd.Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in self.value_net.parameters():
            value_loss += param.pow(2).sum() * self.args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), self.trpo_func.get_flat_grad_from(self.value_net).data.double().numpy())


    def get_loss(self, actions, states, advantages, fixed_log_prob, volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = self.policy_net(torch.autograd.Variable(states))
        else:
            action_means, action_log_stds, action_stds = self.policy_net(torch.autograd.Variable(states))

        log_prob = self.trpo_func.normal_log_density(torch.autograd.Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -torch.autograd.Variable(advantages) * torch.exp(log_prob - torch.autograd.Variable(fixed_log_prob))
        return action_loss.mean()

    def get_kl(self, states):
        mean1, log_std1, std1 = self.policy_net(torch.autograd.Variable(states))

        mean0 = torch.autograd.Variable(mean1.data)
        log_std0 = torch.autograd.Variable(log_std1.data)
        std0 = torch.autograd.Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)


    def execute(self):
        rewards = torch.Tensor(self.batch.reward)
        masks = torch.Tensor(self.batch.mask)
        actions = torch.Tensor(np.concatenate(self.batch.action, 0))
        states = torch.Tensor(self.batch.state)
        values = self.value_net(torch.autograd.Variable(states))

        returns = torch.Tensor(actions.size(0), 1)
        deltas = torch.Tensor(actions.size(0), 1)
        advantages = torch.Tensor(actions.size(0), 1)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.args.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + self.args.gamma * prev_value * masks[i] - values.data[i]
            advantages[i] = deltas[i] + self.args.gamma * self.args.tau * prev_advantage * masks[i]

            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]

        targets = torch.autograd.Variable(returns)

        self.state_trac = states
        self.target_trac = targets

        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(self.get_value_loss, self.trpo_func.get_flat_params_from(self.value_net).double().numpy(), maxiter=25)
        self.trpo_func.set_flat_params_to(self.value_net, torch.Tensor(flat_params))

        advantages = (advantages - advantages.mean()) / advantages.std()

        action_means, action_log_stds, action_stds = self.policy_net(torch.autograd.Variable(states))

        fixed_log_prob = self.trpo_func.normal_log_density(torch.autograd.Variable(actions), action_means,
                                                           action_log_stds, action_stds).data.clone()

        log = [actions, states, advantages, fixed_log_prob]
        #take_trpo_step = trpo_step(self.policy_net, self.get_loss(actions, states, advantages, fixed_log_prob), self.get_kl(states), self.args.max_kl, self.args.damping, self.trpo_func)
        take_trpo_step = trpo_step(self.policy_net, self.get_loss,
                                   self.get_kl(states), self.args.max_kl, self.args.damping, self.trpo_func, log)
        #take_trpo_step = trpo_step(self.policy_net, self.args.max_kl, self.args.damping, self.trpo_func)


        take_trpo_step.eval()