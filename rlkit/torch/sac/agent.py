import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class PEARLAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 policy,
                 **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder
        self.policy = policy

        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()

    def clear_z(self, num_tasks=1,clear_context=True,uniform=False):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu 
        self.z_vars = var

        if not uniform:
        # sample a new z from the prior
            self.sample_z()
        
        if uniform:
            self.z=(torch.rand([1,2])*2)-1
            

        # reset the context collected so far
        if clear_context:
            self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
            self.context_encoder.reset(num_tasks)

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, no,r = inputs

        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])


        if self.use_next_obs_in_context:
            data = torch.cat([o,a,no,r], dim=2)  
        else:
            data = torch.cat([o,a,no,r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors] 
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context ,high_prob_z=False):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        counter=0
        total=0
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        test_params= params.view( -1, self.context_encoder.output_size)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])

        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_z(high_prob_z=high_prob_z)

    def sample_z(self ,high_prob_z=False):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))] 
            z = [d.rsample() for d in posteriors] 
            self.z = torch.stack(z) 
            if high_prob_z and self.z_means[0][0]!=0 and self.z_means[0][1]!=0  :
                while self.z[0][0]>  (self.z_means[0][0] + 0.5*torch.sqrt(self.z_vars[0][0]) ) or self.z[0][0] < ( self.z_means[0][0] -0.5* torch.sqrt(self.z_vars[0][0]) ) or self.z[0][1] > ( self.z_means[0][1] +0.5*torch.sqrt(self.z_vars[0][1]) ) or self.z[0][1] < ( self.z_means[0][1] -0.5*torch.sqrt(self.z_vars[0][1]) ):
                    z = [d.rsample() for d in posteriors]
                    self.z = torch.stack(z)


        else:
            self.z = self.z_means

    def get_action(self, obs, deterministic=False,return_parameters=False,stricted_std=False,scale_std=None):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])   

        in_ = torch.cat([obs, z], dim=1)

        if return_parameters:
            return self.policy.get_actions_parameters(in_ ,deterministic=deterministic,stricted_std=stricted_std,scale_std=scale_std)
      
        else:
            return self.policy.get_action(in_, deterministic=deterministic,stricted_std=stricted_std)


    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context,return_Z_mean_var=False):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context)
        self.sample_z()

        task_z = self.z

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)
      
        z_means=self.z_means  
        z_vars=self.z_vars
        z_means = [z.repeat(b, 1) for z in z_means]
        z_means = torch.cat(z_means, dim=0)
        z_vars = [z.repeat(b, 1) for z in z_vars]
        z_vars = torch.cat(z_vars, dim=0)

        in_ = torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True,return_original_pretanh_prob=True,return_detailed_log_prob=True)

        if return_Z_mean_var:
            return policy_outputs, task_z, z_means,z_vars 
        else:
            return policy_outputs, task_z

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    @property
    def networks(self):
        return [self.context_encoder, self.policy]




