import torch
import math
from torch import nn
from torch.distributions import MultivariateNormal

EPS = 1e-10

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss, gate_loss

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.tacotron2loss = Tacotron2Loss()

    def get_q_y_x(self, z_mu, z_logvar, z_prior_mu, z_prior_sigma):
        '''
        Calculate q(y_l|X) using Monte-Carlo sampling (1 sample)
        Input dimensions:
        q(z|X) dimensions: B x D
        p(z|Y) dimensions: K x D
        Output dimensions:
        q(y|X) dimensions: B x K
        B - batch size, K - numober of mixtures in GMM, D - number of dimensions in the distributions
        '''
        q_z_x = MultivariateNormal(z_mu,
            covariance_matrix=z_logvar.exp().diag_embed())
        k = z_prior_mu.size()[0]
        z_sample = q_z_x.rsample().unsqueeze(1).repeat([1, k, 1]) # B x K x D
        p_z_y = MultivariateNormal(z_prior_mu,
            scale_tril=z_prior_sigma.diag_embed())
        y_probs = p_z_y.log_prob(z_sample).exp()
        q_y_x = y_probs / y_probs.sum(dim=1, keepdim=True) #.add_(EPS)
        return q_y_x

    def kl_multivar_norm_diag(self, mu0, logvar0, mu1, logvar1):
        '''
        Calculate KL divergence between multivariate normal distributions with diagonal covariances.
        Input dimensions: B x K x D or B x D
        Output dimension: B x K or B
        B - batch size, K - numober of mixtures in GMM, D - number of dimensions in the distributions
        '''
        var0 = logvar0.exp()
        var1 = logvar1.exp()
        res = 0.5 * (var0 / var1 + (mu1 - mu0).pow(2) / var1 - 1 + logvar1 - logvar0).sum(dim=-1)
        return res

    def forward(self, model_output, latent_params, observed_params,
                latent_prior_params, observed_prior_params, y_observed, targets):
        '''
        Inputs:
        model_output, latent_params, observed_params, latent_prior_params - 
            values returned by VAE.forward()
        y_observed - speaker IDs, dimensions: batch size
        targets - see forward()
        '''
        mel_loss, gate_loss = self.tacotron2loss(model_output, targets)

        # Parameters of q(z_l | X), dimensions: B x D_l
        latent_mu, latent_logvar = latent_params
        # Parameters of q(z_o | X), dimensions: B x D_o
        observed_mu, observed_logvar = observed_params
        # Parameters of p(z_l | y_l), dimensions: K_l x D_l
        latent_prior_mu, latent_prior_sigma = latent_prior_params
        # Parameters of p(z_o | y_o), dimensions: K_o x D_o
        observed_prior_mu, observed_prior_sigma = observed_prior_params

        # q(y_l | X), dimensions: B x K_l
        q_yl_x = self.get_q_y_x(latent_mu, latent_logvar, latent_prior_mu,
                               latent_prior_sigma)

        # p(z_o | y_o) parameters, dimensions: B x D_o
        d_o = observed_mu.size()[1]
        yo_inds = y_observed.unsqueeze(1).repeat([1, d_o])
        observed_prior_mu_yo = observed_prior_mu.gather(0, yo_inds)
        observed_prior_sigma_yo = observed_prior_sigma.gather(0, yo_inds)

        # D_KL(q(z_o | X) || p(z_o || y_o)), dimensions: B
        kl_z_observed = self.kl_multivar_norm_diag(
            observed_mu, observed_logvar,
            observed_prior_mu_yo, observed_prior_sigma_yo.pow(2).log())

        # D_KL(q(z_l | X) || p(z_l || y_l)), dimensions: B x K_l
        k_l = latent_prior_mu.size()[0]
        b = latent_mu.size()[0]
        kl_z_latent = self.kl_multivar_norm_diag(
            latent_mu.unsqueeze(1).repeat([1, k_l, 1]),
            latent_logvar.unsqueeze(1).repeat([1, k_l, 1]),
            latent_prior_mu.unsqueeze(0).repeat([b, 1, 1]),
            latent_prior_sigma.unsqueeze(0).repeat([b, 1, 1]).pow(2).log())

        # D_KL(q(y_l | X) || p(y_l)), dimensions: B
        kl_y_latent = (q_yl_x * (q_yl_x.add_(EPS).log() + math.log(k_l))).sum(dim=1)

        elbo = mel_loss + kl_z_observed.mean() + (q_yl_x * kl_z_latent).sum(dim=1).mean() + kl_y_latent.mean()
        return elbo, mel_loss, gate_loss
