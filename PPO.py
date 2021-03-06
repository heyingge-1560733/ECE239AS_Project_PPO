import copy
import pdb
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

class PPOClassical:
  def __init__(self, config):
    self.config = config 
    self.mem = config.memory(config.update_every, config.num_env, config.env, config.device)

    self.device = config.device
    
    self.gamma = config.gamma 
    self.epsilon = config.epsilon
    self.beta = config.entropy_beta

    self.model = config.model(config).to(self.device)
    self.old_model = config.model(config).to(self.device)
    self.old_model.load_state_dict(self.model.state_dict())
    
    self.optimiser = optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)
  
  def act(self, x):
    raise NotImplemented
  
  def add_to_mem(self, state, action, reward, log_prob, done):
    raise NotImplemented

  def learn(self, num_learn):
    raise NotImplemented
    
    
class PPOPixel:
  def __init__(self, config):
    
    self.mem = config.memory(
      config.update_every,
      config.num_env, 
      config.env, 
      config.device, 
      config.gamma, 
      config.gae_lambda
    )

    self.device = config.device
    
    self.lr = config.lr
    self.n_steps = config.n_steps
    self.lr_annealing = config.lr_annealing
    
    self.gae = config.gae
    self.epsilon_annealing = config.epsilon_annealing
    self.gamma = config.gamma
    self.epsilon = config.epsilon
    self.beta = config.entropy_beta


    self.model = config.model(config).to(self.device)
    
    self.old_model = config.model(config).to(self.device)
    self.old_model.load_state_dict(self.model.state_dict())

    self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr)

  def act(self, x):
    x = x.to(self.device)
    return self.old_model.act(x)

  def add_to_mem(self, s, a, r, log_p, v, done):
    self.mem.add(s, a, r, log_p, v, done)

  def learn(self, num_learn, last_value, next_done, global_step):
    # Learning Rate Annealing
    frac = 1.0 - (global_step - 1.0) / self.n_steps
    new_lr = self.lr * frac
    if self.lr_annealing:
      self.optimiser.param_groups[0]['lr'] = new_lr

    # Epsilon Annealing
    new_epsilon = self.epsilon
    if self.epsilon_annealing:
      new_epsilon = self.epsilon * frac

    # Calculate advantage and discounted returns using rewards collected from environments
    # self.mem.calculate_advantage(last_value, next_done)
    self.mem.calculate_advantage_gae(last_value, next_done)
    
    for i in range(num_learn):
      # itterate over mini_batches
      for mini_batch_idx in self.mem.get_mini_batch_idxs(mini_batch_size=256):

        # Grab sample from memory
        prev_S, prev_A, prev_log_probs, discounted_R, advantage, _ = self.mem.sample(mini_batch_idx)

        # find ratios
        _, log_probs, _, entropy = self.model.act(prev_S, prev_A)
        r = torch.exp(log_probs - prev_log_probs.detach())
        
        V = self.old_model.get_values(prev_S).reshape(-1)
        
        # Stats
        approx_kl = (prev_log_probs - log_probs).mean()

        advantages = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        # calculate surrogates
        surrogate_1 =  r * advantages
        surrogate_2 = torch.clamp(r, 1-self.epsilon, 1+self.epsilon) * advantages

        # Calculate losses
        new_V = self.model.get_values(prev_S).view(-1)

        V_loss_unclipped = (new_V - discounted_R)**2
        V_clipped = V + torch.clamp(new_V - V, -new_epsilon, new_epsilon)
        V_loss_clipped = (V_clipped - discounted_R)**2
        V_loss = 0.5 * torch.mean(torch.max(V_loss_clipped, V_loss_unclipped))


        pg_loss = -torch.min(surrogate_1, surrogate_2).mean()
        entropy_loss = entropy.mean()

        loss = pg_loss + V_loss - self.beta * entropy_loss

        # calculate gradient
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimiser.step()

        if torch.abs(approx_kl) > 0.05:
          break

        _, new_log_probs, _, _ = self.model.act(prev_S, prev_A)
        if (prev_log_probs - new_log_probs).mean() > 0.03:
          self.model.load_state_dict(self.old_model.state_dict())
          break
    
    # TODO: Check if this is in the right place
    self.old_model.load_state_dict(self.model.state_dict())

    return V_loss, pg_loss, approx_kl, entropy_loss, new_lr

  def save_weights(self, global_step, model_path):
    state = {'net':self.model.state_dict(), 'net_old':self.old_model.state_dict(), 'optimizer':self.optimiser.state_dict(), 'global_step': global_step}
    torch.save(state, model_path)

  def load_weights(self, model_path):
    checkpoint = torch.load(model_path)
    self.optimiser.load_state_dict(checkpoint['optimizer'])
    self.model.load_state_dict(checkpoint['net'])
    self.old_model.load_state_dict(checkpoint['net_old'])
    return checkpoint['global_step']

