import os
import torch
import numpy as np
from collections import deque
from PPO import PPOPixel
import wandb
from envs import make_env, VecPyTorch

def train_pixel(config):
  envs = config.env
  scores_ave = []
  if len(scores_ave) > 99:
    scores_ave.pop(0)

  agent = PPOPixel(config)

  if os.path.exists(config.model_path):
    g_step = agent.load_weights(config.model_path)
  else:
    g_step = 0

  if config.wandb:
    wandb.watch(agent.model)

  while g_step < config.n_steps:
    states = envs.reset()
    values, dones = None, None

    while agent.mem.isFull() == False:
      g_step += config.num_env

      # Take actions
      with torch.no_grad():
        actions, log_probs, values, _ = agent.act(states)
      next_states, rewards, dones, infos = envs.step(actions)
      # envs.render()

      # Add to memory buffer
      agent.add_to_mem(states, actions, rewards, log_probs, values, dones)
      # Update state
      states = next_states

      # Book Keeping
      for info in infos:
        if 'episode' in info:
          score = info['episode']['r']
          if config.wandb:
            wandb.log({
              "episode_reward": score,
              "global_step": g_step
            })

          scores_ave.append(score)

    # update and learn
    value_loss, pg_loss, approx_kl, approx_entropy, lr_now = agent.learn(config.num_learn, values, dones, g_step)
    agent.mem.reset()

    if config.wandb:
      wandb.log({
        "value_loss": value_loss,
        "policy_loss": pg_loss,
        "approx_kl": approx_kl,
        "approx_entropy": approx_entropy,
        "global_step": g_step,
        "learning_rate": lr_now
      })

    if not g_step % 204800:
      agent.save_weights(g_step, config.model_path.split('/')[0] + '/%d.pth' % g_step)

    print("Global Step: %d	Average Score: %.2f"%(g_step, np.mean(scores_ave)))

"""
def train_pixel(config):
  envs = config.env
  scores_deque = deque(maxlen=100)
  scores = []
  average_scores = []
  global_step = 0

  agent = PPOPixel(config)

  global_step = agent.load_weights('checkpoint/3891200.pth')

  if config.wandb:
    wandb.watch(agent.model)

  while global_step < config.n_steps:
    states = envs.reset()
    score = 0
    values, dones = None, None

    while agent.mem.isFull() == False:
      global_step += config.num_env

      # Take actions
      with torch.no_grad():
        actions, log_probs, values, entrs = agent.act(states)
      next_states, rewards, dones, infos = envs.step(actions)
      # envs.render()

      # Add to memory buffer
      agent.add_to_mem(states, actions, rewards, log_probs, values, dones)
      # Update state
      states = next_states


      # Book Keeping
      for info in infos:
        if 'episode' in info:
          score = info['episode']['r']
          config.tb_logger.add_scalar("charts/episode_reward", score, global_step)
          if config.wandb:
            wandb.log({
              "episode_reward": score,
              "global_step": global_step
            })

          scores_deque.append(score)
          scores.append(score)
          average_scores.append(np.mean(scores_deque))

    # update and learn
    value_loss, pg_loss, approx_kl, approx_entropy, lr_now = agent.learn(config.num_learn, values, dones, global_step)
    agent.mem.reset()

    # Book Keeping
    config.tb_logger.add_scalar("losses/value_loss", value_loss.item(), global_step)
    config.tb_logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    config.tb_logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    config.tb_logger.add_scalar("losses/approx_entropy", approx_entropy.item(), global_step)

    if config.wandb:
      wandb.log({
        "value_loss": value_loss,
        "policy_loss": pg_loss,
        "approx_kl": approx_kl,
        "approx_entropy": approx_entropy,
        "global_step": global_step,
        "learning_rate": lr_now
       })

    if not global_step%204800:
      agent.save_weights(global_step, 'checkpoint/%d.pth'%global_step)


    print("Global Step: {}	Average Score: {:.2f}".format(global_step, np.mean(scores_deque)))

  return scores, average_scores
"""