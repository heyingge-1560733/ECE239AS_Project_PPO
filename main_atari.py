import gym
from Config import Config
from util import train_pixel
from Models import ActorCritic
from Memory import Memory
import wandb
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    wandb.init(project="PPO")
    
    env_id = "BreakoutNoFrameskip-v4"
    config = Config(env_id)

    config.update_every = 128
    config.num_learn = 8
    config.win_condition = 230
    config.n_steps = 5e7
    config.hidden_size = 512
    config.lr = 1e-4
    config.lr_annealing = True
    config.epsilon_annealing = True
    config.model_path = 'checkpoint/any.pth'

    config.memory = Memory
    config.model = ActorCritic

    config.init_wandb()
    scores = train_pixel(config)