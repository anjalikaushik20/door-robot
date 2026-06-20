import os
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import CriticNetwork, ActorNetwork

class Agent:
    def __init__(self, actor_learning_rate, critic_learning_rate, input_dims, tau, env, gamma=0.99, update_actor_interval=2, warmup=1000, 
                 n_actions=2, max_size=1000000, layer1_size=256, layer2_size=256, batch_size=256, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_interval = update_actor_interval
        
        #creating the networks
        self.actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                                  name='actor', learning_rate=actor_learning_rate)
        self.critic1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                                    name='critic1', learning_rate=critic_learning_rate)
        self.critic2 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                                    name='critic2', learning_rate=critic_learning_rate)
        
        #target networks
        self.actor_target = ActorNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                                          name='actor_target', learning_rate=actor_learning_rate)
        self.critic1_target = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                                            name='critic1_target', learning_rate=critic_learning_rate)
        self.critic2_target = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                                            name='critic2_target', learning_rate=critic_learning_rate)
        
        self.noise = noise
        self.update_network_parameters(tau=1)
        
    
    def choose_action(self, observation, validation=False):
        if(self.time_step < self.warmup and not validation):
            mu = t.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.actor.device)
        else:
            state = t.tensor(observation, dtype=t.float32).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
            
        mu_prime = mu + t.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.actor.device)
        mu_prime = t.clamp(mu_prime, self.min_action[0], self.max_action[0])
        
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size*10:
            return
        
        state,action,reward,next_state,done = self.memory.sample_buffer(self.batch_size)
        reward = t.tensor(reward, dtype=t.float32).to(self.critic1.device)
        done = t.tensor(done, dtype=t.bool).to(self.critic1.device)
        next_state = t.tensor(next_state, dtype=t.float32).to(self.critic1.device)
        state = t.tensor(state, dtype=t.float32).to(self.critic1.device)
        action = t.tensor(action, dtype=t.float32).to(self.critic1.device)
        
        target_actions = self.actor_target.forward(next_state)
        target_actions = target_actions + t.clamp(t.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = t.clamp(target_actions, self.min_action[0], self.max_action[0])
        
        next_q1 = self.critic1_target.forward(next_state, target_actions)
        next_q2 = self.critic2_target.forward(next_state, target_actions)
        
        q1 = self.critic1.forward(state, action)
        q2 = self.critic2.forward(state, action)
        
        next_q1[done] = 0.0
        next_q2[done] = 0.0
        
        next_q1 = next_q1.view(-1)
        next_q2 = next_q2.view(-1)
        
        next_critic_value = t.min(next_q1, next_q2)
        target = reward + self.gamma * next_critic_value
        target = target.view(self.batch_size, 1)
        
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()
        
        self.learn_step_counter += 1
        
        if self.learn_step_counter % self.update_actor_interval != 0:
            return
        
        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic1.forward(state, self.actor.forward(state))
        actor_loss = -t.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        self.update_network_parameters()
        
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
            
        actor_params = self.actor.named_parameters()
        critic1_params = self.critic1.named_parameters()
        critic2_params = self.critic2.named_parameters()
        target_actor_params = self.actor_target.named_parameters()
        target_critic1_params = self.critic1_target.named_parameters()
        target_critic2_params = self.critic2_target.named_parameters()
        
        actor_state_dict = dict(actor_params)
        critic1_state_dict = dict(critic1_params)
        critic2_state_dict = dict(critic2_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic1_state_dict = dict(target_critic1_params)
        target_critic2_state_dict = dict(target_critic2_params)
        
        for name in critic1_state_dict:
            critic1_state_dict[name] = tau * critic1_state_dict[name].clone() + (1 - tau) * target_critic1_state_dict[name].clone()
            
        for name in critic2_state_dict:
            critic2_state_dict[name] = tau * critic2_state_dict[name].clone() + (1 - tau) * target_critic2_state_dict[name].clone()
            
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[name].clone()
            
        self.critic1_target.load_state_dict(critic1_state_dict)
        self.critic2_target.load_state_dict(critic2_state_dict)
        self.actor_target.load_state_dict(actor_state_dict)

    
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic1_target.save_checkpoint()
        self.critic2_target.save_checkpoint()
        print("Models saved successfully.")
    
    def load_models(self):
        try:
            self.actor.load_checkpoint()
            self.critic1.load_checkpoint()
            self.critic2.load_checkpoint()
            self.actor_target.load_checkpoint()
            self.critic1_target.load_checkpoint()
            self.critic2_target.load_checkpoint()
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
        