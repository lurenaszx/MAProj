import torch.nn as nn
from algo.memory import ReplayMemory, Experience



import torch, os
import numpy as np, random

from algo.random_process import OrnsteinUhlenbeckProcess
from algo.utils import soft_update, hard_update, device
from algo.commnet.network import CommNetWork_Actor, CommNetWork_Critic

class CommNet():

    def __init__(self, s_dim, a_dim, n_agents, args):
        self.cuda = torch.cuda.is_available()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.config = args
        self.batch_size = self.config.batch_size
        self.n_agents = n_agents
        self.device = device
        # Networks
        self.actor = CommNetWork_Actor(s_dim,self.config.batch_size,a_dim)#Actor(s_dim, a_dim, n_agents)
        self.actor_target = CommNetWork_Actor(s_dim,self.config.batch_size,a_dim)#Actor(s_dim, a_dim, n_agents)
        self.critic = CommNetWork_Critic(s_dim,self.config.batch_size,a_dim)#Actor(s_dim, a_dim, n_agents)
        self.critic_target = CommNetWork_Critic(s_dim,self.config.batch_size,a_dim)#Actor(s_dim, a_dim, n_agents)

        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.a_lr)

        self.random_process = OrnsteinUhlenbeckProcess(size=self.a_dim,
                                                       theta=self.config.ou_theta,
                                                       mu=self.config.ou_mu,
                                                       sigma=self.config.ou_sigma)
        self.replay_buffer = list()
        self.epsilon = 1.
        self.depsilon = self.epsilon / self.config.epsilon_decay

        self.c_loss = None
        self.a_loss = None
        self.action_log = list()
        self.memory = ReplayMemory(1e5)
        self.GAMMA = self.config.gamma
        self.var = [1.0 for i in range(n_agents)]
        self.episode = 0

    def load_model(self):
        model_actor_path = "./trained_model/" + str(self.config.algo) + "/actor_" + str(self.config.model_episode) + ".pth"
        model_critic_path = "./trained_model/" + str(self.config.algo) + "/critic_" + str(self.config.model_episode) + ".pth"
        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            print("load model!")
            actor = torch.load(model_actor_path)
            critic = torch.load(model_critic_path)
            self.actor.load_state_dict(actor)
            self.critic.load_state_dict(critic)

    def save_model(self, episode):
        if not os.path.exists("./trained_model/" + str(self.config.algo) + "/"):
            os.mkdir("./trained_model/" + str(self.config.algo) + "/")
        torch.save(self.actor.state_dict(),
                   "./trained_model/" + str(self.config.algo) + "/actor_" + str(episode) + ".pth"),
        torch.save(self.critic.state_dict(),
                   "./trained_model/" + str(self.config.algo) + "/critic_" + str(episode) + ".pth"),

    def choose_action(self, obs, noisy=True):
        obs = torch.Tensor([obs]).to(self.device)

        distributions = self.actor(obs)
        action = distributions.sample()
        self.action_log.append(action)
        # print(action)
        return action

    def reset(self):
        self.random_process.reset_states()
        self.action_log.clear()

    def prep_train(self):
        self.actor.train()
        self.critic.train()

    def prep_eval(self):
        self.actor.eval()
        self.critic.eval()

    def random_action(self):
        return np.random.uniform(low=-1, high=1, size=(self.n_agents, 2))

    def get_batches(self):
        experiences = random.sample(self.replay_buffer, self.config.batch_size)

        state_batches = np.array([_[0] for _ in experiences])
        action_batches = np.array([_[1] for _ in experiences])
        reward_batches = np.array([_[2] for _ in experiences])
        next_state_batches = np.array([_[3] for _ in experiences])
        done_batches = np.array([_[4] for _ in experiences])


        return state_batches, action_batches, reward_batches, next_state_batches, done_batches

    def update(self, i_episode):

        if len(self.memory.memory) < self.batch_size:
            return None, None

        self.use_cuda = torch.cuda.is_available()

        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []

        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        state_batch = torch.stack(batch.states).type(FloatTensor)
        action_batch = torch.stack(batch.actions).type(FloatTensor)
        reward_batch = torch.stack(batch.rewards).type(FloatTensor)
        non_final_next_states = torch.stack(batch.next_states).type(FloatTensor)#torch.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)

        c_loss.append(self.update_critic(state_batch, action_batch, reward_batch, non_final_next_states))
        a_loss.append(self.update_actor(state_batch))

        self.train_num = i_episode

        if self.train_num % 200 == 0:
            soft_update(self.actor, self.actor_target, self.config.tau)
            soft_update(self.critic, self.critic_target, self.config.tau)

        return sum(c_loss) / len(c_loss), sum(a_loss) / len(a_loss)

    def update_critic(self, states, actions, rewards, next_states):
        self.critic_optimizer.zero_grad()
        self.critic.zero_grad()
        whole_state = states.view(self.batch_size*self.n_agents, -1)
        whole_action = actions.view(self.batch_size*self.n_agents, -1)
        next_whole_batch = self.actor_target(next_states).sample().view(self.batch_size*self.n_agents, -1)
        current_Q = self.critic(whole_state).gather(1, whole_action.long()).view(-1, self.n_agents)
        if self.cuda:
            current_Q = current_Q.cuda()
        target_Q = self.critic_target(next_states).gather(1, next_whole_batch.long()).view(-1, self.n_agents)
        if self.cuda:
            target_Q = target_Q.cuda()
        target_Q = target_Q * self.GAMMA + rewards
        loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
        loss_Q.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()
        return loss_Q.item()

    def update_actor(self, states):
        whole_state = states.view(self.batch_size*self.n_agents, -1)
        self.actor_optimizer.zero_grad()
        self.actor.zero_grad()
        whole_distributions = self.actor(whole_state)
        whole_action = whole_distributions.sample().view(-1, 1)
        whole_logits = whole_distributions.logits.gather(1, whole_action.long())
        critic_value = self.critic(whole_state).gather(1, whole_action.long()).detach().view(-1, 1)
        actor_loss = -(whole_logits*critic_value).mean()*100
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.actor_optimizer.step()
        return actor_loss.item()

    def get_loss(self):
        return self.c_loss, self.a_loss

    def get_action_std(self):
        return np.array(self.action_log).std(axis=-1).mean()
