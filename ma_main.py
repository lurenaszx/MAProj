from env.make_env import make_env
import argparse, datetime
from tensorboardX import SummaryWriter
import numpy as np
import torch,os

from algo.bicnet.bicnet_agent import BiCNet
from algo.commnet.commnet_agent import CommNet
from algo.maddpg.maddpg_agent import MADDPG

from algo.normalized_env import ActionNormalizedEnv, ObsEnv, reward_from_state

from algo.utils import *
from copy import deepcopy


def main(args):

    env = make_env(args.scenario)
    env.reset()
    n_agents = env.num_agents
    n_actions = 5
    # env = ActionNormalizedEnv(env)
    # env = ObsEnv(env)
    # print(env.observation_spaces[0])
    n_states = 18

    torch.manual_seed(args.seed)

    if args.tensorboard and args.mode == "train":
        writer = SummaryWriter(log_dir='runs/' + args.algo + "/" + args.log_dir)

    if args.algo == "bicnet":
        model = BiCNet(n_states, n_actions, n_agents, args)

    if args.algo == "commnet":
        model = CommNet(n_states, n_actions, n_agents, args)

    if args.algo == "maddpg":
        model = MADDPG(n_states, n_actions, n_agents, args)

    print(model)
    model.load_model()

    episode = 0
    total_step = 0

    while episode < args.max_episodes:

        state = env.reset()

        episode += 1
        step = 0
        accum_reward = 0
        rewardA = 0
        rewardB = 0
        rewardC = 0
        while True:
            if args.mode == "train":
                observations, infos = env.reset()
                while env.agents:
                    actions = model.choose_action(observations, noisy=True)
                    next_observations, rewards, terminations, truncations, infos = env.step(actions)

                    step += 1
                    total_step += 1
                    rewards = np.array(rewards)
                    if True in truncations:
                        break
                    rew1 = reward_from_state(next_observations)
                    rewards = rew1 + (np.array(rewards, dtype=np.float32) / 100.)
                    accum_reward += sum(rewards)
                    # print(rewards)
                    rewardA += rewards[0]
                    rewardB += rewards[1]
                    rewardC += rewards[2]


                    if args.algo == "maddpg" or args.algo == "commnet":
                        obs = torch.from_numpy(np.stack(observations)).float().to(device)
                        obs_ = torch.from_numpy(np.stack(next_observations)).float().to(device)
                        if step != args.episode_length - 1:
                            next_obs = obs_
                        else:
                            next_obs = None
                        rw_tensor = torch.FloatTensor(rewards).to(device)
                        ac_tensor = torch.FloatTensor(actions.float()).to(device)
                        if args.algo == "commnet" and next_obs is not None:
                            model.memory.push(obs.data, ac_tensor, next_obs, rw_tensor)
                        if args.algo == "maddpg":
                            model.memory.push(obs.data, ac_tensor, next_obs, rw_tensor)
                        obs = next_obs
                    else:
                        model.memory(observations, actions, rewards, next_observations, terminations)

                    observations = next_observations

                if args.episode_length < step or (True in terminations):
                    c_loss, a_loss = model.update(episode)

                    print("[Episode %05d] reward %6.4f" % (episode, accum_reward))
                    if args.tensorboard:
                        writer.add_scalar(tag='agent/reward', global_step=episode, scalar_value=accum_reward.item())
                        writer.add_scalar(tag='agent/reward_0', global_step=episode, scalar_value=rewardA.item())
                        writer.add_scalar(tag='agent/reward_1', global_step=episode, scalar_value=rewardB.item())
                        writer.add_scalar(tag='agent/reward_2', global_step=episode, scalar_value=rewardC.item())
                        if c_loss and a_loss:
                            writer.add_scalars('agent/loss', global_step=episode,
                                               tag_scalar_dict={'actor': a_loss, 'critic': c_loss})

                    if c_loss and a_loss:
                        print(" a_loss %3.2f c_loss %3.2f" % (a_loss, c_loss), end='')


                    if episode % args.save_interval == 0 and args.mode == "train":
                        model.save_model(episode)

                    env.reset()
                    # model.reset()
                    break
            elif args.mode == "eval":
                observations, infos = env.reset()
                while env.agents:
                    action = model.choose_action(observations, noisy=False)
                    next_observations, rewards, terminations, truncations, infos = env.step(action)
                    step += 1
                    total_step += 1
                    observations = next_observations
                    rewards = np.array(rewards)
                    import time
                    time.sleep(0.02)
                    env.render()
                    if True in truncations:
                        break
                    rew1 = reward_from_state(observations)
                    rewards = rew1 + (np.array(rewards, dtype=np.float32) / 100.)
                    accum_reward += sum(rewards)
                    rewardA += rewards[0]
                    rewardB += rewards[1]
                    rewardC += rewards[2]

                if args.episode_length < step or (True in terminations):
                    print("[Episode %05d] reward %6.4f " % (episode, accum_reward))
                    env.reset()
                    break

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="simple_spread", type=str)
    parser.add_argument('--max_episodes', default=1e10, type=int)
    parser.add_argument('--algo', default="commnet", type=str, help="commnet/bicnet/maddpg")
    parser.add_argument('--mode', default="train", type=str, help="train/eval")
    parser.add_argument('--episode_length', default=50, type=int)
    parser.add_argument('--memory_length', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--render_flag', default=False, type=bool)
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--epsilon_decay', default=10000, type=int)
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--model_episode", default=100, type=int)
    parser.add_argument('--episode_before_train', default=1000, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    args = parser.parse_args()
    main(args)
