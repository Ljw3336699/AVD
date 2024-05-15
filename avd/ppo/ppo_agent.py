import copy
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from models import cnn_net, mlp_net, mlp_inv_net
from policy import select_actions, evaluate_actions
from utils.running_filter.running_filter import ZFilter


class ppo_agent:
    def __init__(self, envs, args):
        self.envs = envs
        self.args = args
        if self.args.env_type == 'atari':
            self.net = cnn_net(envs.action_space.n)
        elif self.args.env_type == 'mujoco':
            self.net = mlp_net(envs.observation_space.shape[0], envs.action_space.shape[0], self.args.dist)
            self.intrinsic_net = mlp_inv_net(envs.observation_space.shape[0])

        self.old_net = copy.deepcopy(self.net)

        if self.args.cuda:
            self.net.cuda()
            self.intrinsic_net.cuda()
            self.old_net.cuda()

        self.optimizer = optim.Adam(self.net.parameters(), self.args.lr, eps=self.args.eps)

        self.intrinsic_optimizer = optim.Adam(self.intrinsic_net.parameters(), self.args.lr_in, eps=self.args.eps)

        if self.args.env_type == 'mujoco':
            num_states = self.envs.observation_space.shape[0]
            self.running_state = ZFilter((num_states,), clip=5)

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.batch_ob_shape = (self.args.num_workers * self.args.nsteps,) + self.envs.observation_space.shape
        self.obs = np.zeros((self.args.num_workers,) + self.envs.observation_space.shape,
                            dtype=self.envs.observation_space.dtype.name)

        if self.args.env_type == 'mujoco':
            self.obs[:] = np.expand_dims(self.running_state(self.envs.reset()), 0)
        else:
            self.obs[:] = self.envs.reset()

        self.dones = [False for _ in range(self.args.num_workers)]

        self.state_optims = None

        if not os.path.exists(self.args.log_data_dir):
            os.mkdir(self.args.log_data_dir)

        self.intrinsic_data_path = '{}/reward_delay_{}'.format(self.args.log_data_dir, self.args.reward_delay_freq)
        if not os.path.exists(self.intrinsic_data_path):
            os.mkdir(self.intrinsic_data_path)

        self.intrinsic_datal_path = '{}/seed_{}'.format(self.intrinsic_data_path, self.args.seed)
        if not os.path.exists(self.intrinsic_data_path):
            os.mkdir(self.intrinsic_data_path)

    def learn(self):
        # 保存日志
        log_data = {}
        num_updates = self.args.epochs // (self.args.nsteps * self.args.num_workers)
        episode_rewards = np.zeros((self.args.num_workers,), dtype=np.float32)
        final_rewards = np.zeros((self.args.num_workers,), dtype=np.float32)

        delay_step = 0
        delay_rewards = 0

        for update in range(num_updates):
            buffer_obs, buffer_rewards_ex, buffer_actions, buffer_dones, buffer_values_mix, buffer_values_in, buffer_obs_ = [], [], [], [], [], [], []

            if self.args.lr_decay:
                self._adjust_learning_rate(update, num_updates)

            for step in range(self.args.nsteps):

                with torch.no_grad():
                    obs_tensor = self._get_tensors(self.obs)
                    v_mix, pis = self.net(obs_tensor)
                    actions = select_actions(pis, self.args.dist, self.args.env_type)

                    _, v_in = self.intrinsic_net(obs_tensor)

                if self.args.env_type == 'atari':
                    input_actions = actions
                else:
                    if self.args.dist == 'gauss':
                        input_actions = actions.copy()
                    elif self.args.dist == 'beta':
                        input_actions = -1 + 2 * actions

                buffer_obs.append(np.copy(self.obs))
                buffer_actions.append(actions)
                buffer_dones.append(self.dones)
                buffer_values_mix.append(v_mix.detach().cpu().numpy().squeeze())
                buffer_values_in.append(v_in.detach().cpu().numpy().squeeze())

                obs_, rewards, dones, _ = self.envs.step(input_actions)
                obs_ = np.expand_dims(self.running_state(obs_), 0)
                buffer_obs_.append(np.copy(obs_))

                delay_step += 1
                delay_rewards += rewards

                if dones or delay_step == self.args.reward_delay_freq:
                    rewards = delay_rewards
                    delay_step, delay_rewards = 0, 0
                else:
                    rewards = 0

                if self.args.env_type == 'mujoco':
                    dones = np.array([dones])
                    rewards = np.array([rewards])

                self.dones = dones
                buffer_rewards_ex.append(rewards)

                self.obs = obs_
                for n, done in enumerate(dones):
                    if done:
                        self.obs[n] = self.obs[n] * 0
                        if self.args.env_type == 'mujoco':
                            obs_ = self.envs.reset()
                            self.obs = np.expand_dims(self.running_state(obs_), 0)

                episode_rewards += rewards
                masks = np.array([0.0 if done_ else 1.0 for done_ in dones], dtype=np.float32)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks

            # debug
            buffer_obs_ = np.asarray(buffer_obs_, dtype=np.float32)
            buffer_obs_ = buffer_obs_.swapaxes(0, 1).reshape(self.batch_ob_shape)

            buffer_obs = np.asarray(buffer_obs, dtype=np.float32)
            buffer_obs = buffer_obs.swapaxes(0, 1).reshape(self.batch_ob_shape)

            buffer_rewards_ex = np.asarray(buffer_rewards_ex, dtype=np.float32)

            buffer_actions = np.asarray(buffer_actions, dtype=np.float32)

            buffer_dones = np.asarray(buffer_dones, dtype=np.bool)

            buffer_values_mix = np.asarray(buffer_values_mix, dtype=np.float32)

            buffer_values_in = np.asarray(buffer_values_in, dtype=np.float32)

            buffer_rewards_in = self.compute_rewards(buffer_obs, buffer_obs_)

            if self.args.env_type == 'mujoco':
                buffer_values_mix = np.expand_dims(buffer_values_mix, 1)
                buffer_values_in = np.expand_dims(buffer_values_in, 1)

            with torch.no_grad():
                obs_tensor = self._get_tensors(self.obs)

                last_values_mix, _ = self.net(obs_tensor)
                last_values_mix = last_values_mix.detach().cpu().numpy().squeeze()

                _, last_values_in = self.intrinsic_net(obs_tensor)

                last_values_in = last_values_in.detach().cpu().numpy().squeeze()

            buffer_values_mix_next = np.zeros_like(buffer_values_mix)
            buffer_values_mix_next[:-1] = buffer_values_mix[1:] * (1.0 - buffer_dones[1:])
            buffer_values_mix_next[-1] = last_values_mix * (1 - self.dones)

            td_mix = self.args.gamma * buffer_values_mix_next - buffer_values_mix

            buffer_advs_mix = np.zeros_like(buffer_rewards_ex)
            buffer_advs_ex = np.zeros_like(buffer_rewards_ex)

            buffer_rewards_mix = self.args.r_ext_coef * buffer_rewards_ex + self.args.r_in_coef * buffer_rewards_in

            lastgaelam_mix, lastgaelam_ex = 0, 0

            for t in reversed(range(self.args.nsteps)):
                if t == self.args.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones
                    nextvalues_mix = last_values_mix
                    nextvalues_in = last_values_in
                else:
                    nextnonterminal = 1.0 - buffer_dones[t + 1]
                    nextvalues_mix = buffer_values_mix[t + 1]
                    nextvalues_in = buffer_values_in[t + 1]

                delta_mix = buffer_rewards_mix[t] + self.args.gamma * nextvalues_mix * nextnonterminal - \
                            buffer_values_mix[t]
                delta_ex = buffer_rewards_ex[t] + self.args.gamma * nextvalues_in * nextnonterminal - buffer_values_in[
                    t]
                buffer_advs_mix[
                    t] = lastgaelam_mix = delta_mix + self.args.gamma * self.args.tau * nextnonterminal * lastgaelam_mix
                buffer_advs_ex[
                    t] = lastgaelam_ex = delta_ex + self.args.gamma * self.args.tau * nextnonterminal * lastgaelam_ex

            buffer_returns_mix = buffer_advs_mix + buffer_values_mix
            buffer_returns_ex = buffer_advs_ex + buffer_values_in

            buffer_rewards_ex = buffer_rewards_ex.swapaxes(0, 1).flatten()
            buffer_rewards_in = buffer_rewards_in.swapaxes(0, 1).flatten()

            td_mix = td_mix.swapaxes(0, 1).flatten()
            buffer_dones = buffer_dones.swapaxes(0, 1).flatten()
            buffer_values_mix = buffer_values_mix.swapaxes(0, 1).flatten()

            self.old_net.load_state_dict(self.net.state_dict())

            pl, vl, ent = self._update_network(buffer_obs, buffer_actions, buffer_returns_mix, buffer_returns_ex,
                                               buffer_advs_mix,
                                               buffer_advs_ex, \
                                               buffer_rewards_in, buffer_rewards_ex, td_mix, buffer_dones,
                                               buffer_values_mix, buffer_obs_)
            if update % self.args.display_interval == 0:
                print('[{}] Update: {} / {}, Episodes: {}, Rewards: {:.3f}, Min: {:.3f}, Max: {:.3f},  PL: {:.3f},' \
                      'VL: {:.3f}, Ent: {:.3f}'.format(datetime.now(), update, num_updates,
                                                       (update + 1) * self.args.nsteps * self.args.num_workers, \
                                                       final_rewards.mean(), final_rewards.min(), final_rewards.max(),
                                                       pl, vl, ent))

    def compute_rewards(self, obs, obs_, requires_grad=False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        obs_next_tensor = torch.tensor(obs_, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        if self.args.metric_type == "avd":
            if not requires_grad:
                with torch.no_grad():
                    feats_in, _ = self.intrinsic_net(obs_tensor)
                    feats_in_next, _ = self.intrinsic_net(obs_next_tensor)
            else:
                feats_in, _ = self.intrinsic_net(obs_tensor)
                feats_in_next, _ = self.intrinsic_net(obs_next_tensor)

            cos_dist = torch.nn.functional.cosine_similarity(feats_in, feats_in_next, dim=1)
            rewards_in = 1 - cos_dist.pow(2)
        return rewards_in.unsqueeze(-1) if requires_grad else rewards_in.unsqueeze(-1).detach().cpu().numpy()

    def _update_network(self, temp_obs, temp_actions, temp_returns_mix, temp_returns_ex, temp_advs_mix, temp_advs_ex,
                        temp_rewards_in,
                        temp_rewards_ex, td_mix, temp_dones, temp_values_mix, temp_obs_):

        inds = np.arange(temp_obs.shape[0])
        nbatch_train = temp_obs.shape[0] // self.args.batch_size
        for _ in range(self.args.train_epoch):
            np.random.shuffle(inds)
            for start in range(0, temp_obs.shape[0], nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                coef_mat = np.zeros((nbatch_train, temp_obs.shape[0]), dtype=np.float32)

                for i in range(nbatch_train):
                    coef = 1.0
                    for j in range(mbinds[i], temp_obs.shape[0]):
                        if j > mbinds[i] and (temp_dones[j] or j % self.args.nsteps == 0):
                            break
                        coef_mat[i][j] = coef
                        coef *= self.args.gamma * self.args.tau

                r_in_tensor = self.compute_rewards(temp_obs, temp_obs_, requires_grad=True)
                r_ex_tensor = torch.tensor(temp_rewards_ex, dtype=torch.float32,
                                           device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
                td_mix_tensor = torch.tensor(td_mix, dtype=torch.float32,
                                             device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
                coef_mat_tensor = torch.tensor(coef_mat, dtype=torch.float32,
                                               device='cuda' if self.args.cuda else 'cpu')

                delta_mix = self.args.r_in_coef * r_in_tensor + self.args.r_ext_coef * r_ex_tensor + td_mix_tensor
                adv_mix = torch.matmul(coef_mat_tensor, delta_mix)

                buffer_obs = temp_obs[mbinds]
                buffer_obs_ = temp_obs_[mbinds]
                buffer_actions = temp_actions[mbinds]
                buffer_values_mix = temp_values_mix[mbinds]
                buffer_advs_ex = temp_advs_ex[mbinds]
                buffer_returns_ex = temp_returns_ex[mbinds]
                buffer_obs = self._get_tensors(buffer_obs)
                buffer_obs_ = self._get_tensors(buffer_obs_)

                buffer_actions = torch.tensor(buffer_actions, dtype=torch.float32,
                                              device='cuda' if self.args.cuda else 'cpu')
                buffer_values_mix = torch.tensor(buffer_values_mix, dtype=torch.float32,
                                                 device='cuda' if self.args.cuda else 'cpu').unsqueeze(1)
                buffer_advs_ex = torch.tensor(buffer_advs_ex, dtype=torch.float32,
                                              device='cuda' if self.args.cuda else 'cpu').unsqueeze(1)
                buffer_returns_ex = torch.tensor(buffer_returns_ex, dtype=torch.float32,
                                                 device='cuda' if self.args.cuda else 'cpu').unsqueeze(1)

                temp_returns_mix = adv_mix + buffer_values_mix
                adv_mix = (adv_mix - adv_mix.mean().detach()) / (adv_mix.std().detach() + 1e-8)

                values, pis = self.net(buffer_obs)
                value_loss = (temp_returns_mix - values).pow(2).mean()
                with torch.no_grad():
                    _, old_pis = self.old_net(buffer_obs)
                    old_log_prob, _ = evaluate_actions(old_pis, buffer_actions, self.args.dist, self.args.env_type)
                    old_log_prob = old_log_prob.detach()

                log_prob, ent_loss = evaluate_actions(pis, buffer_actions, self.args.dist, self.args.env_type)
                prob_ratio = torch.exp(log_prob - old_log_prob)
                surr1 = prob_ratio * adv_mix
                surr2 = torch.clamp(prob_ratio, 1 - self.args.clip, 1 + self.args.clip) * adv_mix

                policy_loss = -torch.min(surr1, surr2).mean()
                total_loss = policy_loss + self.args.vloss_coef * value_loss - ent_loss * self.args.ent_coef

                self.optimizer.zero_grad()
                self.intrinsic_optimizer.zero_grad()

                grads = torch.autograd.grad(total_loss, self.net.parameters(), create_graph=True)

                net_new = copy.deepcopy(self.net)

                for (_, param), grad in zip(self.net.named_parameters(), grads):
                    if param.grad is None:
                        param.grad = torch.zeros_like(param.data)
                        param.grad.data.copy_(grad.data)
                    else:
                        param.grad.data.copy_(grad.data)

                self.optimizer.step()

                if self.state_optims is None:
                    self.state_optims = self.optimizer.state_dict()['state'].values()
                    self.init_optim = True

                beta1, beta2 = 0.9, 0.999
                for (_, param), grad, state_optim in zip(net_new.named_parameters(), grads, self.state_optims):
                    if self.init_optim:
                        exp_avg = torch.zeros_like(param)
                        exp_avg_sq = torch.zeros_like(param)
                    else:
                        exp_avg = state_optim['exp_avg'].clone()
                        exp_avg_sq = state_optim['exp_avg_sq'].clone()
                    bias_corr1 = 1 - beta1 ** state_optim['step']
                    bias_corr2 = 1 - beta2 ** state_optim['step']
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad.detach(), grad.detach())
                    step_size = self.args.lr / bias_corr1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(self.args.eps)
                    param.requires_grad = False
                    param.addcdiv_(-step_size, exp_avg, denom)

                self.init_optim = False
                self.state_optims = self.optimizer.state_dict()['state'].values()

                # 计算奖励函数误差
                buffer_advs_ex = (buffer_advs_ex - buffer_advs_ex.mean()) / (buffer_advs_ex.std() + 1e-8)
                _, pis_new = net_new(buffer_obs)
                new_log_prob, _ = evaluate_actions(pis_new, buffer_actions, self.args.dist, self.args.env_type)
                ratio_new = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio_new * buffer_advs_ex
                surr2 = torch.clamp(ratio_new, 1 - self.args.clip, 1 + self.args.clip) * buffer_advs_ex
                in_policy_loss = -torch.min(surr1, surr2).mean()
                _, buffer_values_in = self.intrinsic_net(buffer_obs)
                in_value_loss = (buffer_returns_ex - buffer_values_in).pow(2).mean()

                v_new, _ = net_new(buffer_obs)
                v_old, _ = self.old_net(buffer_obs)

                td = (buffer_values_in - (self.args.alpha * v_new + (1 - self.args.alpha) * v_old)).pow(2).mean()

                in_total_loss = in_policy_loss + self.args.vloss_coef * in_value_loss + td

                eta_grads = torch.autograd.grad(in_total_loss.clone(), self.intrinsic_net.parameters())
                for param, grad in zip(self.intrinsic_net.parameters(), eta_grads):
                    if param.grad is None:
                        param.grad = torch.zeros_like(param.data)
                        param.grad.data.copy_(grad.data)
                    else:
                        param.grad.data.copy_(grad.data)
                self.intrinsic_optimizer.step()

        return policy_loss.item(), value_loss.item(), ent_loss.item()

    def _get_tensors(self, obs):
        if self.args.env_type == 'atari':
            obs_tensor = torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32,
                                      device='cuda' if self.args.cuda else 'cpu')
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        return obs_tensor

    def _adjust_learning_rate(self, update, num_updates):
        lr_frac = 1 - (update / num_updates)
        adjust_lr = self.args.lr * lr_frac
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adjust_lr
