# rl_education_suite.py
# Research scaffold for adaptive education RL experiments
# Implements:
# - EduEnv (simulator)
# - DQN + Prioritized Experience Replay (PER)
# - PPO (discrete)
# - PETS (ensemble dynamics + CEM planner)
# - MBPO (dynamics + short rollouts augmenting DQN)
# - Runner + Plots (learning curves, time-to-mastery, variance bands, compute vs reward)

import os
import time
import math
import random
import argparse
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple("Transition", ("s", "a", "r", "s2", "done"))

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

class EduEnv:
    def __init__(self, K=8, max_steps=100, beta=0.5, gamma=0.05, step_cost=0.01, seed=0):
        self.K = K
        self.max_steps = max_steps
        self.beta = beta
        self.gamma = gamma
        self.step_cost = step_cost
        self.rng = np.random.RandomState(seed)
        self.a = {"E": 0.8, "M": 1.2, "H": 1.6}
        self.b = {"E": -0.5, "M": 0.0, "H": 0.6}
        self.ce = {"video": 0.06, "ppt": 0.05, "text": 0.04, "handout": 0.045}
        self.reset()

    def reset(self):
        self.mastery = np.clip(self.rng.beta(2, 5, size=self.K), 0.0, 1.0)
        self.fatigue = float(self.rng.uniform(0.05, 0.2))
        self.tau = float(self.rng.uniform(0.2, 0.6))
        self.fail_streak = 0
        self.t = 0
        self.done = False
        self.time_to_mastery = None
        return self._state()

    def _state(self):
        fs = min(self.fail_streak / 5.0, 1.0)
        return np.concatenate([self.mastery, [self.fatigue, self.tau, fs]]).astype(np.float32)

    def _avg_mastery(self):
        return float(self.mastery.mean())

    def step(self, a: int):
        if self.done:
            raise RuntimeError("Step called on terminated episode.")
        self.t += 1
        rew = 0.0
        mastery_gain = 0.0

        if a in [0,1,2]:
            diff = ["E","M","H"][a]
            theta = self._avg_mastery() * 2 - 1
            prob = 1.0 / (1.0 + math.exp(- self.a[diff] * (theta - self.b[diff])))
            correct = 1 if self.rng.rand() < prob else 0
            if correct:
                self.fail_streak = 0
                idx = self.rng.randint(0, self.K)
                inc = self.rng.uniform(0.01, 0.03) * (1.0 - self.mastery[idx])
                self.mastery[idx] = min(1.0, self.mastery[idx] + inc)
                mastery_gain = inc
                rew += 1.0
            else:
                self.fail_streak += 1
                self.fatigue = min(1.0, self.fatigue + 0.02)
                rew -= 0.05
            self.tau = float(np.clip(self.tau + self.rng.normal(0, 0.02) + (0.01 if not correct else -0.01), 0.0, 1.0))
        else:
            modality = ["video","ppt","text","handout"][a-3]
            base = self.ce[modality]
            gains = (1.0 - self.mastery) * base * self.rng.uniform(0.8, 1.2, size=self.K)
            self.mastery = np.clip(self.mastery + gains, 0.0, 1.0)
            mastery_gain = float(gains.mean())
            self.fatigue = float(np.clip(self.fatigue - 0.03, 0.0, 1.0))
            self.tau = float(np.clip(self.tau - 0.02, 0.0, 1.0))
            self.fail_streak = 0
            rew += 0.2

        rew += self.beta * mastery_gain - self.gamma * self.fatigue - self.step_cost

        avg_m = self._avg_mastery()
        if avg_m >= 0.95 and self.time_to_mastery is None:
            self.time_to_mastery = self.t
        if self.t >= self.max_steps or avg_m >= 0.98:
            self.done = True

        return self._state(), float(rew), self.done, {"avg_mastery": avg_m, "time_to_mastery": self.time_to_mastery}

    @property
    def obs_dim(self):
        return self.K + 3

    @property
    def n_actions(self):
        return 7

class PERBuffer:
    def __init__(self, cap=100000, alpha=0.6, beta_start=0.4, beta_frames=200000):
        self.cap = cap
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((cap,), dtype=np.float32)
        self.frame = 1

    def push(self, *args):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.cap:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.pos] = Transition(*args)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.cap

    def beta_by_frame(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    def sample(self, batch_size):
        assert len(self.buffer) > 0
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        beta = self.beta_by_frame()
        self.frame += 1
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        states = torch.tensor(np.array([s.s for s in samples]), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array([s.a for s in samples]), dtype=torch.long, device=device)
        rewards = torch.tensor(np.array([s.r for s in samples]), dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array([s.s2 for s in samples]), dtype=torch.float32, device=device)
        dones = torch.tensor(np.array([s.done for s in samples]), dtype=torch.float32, device=device)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, prios):
        for idx, pr in zip(indices, prios):
            self.priorities[idx] = float(pr)

    def __len__(self):
        return len(self.buffer)

class MLP(nn.Module):
    def __init__(self, nin, nout, hidden=(128,128), act=nn.ReLU):
        super().__init__()
        layers = []
        last = nin
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers += [nn.Linear(last, nout)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class QNet(nn.Module):
    def __init__(self, nin, nact):
        super().__init__()
        self.body = MLP(nin, 128, hidden=(256,128))
        self.head = nn.Linear(128, nact)
    def forward(self, x):
        z = self.body(x)
        return self.head(z)

class PolicyNet(nn.Module):
    def __init__(self, nin, nact):
        super().__init__()
        self.body = MLP(nin, 128, hidden=(128,128))
        self.logits = nn.Linear(128, nact)
        self.value = nn.Linear(128, 1)
    def forward(self, x):
        z = self.body(x)
        return self.logits(z), self.value(z)

class DynNet(nn.Module):
    def __init__(self, nin, nact, nout):
        super().__init__()
        self.body = MLP(nin + nact, 128, hidden=(128,128))
        self.mu = nn.Linear(128, nout)
        self.logvar = nn.Linear(128, nout)
    def forward(self, s, a_onehot):
        x = torch.cat([s, a_onehot], dim=-1)
        z = self.body(x)
        return self.mu(z), self.logvar(z)

def train_dqn_per(env_ctor, steps=30000, seed=0, gamma=0.99, lr=3e-4, batch=64, start_learning=1000,
                  eps_init=1.0, eps_final=0.05, eps_decay=20000, target_sync=1000):
    set_seed(seed)
    env = env_ctor(seed=seed)
    q = QNet(env.obs_dim, env.n_actions).to(device)
    qt = QNet(env.obs_dim, env.n_actions).to(device)
    qt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=lr)
    buf = PERBuffer(cap=100000)

    s = env.reset()
    ep_ret, ep_rets = 0.0, []
    rewards_curve = []
    ttm_list = []
    t_start = time.time()

    def epsilon(tstep):
        return eps_final + (eps_init - eps_final) * math.exp(-tstep / eps_decay)

    for t in range(steps):
        eps = epsilon(t)
        if np.random.rand() < eps:
            a = np.random.randint(env.n_actions)
        else:
            with torch.no_grad():
                qs = q(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0))
                a = int(qs.argmax(dim=-1).item())

        s2, r, done, info = env.step(a)
        buf.push(s, a, r, s2, float(done))
        s = s2
        ep_ret += r
        rewards_curve.append(ep_ret)
        if info.get("time_to_mastery") is not None and info["time_to_mastery"] not in ttm_list:
            ttm_list.append(info["time_to_mastery"])

        if done:
            ep_rets.append(ep_ret)
            s = env.reset()
            ep_ret = 0.0

        if len(buf) > start_learning:
            states, actions, rewards_b, next_states, dones, idxs, weights = buf.sample(batch)
            with torch.no_grad():
                q_next = qt(next_states).max(dim=1)[0]
                target = rewards_b + gamma * (1.0 - dones) * q_next
            q_pred = q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            td = target - q_pred
            loss = (weights * td.pow(2)).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0)
            opt.step()

            prios = td.abs().detach().cpu().numpy() + 1e-5
            buf.update_priorities(idxs, prios)

        if t % target_sync == 0 and t > 0:
            qt.load_state_dict(q.state_dict())

    wall = time.time() - t_start
    return {"learning_curve": rewards_curve,
            "ep_returns": ep_rets,
            "time_to_mastery": np.array(ttm_list) if len(ttm_list) > 0 else np.array([np.nan]),
            "wall_clock": wall}

def discount_cumsum(x, discount):
    y = np.zeros_like(x, dtype=np.float32)
    run = 0.0
    for i in reversed(range(len(x))):
        run = x[i] + discount * run
        y[i] = run
    return y

def ppo_iterate(policy, optimizer, states, actions, returns, advantages,
                clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, epochs=4, minibatch=64):
    N = states.size(0)
    with torch.no_grad():
        logits_old, _ = policy(states)
        dist_old = torch.distributions.Categorical(logits=logits_old)
        logp_old_all = dist_old.log_prob(actions)
    for _ in range(epochs):
        idx = torch.randperm(N, device=device)
        for start in range(0, N, minibatch):
            end = min(start + minibatch, N)
            mb = idx[start:end]
            s = states[mb]; a = actions[mb]; ret = returns[mb]; adv = advantages[mb]
            logp_old = logp_old_all[mb]

            logits, v = policy(s)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(a)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - logp_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            pi_loss = -torch.min(surr1, surr2).mean()
            vf_loss = F.mse_loss(v.squeeze(-1), ret)
            loss = pi_loss + vf_coef * vf_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()

def train_ppo(env_ctor, steps=30000, seed=0, gamma=0.99, lam=0.95, lr=3e-4, horizon=2048):
    set_seed(seed)
    env = env_ctor(seed=seed)
    policy = PolicyNet(env.obs_dim, env.n_actions).to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)

    s = env.reset()
    ep_ret, ep_rets = 0.0, []
    rewards_curve, ttm_list = [], []
    t = 0
    t_start = time.time()

    buf_s, buf_a, buf_r, buf_v, buf_d = [], [], [], [], []

    while t < steps:
        for _ in range(horizon):
            with torch.no_grad():
                logits, v = policy(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits)
                a = int(dist.sample().item())
                v_item = float(v.item())
            s2, r, done, info = env.step(a)

            buf_s.append(s); buf_a.append(a); buf_r.append(r); buf_v.append(v_item); buf_d.append(done)

            s = s2
            ep_ret += r
            rewards_curve.append(ep_ret)
            t += 1
            if info.get("time_to_mastery") is not None and info["time_to_mastery"] not in ttm_list:
                ttm_list.append(info["time_to_mastery"])
            if done:
                s = env.reset()
                ep_rets.append(ep_ret)
                ep_ret = 0.0
            if t >= steps: break

        with torch.no_grad():
            _, last_v_t = policy(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0))
            last_v = float(last_v_t.item())

        rews = np.array(buf_r + [last_v], dtype=np.float32)
        vals = np.array(buf_v + [last_v], dtype=np.float32)
        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
        adv = discount_cumsum(deltas, gamma * lam)
        ret = discount_cumsum(rews, gamma)[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states = torch.tensor(np.array(buf_s), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(buf_a), dtype=torch.long, device=device)
        returns = torch.tensor(ret, dtype=torch.float32, device=device)
        advantages = torch.tensor(adv, dtype=torch.float32, device=device)

        ppo_iterate(policy, opt, states, actions, returns, advantages,
                    clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, epochs=4, minibatch=128)

        buf_s, buf_a, buf_r, buf_v, buf_d = [], [], [], [], []

    wall = time.time() - t_start
    return {"learning_curve": rewards_curve,
            "ep_returns": ep_rets,
            "time_to_mastery": np.array(ttm_list) if len(ttm_list) > 0 else np.array([np.nan]),
            "wall_clock": wall}

def cem_plan(state_tensor, dyn_ensemble, n_actions, horizon=5, pop=200, elites=20, iters=3):
    S = state_tensor.repeat(pop, 1)
    logits = [torch.zeros((pop, n_actions), device=device) for _ in range(horizon)]

    def sample_actions(logits_list):
        acts = []
        for lg in logits_list:
            dist = torch.distributions.Categorical(logits=lg)
            a = dist.sample()
            acts.append(a)
        return torch.stack(acts, dim=1)

    def rollout(actions):
        total_r = torch.zeros((pop,), device=device)
        s = S.clone()
        for t in range(horizon):
            a = actions[:, t]
            a1 = F.one_hot(a, n_actions).float()
            r_acc = torch.zeros((pop,), device=device)
            s_acc = torch.zeros_like(s)
            for dnet in dyn_ensemble:
                mu, logv = dnet(s, a1)
                eps = torch.randn_like(mu)
                delta = mu + torch.exp(0.5 * logv) * eps
                ds = delta[:, :s.size(1)]
                rr = delta[:, s.size(1):s.size(1)+1].squeeze(-1)
                s_acc += torch.clamp(s + ds, 0.0, 1.0)
                r_acc += rr
            s = s_acc / len(dyn_ensemble)
            total_r += r_acc / len(dyn_ensemble)
        return total_r

    for _ in range(iters):
        acts = sample_actions(logits)
        scores = rollout(acts)
        elite_idx = torch.topk(scores, elites).indices
        for t in range(horizon):
            elite_actions_t = acts[elite_idx, t]
            counts = torch.bincount(elite_actions_t, minlength=n_actions).float()
            probs = (counts + 1.0) / (counts.sum() + n_actions)
            logits[t] = torch.log(probs + 1e-8).unsqueeze(0).repeat(pop, 1)

    acts = sample_actions(logits)
    scores = rollout(acts)
    best = int(torch.argmax(scores).item())
    return int(acts[best, 0].item())

def train_pets(env_ctor, steps=30000, seed=0, ensemble=5, horizon=5, lr=1e-3, batch=128, train_every=50):
    set_seed(seed)
    env = env_ctor(seed=seed)
    s_dim = env.obs_dim
    nA = env.n_actions
    nets = [DynNet(s_dim, nA, s_dim + 1).to(device) for _ in range(ensemble)]
    opts = [optim.Adam(n.parameters(), lr=lr) for n in nets]
    buf = deque(maxlen=50000)

    s = env.reset()
    ep_ret, ep_rets = 0.0, []
    rewards_curve, ttm_list = [], []
    t_start = time.time()

    for t in range(steps):
        with torch.no_grad():
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            a = cem_plan(st, nets, nA, horizon=horizon)
        s2, r, done, info = env.step(a)
        buf.append(Transition(s, a, r, s2, float(done)))
        s = s2
        ep_ret += r
        rewards_curve.append(ep_ret)
        if info.get("time_to_mastery") is not None and info["time_to_mastery"] not in ttm_list:
            ttm_list.append(info["time_to_mastery"])
        if done:
            ep_rets.append(ep_ret)
            s = env.reset()
            ep_ret = 0.0

        if t % train_every == 0 and len(buf) >= batch:
            idx = np.random.choice(len(buf), batch, replace=False)
            B = [buf[i] for i in idx]
            S = torch.tensor(np.array([b.s for b in B]), dtype=torch.float32, device=device)
            A = torch.tensor(np.array([b.a for b in B]), dtype=torch.long, device=device)
            A1 = F.one_hot(A, nA).float()
            S2 = torch.tensor(np.array([b.s2 for b in B]), dtype=torch.float32, device=device)
            R = torch.tensor(np.array([b.r for b in B]), dtype=torch.float32, device=device).unsqueeze(-1)
            target = torch.cat([S2 - S, R], dim=-1)
            for net, opt in zip(nets, opts):
                mu, logv = net(S, A1)
                inv = torch.exp(-logv)
                loss = ((mu - target).pow(2) * inv + logv).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()

    wall = time.time() - t_start
    return {"learning_curve": rewards_curve,
            "ep_returns": ep_rets,
            "time_to_mastery": np.array(ttm_list) if len(ttm_list) > 0 else np.array([np.nan]),
            "wall_clock": wall}

def train_mbpo(env_ctor, steps=30000, seed=0, gamma=0.99, lr=3e-4, batch=64, start_learning=1000,
               target_sync=1000, model_lr=1e-3, rollout_len=3, train_every=50, ensemble=5):
    set_seed(seed)
    env = env_ctor(seed=seed)
    q = QNet(env.obs_dim, env.n_actions).to(device)
    qt = QNet(env.obs_dim, env.n_actions).to(device)
    qt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=lr)
    buf_real = PERBuffer(cap=100000)
    buf_model = PERBuffer(cap=200000)

    dyns = [DynNet(env.obs_dim, env.n_actions, env.obs_dim + 1).to(device) for _ in range(ensemble)]
    dyn_opts = [optim.Adam(d.parameters(), lr=model_lr) for d in dyns]

    s = env.reset()
    ep_ret, ep_rets = 0.0, []
    rewards_curve, ttm_list = [], []
    t_start = time.time()

    for t in range(steps):
        eps = max(0.05, 1.0 - t / 20000.0)
        if np.random.rand() < eps:
            a = np.random.randint(env.n_actions)
        else:
            with torch.no_grad():
                qs = q(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0))
                a = int(qs.argmax(dim=-1).item())

        s2, r, done, info = env.step(a)
        buf_real.push(s, a, r, s2, float(done))
        s = s2
        ep_ret += r
        rewards_curve.append(ep_ret)
        if info.get("time_to_mastery") is not None and info["time_to_mastery"] not in ttm_list:
            ttm_list.append(info["time_to_mastery"])
        if done:
            ep_rets.append(ep_ret)
            s = env.reset()
            ep_ret = 0.0

        if t % train_every == 0 and len(buf_real) >= batch:
            states, actions, rewards_b, next_states, dones, idxs, weights = buf_real.sample(batch)
            A1 = F.one_hot(actions, env.n_actions).float()
            target = torch.cat([next_states - states, rewards_b.unsqueeze(-1)], dim=-1)
            for dyn, optd in zip(dyns, dyn_opts):
                mu, logv = dyn(states, A1)
                inv = torch.exp(-logv)
                loss = ((mu - target).pow(2) * inv + logv).mean()
                optd.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dyn.parameters(), 5.0)
                optd.step()

            with torch.no_grad():
                st = states
                for _ in range(rollout_len):
                    qs = q(st)
                    a = qs.argmax(dim=-1)
                    a1 = F.one_hot(a, env.n_actions).float()
                    d = dyns[np.random.randint(len(dyns))]
                    mu, logv = d(st, a1)
                    epsn = torch.randn_like(mu)
                    delta = mu + torch.exp(0.5 * logv) * epsn
                    ds = delta[:, :env.obs_dim]
                    rr = delta[:, env.obs_dim:env.obs_dim+1].squeeze(-1)
                    s_next = torch.clamp(st + ds, 0.0, 1.0)
                    done_model = torch.zeros_like(rr)
                    for i in range(s_next.size(0)):
                        buf_model.push(st[i].cpu().numpy(), int(a[i].item()), float(rr[i].item()),
                                       s_next[i].cpu().numpy(), float(done_model[i].item()))
                    st = s_next

        if len(buf_real) + len(buf_model) > start_learning:
            b_half = batch // 2

            def sample_mix(buf, n):
                if len(buf) >= n:
                    return buf.sample(n)
                elif len(buf) > 0:
                    return buf.sample(len(buf))
                else:
                    s_ = torch.zeros((0, env.obs_dim), dtype=torch.float32, device=device)
                    a_ = torch.zeros((0,), dtype=torch.long, device=device)
                    r_ = torch.zeros((0,), dtype=torch.float32, device=device)
                    s2_ = torch.zeros((0, env.obs_dim), dtype=torch.float32, device=device)
                    d_ = torch.zeros((0,), dtype=torch.float32, device=device)
                    return s_, a_, r_, s2_, d_, np.array([], dtype=int), torch.zeros((0,), device=device)

            r_batch = sample_mix(buf_real, b_half)
            m_batch = sample_mix(buf_model, batch - b_half)

            states = torch.cat([r_batch[0], m_batch[0]], dim=0)
            actions = torch.cat([r_batch[1], m_batch[1]], dim=0)
            rewards_b = torch.cat([r_batch[2], m_batch[2]], dim=0)
            next_states = torch.cat([r_batch[3], m_batch[3]], dim=0)
            dones = torch.cat([r_batch[4], m_batch[4]], dim=0)
            weights = torch.cat([
                r_batch[6] if isinstance(r_batch[6], torch.Tensor) else torch.ones(0, device=device),
                m_batch[6] if isinstance(m_batch[6], torch.Tensor) else torch.ones(0, device=device)
            ], dim=0)

            with torch.no_grad():
                q_next = qt(next_states).max(dim=1)[0]
                target = rewards_b + gamma * (1.0 - dones) * q_next
            q_pred = q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            td = target - q_pred
            loss = (td.pow(2) if weights.numel() == 0 else (weights * td.pow(2))).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0)
            opt.step()

            if isinstance(r_batch[5], np.ndarray) and r_batch[5].size > 0:
                prios_r = td[:len(r_batch[5])].abs().detach().cpu().numpy() + 1e-5
                buf_real.update_priorities(r_batch[5], prios_r)

        if t % target_sync == 0 and t > 0:
            qt.load_state_dict(q.state_dict())

    wall = time.time() - t_start
    return {"learning_curve": rewards_curve,
            "ep_returns": ep_rets,
            "time_to_mastery": np.array(ttm_list) if len(ttm_list) > 0 else np.array([np.nan]),
            "wall_clock": wall}

def run_experiment(algo, seeds=5, steps=30000, outdir="figures"):
    ensure_dir(outdir)

    def env_ctor(seed=0):
        return EduEnv(K=8, max_steps=100, beta=0.5, gamma=0.05, step_cost=0.01, seed=seed)

    methods = {
        "dqn_per": lambda sd: train_dqn_per(env_ctor, steps=steps, seed=sd),
        "ppo":     lambda sd: train_ppo(env_ctor, steps=steps, seed=sd),
        "pets":    lambda sd: train_pets(env_ctor, steps=steps, seed=sd),
        "mbpo":    lambda sd: train_mbpo(env_ctor, steps=steps, seed=sd),
    }
    if algo not in {"dqn_per", "ppo", "pets", "mbpo", "all"}:
        raise ValueError("Unknown --algo. Choose from: dqn_per | ppo | pets | mbpo | all")
    algos = ["dqn_per", "ppo", "pets", "mbpo"] if algo == "all" else [algo]

    results = {k: [] for k in algos}
    for alg in algos:
        print(f"===> Running {alg} for {seeds} seeds")
        for i in range(seeds):
            res = methods[alg](i)
            results[alg].append(res)

    # Learning curves (mean ± 95% CI)
    plt.figure(figsize=(8,4))
    for alg in algos:
        minL = min(len(r["learning_curve"]) for r in results[alg])
        arr = np.array([r["learning_curve"][:minL] for r in results[alg]])
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)
        se = sd / np.sqrt(arr.shape[0])
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
        x = np.arange(minL)
        plt.plot(x, mean, label=alg.upper())
        plt.fill_between(x, lo, hi, alpha=0.15)
    plt.xlabel("Environment Steps")
    plt.ylabel("Cumulative Reward (episode running sum)")
    plt.title("Learning Curves (mean ± 95% CI)")
    plt.legend()
    lc_path = os.path.join(outdir, "learning_curves_all.png")
    plt.tight_layout(); plt.savefig(lc_path, dpi=200); plt.close()

    # Time to mastery (bar with 95% CI)
    plt.figure(figsize=(6,4))
    labels, means, errs = [], [], []
    for alg in algos:
        ttm = []
        for r in results[alg]:
            x = r["time_to_mastery"]
            x = x[np.isfinite(x)]
            if x.size > 0:
                ttm.append(float(np.nanmin(x)))
        if len(ttm) == 0:
            m, e = np.nan, np.nan
        else:
            m = np.mean(ttm)
            sd = np.std(ttm, ddof=1) if len(ttm) > 1 else 0.0
            e = 1.96 * sd / math.sqrt(len(ttm)) if len(ttm) > 1 else 0.0
        labels.append(alg.upper()); means.append(m); errs.append(e)
    xpos = np.arange(len(labels))
    plt.bar(xpos, means, yerr=errs, capsize=4)
    plt.xticks(xpos, labels)
    plt.ylabel("Steps to Reach 0.95 Avg Mastery")
    plt.title("Time-to-Mastery (mean ± 95% CI)")
    ttm_path = os.path.join(outdir, "time_to_mastery_all.png")
    plt.tight_layout(); plt.savefig(ttm_path, dpi=200); plt.close()

    # Variance bands across seeds
    plt.figure(figsize=(8,4))
    for alg in algos:
        minL = min(len(r["learning_curve"]) for r in results[alg])
        arr = np.array([r["learning_curve"][:minL] for r in results[alg]])
        v = arr.var(axis=0, ddof=1)
        x = np.arange(minL)
        plt.plot(x, v, label=alg.upper())
    plt.xlabel("Environment Steps")
    plt.ylabel("Variance of Cumulative Reward")
    plt.title("Variance Bands Across Seeds")
    plt.legend()
    var_path = os.path.join(outdir, "variance_bands_all.png")
    plt.tight_layout(); plt.savefig(var_path, dpi=200); plt.close()

    # Compute vs Reward
    plt.figure(figsize=(6,4))
    for alg in algos:
        ws = [r["wall_clock"] for r in results[alg]]
        finals = [(r["learning_curve"][-1] if len(r["learning_curve"]) > 0 else 0.0) for r in results[alg]]
        plt.scatter(ws, finals, label=alg.upper(), alpha=0.7)
    plt.xlabel("Wall-clock Time (s)")
    plt.ylabel("Final Cumulative Reward")
    plt.title("Compute vs Reward")
    plt.legend()
    cvr_path = os.path.join(outdir, "compute_vs_reward.png")
    plt.tight_layout(); plt.savefig(cvr_path, dpi=200); plt.close()

    print("Saved figures:")
    for p in [lc_path, ttm_path, var_path, cvr_path]:
        print(" -", p)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="all", help="dqn_per | ppo | pets | mbpo | all")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--outdir", type=str, default="figures")
    args = parser.parse_args()
    run_experiment(args.algo, seeds=args.seeds, steps=args.steps, outdir=args.outdir)

if __name__ == "__main__":
    main()
