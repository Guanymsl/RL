import numpy as np
import json
from collections import deque

from gridworld import GridWorld

from tqdm import tqdm

# =========================== 2.1 model free prediction ===========================
class ModelFreePrediction:
    """
    Base class for ModelFreePrediction algorithms
    """

    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
            seed (int): seed for sampling action from the policy
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.max_episode = max_episode
        self.episode_counter = 0
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.values       = np.zeros(self.state_space)
        self.rng = np.random.default_rng(seed)      # only call this in collect_data()
        if policy:
            self.policy = policy
        else:
            self.policy = np.ones((self.state_space, self.action_space)) / self.action_space  # random policy

    def get_all_state_values(self) -> np.array:
        return self.values

    def collect_data(self) -> tuple:
        """
        Use the stochastic policy to interact with the environment and collect one step of data.
        Samples an action based on the action probability distribution for the current state.
        """

        current_state = self.grid_world.get_current_state()  # Get the current state

        # Sample an action based on the stochastic policy's probabilities for the current state
        action_probs = self.policy[current_state]
        action = self.rng.choice(self.action_space, p=action_probs)

        next_state, reward, done = self.grid_world.step(action)
        if done:
            self.episode_counter +=1
        return next_state, reward, done


class MonteCarloPrediction(ModelFreePrediction):
    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Constructor for MonteCarloPrediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
        """
        super().__init__(grid_world, policy, discount_factor, max_episode, seed)

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with first-visit Monte-Carlo method

        current_state = self.grid_world.reset()
        N = np.zeros(self.state_space)
        while self.episode_counter < self.max_episode:
            pi = []
            done = False
            while not done:
                next_state, reward, done = self.collect_data()
                pi.append((current_state, reward))
                current_state = next_state

            G = 0
            returns = np.zeros(self.state_space)
            for t in reversed(range(len(pi))):
                s, r = pi[t]
                G = self.discount_factor * G + r
                returns[s] = G

            visited = set()
            for t in range(len(pi)):
                s, r = pi[t]
                if s not in visited:
                    visited.add(s)
                    N[s] += 1
                    self.values[s] += (returns[s] - self.values[s]) / N[s]


class TDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld,learning_rate: float, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate

    def run(self) -> None:
        """Run the algorithm until max episode"""
        # TODO: Update self.values with TD(0) Algorithm

        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            done = False
            while not done:
                next_state, reward, done = self.collect_data()
                self.values[current_state] += self.lr * (reward + (0 if done else self.discount_factor * self.values[next_state]) - self.values[current_state])
                current_state = next_state


class NstepTDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld, learning_rate: float, num_step: int, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
            num_step (int): n_step look ahead for TD
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate
        self.n      = num_step

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with N-step TD Algorithm

        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            t = 0
            S = [current_state]
            R = [0]
            T = float('inf')
            while True:
                if t < T:
                    next_state, reward, done = self.collect_data()
                    S.append(next_state)
                    R.append(reward)
                    if done:
                        T = t + 1

                tau = t - self.n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        G += (self.discount_factor ** (i - tau - 1)) * R[i]
                    if tau + self.n < T:
                        G += (self.discount_factor ** self.n) * self.values[S[tau + self.n]]
                    self.values[S[tau]] += self.lr * (G - self.values[S[tau]])

                if tau == T - 1:
                    break
                t += 1

# =========================== 2.2 model free control ===========================
class ModelFreeControl:
    """
    Base class for model free control algorithms
    """

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space # stocastic policy
        self.policy_index = np.zeros(self.state_space, dtype=int)                              # deterministic policy

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index

    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values


class MonteCarloPolicyIteration(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> float:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)

        G = 0
        for t in reversed(range(len(state_trace) - 1)):
            s, a, r = state_trace[t], action_trace[t], reward_trace[t]
            G = self.discount_factor * G + r
            self.q_values[s, a] += self.lr * (G - self.q_values[s, a])

        return abs(G - self.q_values[s, a])

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy

        a_stars = self.get_policy_index()
        for s in range(self.state_space):
            for a in range(self.action_space):
                if a == a_stars[s]:
                    self.policy[s, a] = 1 - self.epsilon + self.epsilon / self.action_space
                else:
                    self.policy[s, a] = self.epsilon / self.action_space

    def run(self, max_episode=1000) -> tuple:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy

        learn = np.zeros(max_episode)
        el    = np.zeros(max_episode)

        current_state = self.grid_world.reset()
        state_trace   = [current_state]
        action_trace  = []
        reward_trace  = []
        rng = np.random.default_rng(1)
        for iter_episode in tqdm(range(max_episode)):
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here

            done = False
            while not done:
                action = rng.choice(self.action_space, p=self.policy[current_state])
                next_state, reward, done = self.grid_world.step(action)
                state_trace.append(next_state)
                action_trace.append(action)
                reward_trace.append(reward)
                current_state = next_state

            el[iter_episode] = self.policy_evaluation(state_trace, action_trace, reward_trace)
            self.policy_improvement()
            learn[iter_episode] = np.mean(reward_trace)
            state_trace  = [current_state]
            action_trace = []
            reward_trace = []

        return learn, el


class SARSA(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> float:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy

        self.q_values[s, a] += self.lr * (r + (0 if is_done else self.discount_factor * self.q_values[s2, a2]) - self.q_values[s, a])

        a_star = np.argmax(self.q_values[s])
        for action in range(self.action_space):
            if action == a_star:
                self.policy[s, action] = 1 - self.epsilon + self.epsilon / self.action_space
            else:
                self.policy[s, action] = self.epsilon / self.action_space

        return abs(r + (0 if is_done else self.discount_factor * self.q_values[s2, a2]) - self.q_values[s, a])

    def run(self, max_episode=1000) -> np.array:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy

        learn = np.zeros(max_episode)
        el    = np.zeros(max_episode)

        current_state = self.grid_world.reset()
        rng = np.random.default_rng(1)
        prev_s = current_state
        for iter_episode in tqdm(range(max_episode)):
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here

            rs = []
            els = []

            prev_a = rng.choice(self.action_space, p=self.policy[prev_s])
            is_done = False
            while not is_done:
                s, prev_r, is_done = self.grid_world.step(prev_a)
                rs.append(prev_r)
                a = rng.choice(self.action_space, p=self.policy[s]) if not is_done else None
                els.append(self.policy_eval_improve(prev_s, prev_a, prev_r, s, a, is_done))
                prev_s, prev_a = s, a

            learn[iter_episode] = np.mean(rs)
            el[iter_episode]    = np.mean(els)

        return learn, el

class Q_Learning(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer

        self.buffer.append((s, a, r, s2, d))

    def sample_batch(self) -> np.ndarray:
        # TODO: sample a batch of index of transitions from the buffer

        return [self.buffer[i] for i in np.random.choice(len(self.buffer), size=min(self.sample_batch_size, len(self.buffer)), replace=False)]

    def policy_eval_improve(self, s, a, r, s2, is_done) -> float:
        """Evaluate the policy and update the values after one step"""
        #TODO: Evaluate Q value after one step and improve the policy

        self.q_values[s, a] += self.lr * (r + (0 if is_done else self.discount_factor * np.max(self.q_values[s2])) - self.q_values[s, a])

        a_star = np.argmax(self.q_values[s])
        for action in range(self.action_space):
            if action == a_star:
                self.policy[s, action] = 1 - self.epsilon + self.epsilon / self.action_space
            else:
                self.policy[s, action] = self.epsilon / self.action_space

        return abs(r + (0 if is_done else self.discount_factor * np.max(self.q_values[s2])) - self.q_values[s, a])

    def run(self, max_episode=1000) -> tuple:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm

        learn = np.zeros(max_episode)
        el    = np.zeros(max_episode)

        iter_episode = 0
        current_state = self.grid_world.reset()
        rng = np.random.default_rng(1)
        prev_s = current_state
        transition_count = 0
        for iter_episode in tqdm(range(max_episode)):
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here

            rs = []
            els = []

            is_done = False
            while not is_done:
                prev_a = rng.choice(self.action_space, p=self.policy[prev_s])
                s, prev_r, is_done = self.grid_world.step(prev_a)
                rs.append(prev_r)
                self.add_buffer(prev_s, prev_a, prev_r, s, is_done)
                transition_count += 1

                if transition_count % self.update_frequency == 0 and len(self.buffer) > 0:
                    batch = self.sample_batch()
                    for (bs, ba, br, bs2, bd) in batch:
                        els.append(self.policy_eval_improve(bs, ba, br, bs2, bd))

                prev_s = s

            learn[iter_episode] = np.mean(rs)
            el[iter_episode]    = np.mean(els) if len(els) > 0 else min(el)

        return learn, el
