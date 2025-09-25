import numpy as np
import heapq

from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        # TODO: Get reward from the environment and calculate the q-value
        next_state, reward, done = self.grid_world.step(state, action)
        if done:
            return reward
        return reward + self.discount_factor * self.values[next_state]


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        # TODO: Get the value for a state by calculating the q-values
        q_values = np.array([self.get_q_value(state, a) for a in range(self.grid_world.get_action_space())])
        return float((self.policy[state] * q_values).sum())

    def evaluate(self):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        new_values = np.zeros_like(self.values, dtype=float)
        for s in range(self.grid_world.get_state_space()):
            new_values[s] = self.get_state_value(s)
        delta = np.max(np.abs(new_values - self.values))
        self.values = new_values
        return delta

    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        while True:
            delta = self.evaluate()
            if delta < self.threshold:
                break


class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        action = self.policy[state]
        return self.get_q_value(state, action)

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        while True:
            new_values = np.zeros_like(self.values, dtype=float)
            for s in range(self.grid_world.get_state_space()):
                new_values[s] = self.get_state_value(s)
            delta = np.max(np.abs(new_values - self.values))
            self.values = new_values
            if delta < self.threshold:
                break

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        stable = True
        for s in range(self.grid_world.get_state_space()):
            old_action = self.policy[s]
            q_values = np.array([self.get_q_value(s, a) for a in range(self.grid_world.get_action_space())])
            self.policy[s] = int(np.argmax(q_values))
            if old_action != self.policy[s]:
                stable = False
        return stable

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        while True:
            self.policy_evaluation()
            stable = self.policy_improvement()
            if stable:
                break


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        q_values = np.array([self.get_q_value(state, a) for a in range(self.grid_world.get_action_space())])
        return float(np.max(q_values))

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        new_values = np.zeros_like(self.values, dtype=float)
        for s in range(self.grid_world.get_state_space()):
            new_values[s] = self.get_state_value(s)
        delta = np.max(np.abs(new_values - self.values))
        self.values = new_values
        return delta

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        for s in range(self.grid_world.get_state_space()):
            q_values = [self.get_q_value(s, a) for a in range(self.grid_world.get_action_space())]
            self.policy[s] = int(np.argmax(q_values))

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        while True:
            delta = self.policy_evaluation()
            if delta < self.threshold:
                break
        self.policy_improvement()


class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        INPLACE = 0
        PRIORITIZED = 1
        REALTIME = 2

        MODE = PRIORITIZED

        if MODE == INPLACE:
            while True:
                delta = 0.0
                for s in range(self.grid_world.get_state_space()):
                    v = self.values[s]
                    self.values[s] = float(np.max(np.array([self.get_q_value(s, a) for a in range(self.grid_world.get_action_space())])))
                    delta = max(delta, abs(self.values[s] - v))
                if delta < self.threshold:
                    break
            for s in range(self.grid_world.get_state_space()):
                q_values = [self.get_q_value(s, a) for a in range(self.grid_world.get_action_space())]
                self.policy[s] = int(np.argmax(q_values))

        elif MODE == PRIORITIZED:
            predecessors = [set() for _ in range(self.grid_world.get_state_space())]
            for s in range(self.grid_world.get_state_space()):
                for a in range(self.grid_world.get_action_space()):
                    ns, _, _ = self.grid_world.step(s, a)
                    predecessors[ns].add(s)

            def optimal_backup(s: int) -> float:
                best = -np.inf
                for a in range(self.grid_world.get_action_space()):
                    ns, r, d = self.grid_world.step(s, a)
                    q = float(r) if d else float(r + self.discount_factor * self.values[ns])
                    if q > best:
                        best = q
                return best

            def residual(s: int) -> float:
                return abs(optimal_backup(s) - self.values[s])

            pq = []
            for s in range(self.grid_world.get_state_space()):
                pr = residual(s)
                if pr > self.threshold:
                    heapq.heappush(pq, (-pr, s))

            while pq:
                top = -pq[0][0]
                if top < self.threshold:
                    break
                _, s = heapq.heappop(pq)
                old_v = self.values[s]
                self.values[s] = optimal_backup(s)
                if self.values[s] == old_v:
                    continue
                pr_self = residual(s)
                if pr_self > self.threshold:
                    heapq.heappush(pq, (-pr_self, s))
                for p in predecessors[s]:
                    pr = residual(p)
                    if pr > self.threshold:
                        heapq.heappush(pq, (-pr, p))

            for s in range(self.grid_world.get_state_space()):
                q_values = [self.get_q_value(s, a) for a in range(self.grid_world.get_action_space())]
                self.policy[s] = int(np.argmax(q_values))

        elif MODE == REALTIME:
            pass

        else:
            pass
