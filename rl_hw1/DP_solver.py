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
        q_values = np.array([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
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
            for state in range(self.grid_world.get_state_space()):
                new_values[state] = self.get_state_value(state)
            delta = np.max(np.abs(new_values - self.values))
            self.values = new_values
            if delta < self.threshold:
                break

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        stable = True
        for state in range(self.grid_world.get_state_space()):
            old_action = self.policy[state]
            q_values = np.array([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
            self.policy[state] = int(np.argmax(q_values))
            if old_action != self.policy[state]:
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
        q_values = np.array([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
        return float(np.max(q_values))

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        new_values = np.zeros_like(self.values, dtype=float)
        for state in range(self.grid_world.get_state_space()):
            new_values[state] = self.get_state_value(state)
        delta = np.max(np.abs(new_values - self.values))
        self.values = new_values
        return delta

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        for state in range(self.grid_world.get_state_space()):
            q_values = [self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())]
            self.policy[state] = int(np.argmax(q_values))

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
        OTHER = 2

        MODE = OTHER

        if MODE == INPLACE:
            while True:
                delta = 0.0
                for state in range(self.grid_world.get_state_space()):
                    v = self.values[state]
                    self.values[state] = float(np.max(np.array([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])))
                    delta = max(delta, abs(self.values[state] - v))
                if delta < self.threshold:
                    break
            for state in range(self.grid_world.get_state_space()):
                q_values = [self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())]
                self.policy[state] = int(np.argmax(q_values))

        elif MODE == PRIORITIZED:
            predecessors = [set() for _ in range(self.grid_world.get_state_space())]
            for state in range(self.grid_world.get_state_space()):
                for action in range(self.grid_world.get_action_space()):
                    next_state, _, _ = self.grid_world.step(state, action)
                    predecessors[next_state].add(state)

            def optimal_backup(state: int) -> float:
                return float(np.max(np.array([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])))

            def residual(state: int) -> float:
                return abs(optimal_backup(state) - self.values[state])

            pq = []
            for state in range(self.grid_world.get_state_space()):
                pr = residual(state)
                if pr > self.threshold:
                    heapq.heappush(pq, (-pr, state))

            while pq:
                top = -pq[0][0]
                if top < self.threshold:
                    break
                _, state = heapq.heappop(pq)
                old_v = self.values[state]
                self.values[state] = optimal_backup(state)
                if self.values[state] == old_v:
                    continue
                pr_self = residual(state)
                if pr_self > self.threshold:
                    heapq.heappush(pq, (-pr_self, state))
                for p in predecessors[state]:
                    pr = residual(p)
                    if pr > self.threshold:
                        heapq.heappush(pq, (-pr, p))

            for state in range(self.grid_world.get_state_space()):
                q_values = [self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())]
                self.policy[state] = int(np.argmax(q_values))

        else:
            next_state = np.zeros((self.grid_world.get_state_space(), self.grid_world.get_action_space()), dtype=int)
            reward     = np.zeros((self.grid_world.get_state_space(), self.grid_world.get_action_space()), dtype=float)
            done       = np.zeros((self.grid_world.get_state_space(), self.grid_world.get_action_space()), dtype=bool)

            for state in range(self.grid_world.get_state_space()):
                for action in range(self.grid_world.get_action_space()):
                    ns, r, d = self.grid_world.step(state, action)
                    next_state[state, action] = ns
                    reward[state, action]     = r
                    done[state, action]       = d

            while True:
                delta = 0.0
                for state in range(self.grid_world.get_state_space()):
                    v = self.values[state]
                    q_values = [
                        reward[state, action] if done[state, action]
                        else reward[state, action] + self.discount_factor * self.values[next_state[state, action]]
                        for action in range(self.grid_world.get_action_space())
                    ]
                    self.values[state] = float(np.max(np.array(q_values)))
                    delta = max(delta, abs(self.values[state] - v))
                if delta < self.threshold:
                    break

            for state in range(self.grid_world.get_state_space()):
                q_values = [
                    reward[state, action] if done[state, action]
                    else reward[state, action] + self.discount_factor * self.values[next_state[state, action]]
                    for action in range(self.grid_world.get_action_space())
                ]
                self.policy[state] = int(np.argmax(q_values))
