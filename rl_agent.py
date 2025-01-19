import numpy as np
import random

class QLearningAgent:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}  # Use a dictionary for the Q-table

    def discretize_state(self, state):
        # Discretize each fog node's state and combine them into a single state representation
        discretized_state = []
        for reliability, workload_distribution in state:
            reliability_idx = min(int(reliability * 10), 9)
            workload_idx = min(int(workload_distribution * 10), 9)
            discretized_state.append(reliability_idx)
            discretized_state.append(workload_idx)
        return tuple(discretized_state)

    def get_q_value(self, state_idx, primary, backup):
        if state_idx not in self.q_table:
            self.q_table[state_idx] = np.zeros((self.num_actions, self.num_actions))
        return self.q_table[state_idx][primary, backup]

    def set_q_value(self, state_idx, primary, backup, value):
        if state_idx not in self.q_table:
            self.q_table[state_idx] = np.zeros((self.num_actions, self.num_actions))
        self.q_table[state_idx][primary, backup] = value

    def choose_action(self, state):
        state_idx = self.discretize_state(state)
        if random.uniform(0, 1) < self.exploration_rate:
            return {'primary': random.randint(0, self.num_actions - 1), 'backup': random.randint(0, self.num_actions - 1)}
        else:
            q_values = self.q_table.get(state_idx, np.zeros((self.num_actions, self.num_actions)))
            primary, backup = np.unravel_index(np.argmax(q_values), q_values.shape)
            return {'primary': primary, 'backup': backup}

    def learn(self, state, action, reward, next_state):
        state_idx = self.discretize_state(state)
        next_state_idx = self.discretize_state(next_state)
        primary = action['primary']
        backup = action['backup']
        predict = self.get_q_value(state_idx, primary, backup)
        target = reward + self.discount_factor * np.max(self.q_table.get(next_state_idx, np.zeros((self.num_actions, self.num_actions))))
        self.set_q_value(state_idx, primary, backup, predict + self.learning_rate * (target - predict))
        self.exploration_rate *= self.exploration_decay

    def calculate_reward(self, state, action, next_state, env, task):
        # Calculate the average reliability and workload distribution for the current and next states
        avg_reliability = sum(reliability for reliability, _ in state) / len(state)
        avg_workload_distribution = sum(workload_distribution for _, workload_distribution in state) / len(state)
        next_avg_reliability = sum(reliability for reliability, _ in next_state) / len(next_state)
        next_avg_workload_distribution = sum(workload_distribution for _, workload_distribution in next_state) / len(next_state)
        
        delay = env.calculate_delay(task, env.fog_nodes[action['primary']], env.fog_nodes[action['backup']])
        
        # Add epsilon to avoid division by zero
        epsilon = 1e-10
        reward = 0.36 * (avg_workload_distribution / (next_avg_workload_distribution + epsilon)) + \
                 0.27 * (delay / (task['deadline'] + epsilon)) + \
                 0.29 * (avg_reliability / (next_avg_reliability + epsilon))
        return reward