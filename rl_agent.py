import numpy as np
import random

class QLearningAgent:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros((10, 10, num_actions, num_actions))  # Discretized state space

    def discretize_state(self, state):
        reliability, workload_distribution = state
        reliability_idx = min(int(reliability * 10), 9)
        workload_idx = min(int(workload_distribution * 10), 9)
        return (reliability_idx, workload_idx)

    def choose_action(self, state):
        state_idx = self.discretize_state(state)
        if random.uniform(0, 1) < self.exploration_rate:
            return {'primary': random.randint(0, self.num_actions - 1), 'backup': random.randint(0, self.num_actions - 1)}
        else:
            primary = np.argmax(self.q_table[state_idx][:, :self.num_actions])
            primary = np.clip(primary, 0, self.num_actions - 1)
            backup = np.argmax(self.q_table[state_idx][primary, :self.num_actions])
            backup = np.clip(backup, 0, self.num_actions - 1)
            return {'primary': primary, 'backup': backup}

    def learn(self, state, action, reward, next_state):
        state_idx = self.discretize_state(state)
        next_state_idx = self.discretize_state(next_state)
        primary = action['primary']
        backup = action['backup']
        predict = self.q_table[state_idx][primary, backup]
        target = reward + self.discount_factor * np.max(self.q_table[next_state_idx])
        self.q_table[state_idx][primary, backup] += self.learning_rate * (target - predict)
        self.exploration_rate *= self.exploration_decay

    def calculate_reward(self, state, action, next_state, env, task):
        reliability, workload_distribution = state
        next_reliability, next_workload_distribution = next_state
        delay = env.calculate_delay(task, env.fog_nodes[action['primary']], env.fog_nodes[action['backup']])
        
        # Add epsilon to avoid division by zero
        epsilon = 1e-10
        reward = 0.36 * (workload_distribution / (next_workload_distribution + epsilon)) + \
                 0.27 * (delay / (task['deadline'] + epsilon)) + \
                 0.29 * (reliability / (next_reliability + epsilon))
        return reward