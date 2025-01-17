from environment import Environment
from rl_agent import QLearningAgent
import random

# Initialize environment and agent
transmission_params = {
    'processing_time': 0.1,
    'bandwidth': 50e6,  # 50 Mb/s
    'channel_gain': 1,
    'transmission_power': 1e-3,  # 10^-3 W
    'noise': 1e-10,  # 10^-10 W
    'link_failure_rate': 0.01
}

env = Environment(
    num_fog_nodes=10,
    node_capacity=100,
    node_frequency=2.5e9,
    node_failure_rate=0.01,
    transmission_params=transmission_params
)

agent = QLearningAgent(num_actions=10)

# Generate 100 random tasks
task_size_range = (50, 300)
task_length_range = (100, 1000)
task_deadline_range = (100, 5000)
task_frequency_range = (1e9, 5e9)

for i in range(100):
    task = {
        'id': i,
        'load': random.randint(task_size_range[0], task_size_range[1]),
        'size': random.randint(task_length_range[0], task_length_range[1]),
        'length': random.uniform(task_frequency_range[0], task_frequency_range[1]),
        'deadline': random.randint(task_deadline_range[0], task_deadline_range[1])
    }
    env.add_task(task)

# Run simulation
while len(env.tasks) > 0:
    state = env.get_state()
    action = agent.choose_action(state)
    task, reward = env.step(action)
    next_state = env.get_state()
    reward = agent.calculate_reward(state, action, next_state, env, task)
    agent.learn(state, action, reward, next_state)
    state = next_state

    # Display the allocation of tasks to fog nodes at each iteration
    print(f"Task {task['id']} assigned to:")
    print(f"  Primary Node: {action['primary']}")
    print(f"  Backup Node: {action['backup']}")
    print("\n")

print("Training completed.")

# Display the final allocation of tasks to fog nodes
for i, node in enumerate(env.fog_nodes):
    print(f"Fog Node {i}:")
    print(f"  Primary Queue: {[task['id'] for task in node.primary_queue]}")
    print(f"  Backup Queue: {[task['id'] for task in node.backup_queue]}")