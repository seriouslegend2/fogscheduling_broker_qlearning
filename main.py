from environment import Environment
from rl_agent import QLearningAgent
import random
import matplotlib.pyplot as plt

# Initialize transmission parameters
transmission_params = {
    'processing_time': 0.1,
    'bandwidth': 50e6,  # 50 Mb/s
    'channel_gain': 1,
    'transmission_power': 1e-3,  # 10^-3 W
    'noise': 1e-10,  # 10^-10 W
    'link_failure_rate': 0.01
}

node_capacity = 100
node_frequency = 2.5e9

task_size_range = (50, 300)
task_length_range = (100, 1000)
task_deadline_range = (100, 5000)
task_frequency_range = (1e9, 5e9)

# List to store the results for number of tasks
num_tasks_list = range(100, 501, 50)
total_reliability_list_tasks = []

# List to store the results for number of fog nodes
num_fog_nodes_list = [5, 10, 15, 20]
total_reliability_list_fog_nodes = []

# List to store the results for failure rates
failure_rate_list = [0.0001, 0.001, 0.01, 0.1, 0.5]
total_reliability_list_failure_rate = []

# Run simulations for varying number of tasks
for num_tasks in num_tasks_list:
    env = Environment(
        num_fog_nodes=10,
        node_capacity=node_capacity,
        node_frequency=node_frequency,
        node_failure_rate=0.01,
        transmission_params=transmission_params
    )

    agent = QLearningAgent(num_actions=10)

    # Generate random tasks
    for i in range(num_tasks):
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

    # Execute all tasks in the fog nodes
    env.execute_all_tasks()

    # Record the total reliability
    total_reliability = env.calculate_total_reliability()
    total_reliability_list_tasks.append(total_reliability)

# Run simulations for varying number of fog nodes
for num_fog_nodes in num_fog_nodes_list:
    env = Environment(
        num_fog_nodes=num_fog_nodes,
        node_capacity=node_capacity,
        node_frequency=node_frequency,
        node_failure_rate=0.01,
        transmission_params=transmission_params
    )

    agent = QLearningAgent(num_actions=num_fog_nodes)

    # Generate random tasks (fixed number of tasks, e.g., 200)
    num_tasks = 200
    for i in range(num_tasks):
        task = {
            'id': i,
            'load': random.randint(task_size_range[0], task_size_range[1]),
            'size': random.randint(task_length_range[0], task_length_range[1]),
            'length': random.uniform(task_frequency_range[0], task_frequency_range[1]),
            'deadline': random.randint(task_deadline_range[0], task_deadline_range[0])
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

    # Execute all tasks in the fog nodes
    env.execute_all_tasks()

    # Record the total reliability
    total_reliability = env.calculate_total_reliability()
    total_reliability_list_fog_nodes.append(total_reliability)

# Run simulations for varying failure rates
for failure_rate in failure_rate_list:
    env = Environment(
        num_fog_nodes=10,
        node_capacity=node_capacity,
        node_frequency=node_frequency,
        node_failure_rate=failure_rate,
        transmission_params=transmission_params
    )

    agent = QLearningAgent(num_actions=10)

    # Generate random tasks (fixed number of tasks, e.g., 200)
    num_tasks = 200
    for i in range(num_tasks):
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

    # Execute all tasks in the fog nodes
    env.execute_all_tasks()

    # Record the total reliability
    total_reliability = env.calculate_total_reliability()
    total_reliability_list_failure_rate.append(total_reliability)

# Plot the results for number of tasks
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(num_tasks_list, total_reliability_list_tasks, marker='o')
plt.xlabel('Number of Tasks')
plt.ylabel('Total Reliability')
plt.title('Total Reliability vs. Number of Tasks')
plt.grid(True)

# Plot the results for number of fog nodes
plt.subplot(1, 3, 2)
plt.plot(num_fog_nodes_list, total_reliability_list_fog_nodes, marker='o')
plt.xlabel('Number of Fog Nodes')
plt.ylabel('Total Reliability')
plt.title('Total Reliability vs. Number of Fog Nodes')
plt.grid(True)

# Plot the results for failure rates
plt.subplot(1, 3, 3)
plt.plot(failure_rate_list, total_reliability_list_failure_rate, marker='o')
plt.xlabel('Failure Rate')
plt.ylabel('Total Reliability')
plt.title('Total Reliability vs. Failure Rate')
plt.grid(True)

plt.tight_layout()
plt.show()