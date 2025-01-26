from environment import Environment
from rl_agent import QLearningAgent
import random

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

# Number of tasks is constant
num_tasks = 100

def log_state_action_reward(iteration, state, action, reward, q_table, log_file):
    primary_node_state = state[action['primary']]
    backup_node_state = state[action['backup']]
    num_entries = len(q_table)
    num_q_values = num_entries * 10 * 10  # Assuming 10 actions for primary and 10 for backup
    log_file.write(f"Iteration: {iteration}\n")
    log_file.write(f"State: {state}\n")
    log_file.write(f"Action: {action}\n")
    log_file.write(f"Reward: {reward}\n")
    log_file.write(f"Q-Table Size: {num_entries} entries, {num_q_values} Q-values\n")
    log_file.write(f"Q-Table: {q_table}\n")
    log_file.write(f"Current Focus: Primary Node {action['primary']} (Reliability: {primary_node_state[0]}, Workload: {primary_node_state[1]}), "
                   f"Backup Node {action['backup']} (Reliability: {backup_node_state[0]}, Workload: {backup_node_state[1]})\n\n")

def run_simulation(env, agent, num_tasks, log_file):
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

    iteration = 0  # Initialize iteration counter
    reliability_list = []  # List to store reliability at each iteration

    # Run simulation
    while len(env.tasks) > 0:
        task = env.tasks.pop(0)  # Get the next task to process
        # Calculate reliability and workload distribution for each fog node
        state = env.get_state(log_file)
        # Choose an action based on the current state
        action = agent.choose_action(state)

        # Assign the primary task to the selected node
        task, reward = env.step(action)

        # Check if the task result is received before the deadline
        task_remaining_deadline = task['deadline'] - env.broker.processing_time
        while task_remaining_deadline > env.broker.calculate_transmission_delay(task['size'], 1) + env.broker.processing_time:
            if task in env.fog_nodes[action['primary']].primary_queue:
                env.fog_nodes[action['primary']].release_task(task)
                break
            task_remaining_deadline -= 1

        # If the task result is not received, send the backup task
        if task in env.fog_nodes[action['primary']].primary_queue:
            env.fog_nodes[action['primary']].release_task(task)
            env.fog_nodes[action['backup']].assign_task(task, is_backup=True)

        # Calculate the reward and update the Q-values
        next_state = env.get_state(log_file)
        reward = agent.calculate_reward(state, action, next_state, env, task)
        agent.learn(state, action, reward, next_state)

        # Log state, action, reward, and Q-values
        log_state_action_reward(iteration, state, action, reward, agent.q_table, log_file)

        # Log the next state after the action
        log_file.write(f"Next State: {next_state}\n\n")

        # Calculate and log the total reliability at this iteration
        total_reliability = env.calculate_total_reliability(log_file)
        reliability_list.append(total_reliability)

        iteration += 1  # Increment iteration counter

    # Execute all tasks in the fog nodes
    env.execute_all_tasks()

    return reliability_list

# Open a log file to write the output
with open('simulation_log.txt', 'w') as log_file:

    # Run simulation with a constant number of tasks
    env = Environment(
        num_fog_nodes=10,
        node_capacity=node_capacity,
        node_frequency=node_frequency,
        node_failure_rate=0.01,
        transmission_params=transmission_params
    )

    agent = QLearningAgent(num_actions=10)
    reliability_list = run_simulation(env, agent, num_tasks, log_file)
