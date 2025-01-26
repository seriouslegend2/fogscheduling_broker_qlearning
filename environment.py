import math
import numpy as np
import random

class FogNode:
    def __init__(self, node_id, capacity, frequency, failure_rate):
        self.node_id = node_id
        self.capacity = capacity
        self.frequency = frequency
        self.failure_rate = failure_rate
        self.current_load = 0
        self.primary_queue = []
        self.backup_queue = []

    def assign_task(self, task, is_backup=False):
        if self.current_load + task['load'] <= self.capacity:
            self.current_load += task['load']
            if is_backup:
                self.backup_queue.append(task)
            else:
                self.primary_queue.append(task)
            return True
        return False

    def release_task(self, task):
        self.current_load -= task['load']
        if task in self.primary_queue:
            self.primary_queue.remove(task)
        elif task in self.backup_queue:
            self.backup_queue.remove(task)

    def get_total_workload(self):
        return sum(t['length'] for t in self.primary_queue) + sum(t['length'] for t in self.backup_queue)

    def schedule_tasks_by_deadline(self):
        # Sort tasks by their deadlines in both primary and backup queues
        self.primary_queue.sort(key=lambda task: task['deadline'])
        self.backup_queue.sort(key=lambda task: task['deadline'])

    def execute_tasks(self):
        # Schedule tasks in both queues using EDF
        self.schedule_tasks_by_deadline()

        # Execute tasks in the primary queue
        for task in self.primary_queue:
            self.release_task(task)

        # Execute tasks in the backup queue
        for task in self.backup_queue:
            self.release_task(task)

    def calculate_reliability(self, broker, log_file):
        total_reliability = 1
        for task in self.primary_queue:
            print(f"Calculating reliability for node {self.node_id}")
            Rc_fi = math.exp(-self.failure_rate * task['length'] / self.frequency)
            TRB_fj = broker.bandwidth * math.log2(1 + broker.channel_gain * broker.transmission_power / broker.noise)
            Rl_fi = math.exp(-broker.link_failure_rate * task['size'] / TRB_fj)
            R0_i = Rc_fi * Rl_fi
            task_reliability = 2 * R0_i - R0_i ** 2

            # Check if the task meets its deadline
            delay = broker.calculate_transmission_delay(task['size'], 1) + task['length'] / self.frequency + self.get_total_workload() / self.frequency
            if delay > task['deadline']:
                task_reliability = 0  # Task fails if it misses the deadline

            total_reliability *= task_reliability

            # Log intermediate values for debugging
            log_file.write(f"Task ID: {task['id']}, Rc_fi: {Rc_fi}, TRB_fj: {TRB_fj}, Rl_fi: {Rl_fi}, R0_i: {R0_i}, Task Reliability: {task_reliability}, Total Reliability: {total_reliability}\n")

        return total_reliability


class Broker:
    def __init__(self, processing_time, bandwidth, channel_gain, transmission_power, noise, link_failure_rate):
        self.processing_time = processing_time
        self.bandwidth = bandwidth
        self.channel_gain = channel_gain
        self.transmission_power = transmission_power
        self.noise = noise
        self.link_failure_rate = link_failure_rate

    def calculate_transmission_delay(self, task_size, distance):
        TRB_fj = self.bandwidth * math.log2(1 + self.channel_gain * self.transmission_power / self.noise)
        T_ufj = task_size / TRB_fj
        return self.processing_time + T_ufj


class Environment:
    def __init__(self, num_fog_nodes, node_capacity, node_frequency, node_failure_rate, transmission_params):
        # Initialize fog nodes with given parameters
        self.fog_nodes = [FogNode(i, node_capacity, node_frequency, node_failure_rate) for i in range(num_fog_nodes)]
        self.tasks = []
        self.broker = Broker(**transmission_params)

    def add_task(self, task):
        self.tasks.append(task)

    def get_state(self, log_file):
        state = []
        for node in self.fog_nodes:
            reliability = node.calculate_reliability(self.broker, log_file)
            workload_distribution = node.get_total_workload()
            state.append((reliability, workload_distribution))
        return state

    def step(self, action):
        task = self.tasks.pop(0)
        primary_node = self.fog_nodes[action['primary']]
        backup_node = self.fog_nodes[action['backup']]
        task['primary'] = action['primary']
        task['backup'] = action['backup']

        if primary_node.assign_task(task):
            if not backup_node.assign_task(task, is_backup=True):
                primary_node.release_task(task)
                return task, -1  # Backup assignment failed
            return task, 1  # Successful assignment
        return task, 0  # Primary assignment failed

    def reset(self):
        for node in self.fog_nodes:
            node.current_load = 0
            node.primary_queue = []
            node.backup_queue = []
        self.tasks = []

    def calculate_delay(self, task, primary_node, backup_node):
        T_ufj = self.broker.calculate_transmission_delay(task['size'], 1)  # Assuming distance is 1 for simplicity
        T_efj = task['length'] / primary_node.frequency
        T_Qfj = primary_node.get_total_workload() / primary_node.frequency
        return T_ufj + T_efj + T_Qfj

    def calculate_total_reliability(self, log_file):
        total_reliability = 1
        for node in self.fog_nodes:
            node_reliability = node.calculate_reliability(self.broker, log_file)
            total_reliability *= node_reliability
        return total_reliability

    def execute_all_tasks(self):
        for node in self.fog_nodes:
            node.execute_tasks()

    def task_scheduler(self):
        for node in self.fog_nodes:
            if node.backup_queue:
                node.schedule_tasks_by_deadline()  # EDF for backup queue
            else:
                node.schedule_tasks_by_deadline()  # EDF for primary queue
