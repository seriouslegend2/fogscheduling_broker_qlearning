import random
import math

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

    def execute_tasks(self):
        # Execute tasks based on EDF scheduling
        if self.backup_queue:
            self.backup_queue.sort(key=lambda x: x['deadline'])
            for task in self.backup_queue:
                self.process_task(task)
        else:
            self.primary_queue.sort(key=lambda x: x['deadline'])
            for task in self.primary_queue:
                self.process_task(task)

    def process_task(self, task):
        # Simulate task processing
        pass

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
        self.fog_nodes = [FogNode(i, node_capacity, node_frequency, node_failure_rate) for i in range(num_fog_nodes)]
        self.tasks = []
        self.broker = Broker(**transmission_params)

    def add_task(self, task):
        self.tasks.append(task)

    def get_state(self):
        reliability = self.calculate_total_reliability()
        workload_distribution = self.calculate_workload_distribution()
        return (reliability, workload_distribution)

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

    def calculate_total_reliability(self):
        total_reliability = 1
        for node in self.fog_nodes:
            for task in node.primary_queue:
                primary_node = self.fog_nodes[task['primary']]
                backup_node = self.fog_nodes[task['backup']]
                Rc_fi = math.exp(-primary_node.failure_rate * task['length'] / primary_node.frequency)
                TRB_fj = self.broker.bandwidth * math.log2(1 + self.broker.channel_gain * self.broker.transmission_power / self.broker.noise)
                Rl_fi = math.exp(-self.broker.link_failure_rate * task['size'] / TRB_fj)
                R0_i = Rc_fi * Rl_fi
                task_reliability = 2 * R0_i - R0_i ** 2

                # Check if the task meets its deadline
                delay = self.calculate_delay(task, primary_node, backup_node)
                if delay > task['deadline']:
                    task_reliability = 0  # Task fails if it misses the deadline

                total_reliability *= task_reliability
        return total_reliability

    def calculate_workload_distribution(self):
        total_workload = sum(node.get_total_workload() for node in self.fog_nodes)
        avg_workload = total_workload / len(self.fog_nodes)
        workload_distribution = sum(abs(node.get_total_workload() - avg_workload) for node in self.fog_nodes)
        return workload_distribution

    def calculate_delay(self, task, primary_node, backup_node):
        T_ufj = self.broker.calculate_transmission_delay(task['size'], 1)  # Assuming distance is 1 for simplicity
        T_efj = task['length'] / primary_node.frequency
        T_Qfj = primary_node.get_total_workload() / primary_node.frequency
        return T_ufj + T_efj + T_Qfj

    def execute_all_tasks(self):
        for node in self.fog_nodes:
            node.execute_tasks()