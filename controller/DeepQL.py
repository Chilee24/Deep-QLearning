import json
import numpy as np
import random
import os
from collections import deque, namedtuple
from controller.Controller import (Controller, action_space, decision_movement, convertphi, convertdeltaphi,
                                   convertdeltad, angle, remap_keys)

import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
GAMMA = 0.95  # Discount factor
EPSILON = 0.7  # Initial exploration rate
EPSILON_MIN = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.98  # Decay rate for exploration

LEARNING_RATE = 0.0005
BATCH_SIZE = 128
MEMORY_SIZE = 20000
TARGET_UPDATE_FREQ = 20  # Steps between target network updates

collisionDiscount = -20
successReward = 50
OBSTACLE_REWARD_FACTOR = 0.5
EPS = 1e-6  # To avoid division by zero

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural network to approximate Q-function
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.dropout(x)
        x = torch.nn.functional.leaky_relu(self.fc3(x), negative_slope=0.01)
        x = self.fc4(x)  # No activation for output layer
        return x

# Named tuple for transitions with priority
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'priority'))

class DeepQLearning(Controller):
    def __init__(self, cell_size, env_size, env_padding, goal, n_steps=5):
        super().__init__(cell_size, env_size, env_padding, goal)
        self.goal = goal
        self.state_dim = 5
        self.action_dim = len(action_space)
        self.n_steps = n_steps

        # Initialize neural networks and move to GPU if available
        self.policy_net = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.steps_done = 0
        self.episodeDecisions = []
        self.sumOfRewards = []
        self.averageReward = []

        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")

        self.reset()

    def reset(self) -> None:
        global EPSILON
        EPSILON = 0.7
        self.epsilon = EPSILON
        self.episodeDecisions.clear()
        self.sumOfRewards.clear()
        self.averageReward.clear()
        self.memory.clear()
        self.steps_done = 0

    def state_to_tensor(self, state):
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            noise = torch.randn_like(q_values) * 0.1
            q_values += noise
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done, priority=1.0):
        self.memory.append(Transition(state, action, reward, next_state, done, priority))

    def update_model(self):
        if len(self.memory) < BATCH_SIZE + self.n_steps - 1:
            return

        # Sample based on priority (simplified PER)
        priorities = [t.priority for t in self.memory]
        total_priority = sum(priorities)
        probs = [p / total_priority for p in priorities]
        indices = np.random.choice(len(self.memory), BATCH_SIZE, p=probs, replace=False)

        batch = [self.memory[idx] for idx in indices]
        states, actions, rewards, next_states, dones, _ = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            max_next_q = self.target_net(next_states).gather(1, next_actions)
            expected_q = rewards + (GAMMA ** self.n_steps) * max_next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities based on TD-error
        td_errors = (current_q - expected_q).abs().detach().cpu().numpy()
        for idx, td_error in zip(indices, td_errors):
            self.memory[idx] = self.memory[idx]._replace(priority=td_error.item() + 1e-6)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self):
        self.update_model()
        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.update_target_net()
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def setCollision(self, rb) -> None:
        if len(self.episodeDecisions) > 0:
            state, action_idx, reward = self.episodeDecisions[-1]
            reward += collisionDiscount
            self.episodeDecisions.pop()
            self.episodeDecisions.append((state, action_idx, reward))
            next_state = self.convertState(rb)
            next_state = np.array([next_state[0], next_state[1], -10, -10, -10], dtype=np.float32)
            self.store_transition(state, action_idx, reward, next_state, done=True, priority=10.0)  # High priority for collision
            self.learn()
            self.calculateReward()
        self.episodeDecisions.clear()

    def setSuccess(self) -> None:
        self.epsilon *= EPSILON_DECAY
        if len(self.episodeDecisions) > 0:
            state, action_idx, reward = self.episodeDecisions[-1]
            reward += successReward
            self.episodeDecisions.pop()
            self.episodeDecisions.append((state, action_idx, reward))
            goal_pos = np.array([int((self.goal[0] - self.env_padding) / self.cell_size),
                                 int((self.goal[1] - self.env_padding) / self.cell_size),
                                 -10, -10, -10], dtype=np.float32)
            self.store_transition(state, action_idx, reward, goal_pos, done=True, priority=10.0)  # High priority for success
            self.learn()
            self.calculateReward()
        self.episodeDecisions.clear()

    def calculateReward(self) -> None:
        sumOfReward = sum(reward for _, _, reward in self.episodeDecisions)
        self.sumOfRewards.append(sumOfReward)
        self.averageReward.append(sumOfReward / (len(self.episodeDecisions) + 1e-6))

    def outputPolicy(self, scenario, current_map, run_index) -> None:
        policy_dict = {}
        grid_size = int(self.env_size / self.cell_size)
        for i in range(grid_size):
            for j in range(grid_size):
                state = np.array([i, j, -10, -10, -10], dtype=np.float32)
                state_tensor = self.state_to_tensor(state)
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                best_action_idx = q_values.argmax().item()
                best_action = action_space[best_action_idx]
                key = (i, j, -10, -10, -10)
                policy_dict[key] = best_action
                for phi in range(3):
                    for delta_phi in range(-2, 3):
                        for delta_d in range(-1, 2):
                            state = np.array([i, j, phi, delta_phi, delta_d], dtype=np.float32)
                            state_tensor = self.state_to_tensor(state)
                            with torch.no_grad():
                                q_values = self.policy_net(state_tensor)
                            best_action_idx = q_values.argmax().item()
                            best_action = action_space[best_action_idx]
                            key = (i, j, phi, delta_phi, delta_d)
                            policy_dict[key] = best_action
        os.makedirs(f"policy/{scenario}/{current_map}/DeepQL/{run_index}", exist_ok=True)
        with open(f"policy/{scenario}/{current_map}/DeepQL/{run_index}/sumOfRewards.txt", "w") as outfile:
            outfile.write(str(self.sumOfRewards))
        with open(f"policy/{scenario}/{current_map}/DeepQL/{run_index}/averageReward.txt", "w") as outfile:
            outfile.write(str(self.averageReward))
        model_state_dict = self.policy_net.state_dict()
        torch.save(model_state_dict, f"policy/{scenario}/{current_map}/DeepQL/{run_index}/policy_net.pth")
        self.policy_net.to(device)
        self.target_net.to(device)

    def makeDecision(self, rb) -> tuple:
        state_arr = self.convertState(rb)
        state = np.array([state_arr[0], state_arr[1], -10, -10, -10], dtype=np.float32)
        action_idx = self.choose_action(state)
        action = action_space[action_idx]
        distance = self.calculateDistanceToGoal(state)
        movement = decision_movement[action]
        next_state = state.copy()
        next_state[0] += movement[0]
        next_state[1] += movement[1]
        next_distance = self.calculateDistanceToGoal(next_state)
        if action in ["up_1", "down_1", "left_1", "right_1"]:
            weight = 1.0
        else:
            weight = 1.0 / np.sqrt(2)
        reward = ((distance - next_distance) / (abs(distance - next_distance) + EPS)) * weight
        self.episodeDecisions.append((state, action_idx, reward))
        self.store_transition(state, action_idx, reward, next_state, done=False)
        self.learn()
        return decision_movement[action]

    def makeObstacleDecision(self, rb, obstacle_position) -> tuple:
        obstacle_before = obstacle_position[0]
        obstacle_after = obstacle_position[1]
        distance_to_obstacle = np.sqrt((rb.pos[0] - obstacle_before[0]) ** 2 + (rb.pos[1] - obstacle_before[1]) ** 2)
        distance_to_obstacle_next = np.sqrt((rb.pos[0] - obstacle_after[0]) ** 2 + (rb.pos[1] - obstacle_after[1]) ** 2)
        rb_direction = rb.nextPosition(self.goal)
        phi = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1],
                    obstacle_before[0] - rb.pos[0], obstacle_before[1] - rb.pos[1])
        phi_next = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1],
                         obstacle_after[0] - rb.pos[0], obstacle_after[1] - rb.pos[1])
        c_phi = convertphi(phi / np.pi * 180)
        c_deltaphi = convertdeltaphi((phi_next - phi) / np.pi * 180)
        c_deltad = convertdeltad((distance_to_obstacle_next - distance_to_obstacle))
        base_state = self.convertState(rb)
        state = np.array([base_state[0], base_state[1], c_phi, c_deltaphi, c_deltad], dtype=np.float32)
        action_idx = self.choose_action(state)
        action = action_space[action_idx]
        distance = self.calculateDistanceToGoal(state)
        movement = decision_movement[action]
        next_state = state.copy()
        next_state[0] += movement[0]
        next_state[1] += movement[1]
        next_distance = self.calculateDistanceToGoal(next_state)
        new_pos = (rb.pos[0] + movement[0] * self.cell_size, rb.pos[1] + movement[1] * self.cell_size)
        distance_to_obstacle_after = np.sqrt((new_pos[0] - obstacle_after[0]) ** 2 + (new_pos[1] - obstacle_after[1]) ** 2)
        if action in ["up_1", "down_1", "left_1", "right_1"]:
            weight = 1.0
        else:
            weight = 1.0 / np.sqrt(2)
        r_goal = ((distance - next_distance) / (abs(distance - next_distance) + EPS)) * weight
        r_obstacle = ((distance_to_obstacle_after - distance_to_obstacle) /
                      (abs(distance_to_obstacle_after - distance_to_obstacle) + EPS)) * weight
        reward = r_goal + OBSTACLE_REWARD_FACTOR * r_obstacle
        safety_threshold = 2 * self.cell_size
        if distance_to_obstacle_after < safety_threshold:
            reward -= 2
        elif distance_to_obstacle_after > distance_to_obstacle:
            reward += 1
        self.episodeDecisions.append((state, action_idx, reward))
        self.store_transition(state, action_idx, reward, next_state, done=False)
        self.learn()
        return decision_movement[action]