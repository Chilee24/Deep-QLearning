import json
import numpy as np
import random
import os
from collections import deque
from controller.Controller import (Controller, action_space, decision_movement, convertphi, convertdeltaphi,
                                   convertdeltad, angle, remap_keys)

import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
GAMMA = 0.9  # Discount factor
EPSILON = 0.5  # Initial exploration rate
EPSILON_MIN = 0.05  # Minimum exploration rate
EPSILON_DECAY = 0.95  # Decay rate for exploration

LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 10  # Steps between target network updates

collisionDiscount = -5
successReward = 15
OBSTACLE_REWARD_FACTOR = 0.2
EPS = 1e-6  # To avoid division by zero
# Hyperparameters
# GAMMA = 0.95  # Discount factor tăng nhẹ để ưu tiên phần thưởng dài hạn
# EPSILON = 0.7  # Initial exploration rate tăng để khám phá nhiều hơn ban đầu
# EPSILON_MIN = 0.01  # Minimum exploration rate giảm để tập trung khai thác khi đã học tốt
# EPSILON_DECAY = 0.98  # Decay rate giảm chậm hơn để duy trì khám phá lâu hơn

# LEARNING_RATE = 0.0005  # Giảm nhẹ để ổn định quá trình học
# BATCH_SIZE = 64  # Tăng để học từ nhiều mẫu hơn mỗi lần cập nhật
# MEMORY_SIZE = 20000  # Tăng để lưu trữ nhiều kinh nghiệm hơn
# TARGET_UPDATE_FREQ = 20  # Tăng để cập nhật mạng mục tiêu chậm hơn, ổn định hơn

# collisionDiscount = -20  # Phạt mạnh hơn khi va chạm
# successReward = 50  # Thưởng lớn hơn khi thành công
# OBSTACLE_REWARD_FACTOR = 0.5  # Tăng ảnh hưởng của phần thưởng liên quan đến chướng ngại vật
# EPS = 1e-6  # Giữ nguyên để tránh chia cho 0

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Neural network to approximate Q-function
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DeepQLearning(Controller):
    def __init__(self, cell_size, env_size, env_padding, goal, n_steps=3):
        # Initialize base controller
        super().__init__(cell_size, env_size, env_padding, goal)
        self.goal = goal
        
        # State dimension: (i, j, phi, delta_phi, delta_d)
        # For no obstacle case: (i, j, -10, -10, -10)
        self.state_dim = 5
        self.action_dim = len(action_space)

        # Initialize neural networks and move to GPU if available
        self.policy_net = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.n_steps = n_steps

        # Initialize optimizer and replay memory
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        # Initialize tracking variables
        self.epsilon = EPSILON
        self.steps_done = 0
        self.episodeDecisions = []
        self.sumOfRewards = []
        self.averageReward = []
        
        # Log GPU status
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")

        self.reset()

    def reset(self) -> None:
        global EPSILON
        EPSILON = 0.5
        self.epsilon = EPSILON

        self.episodeDecisions.clear()
        self.sumOfRewards.clear()
        self.averageReward.clear()
        self.memory.clear()
        self.steps_done = 0

    # State to tensor conversion with GPU support
    def state_to_tensor(self, state):
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    # Choose action using epsilon-greedy strategy
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    # Store transition in replay memory
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Update neural network model
    # def update_model(self):
    #     if len(self.memory) < BATCH_SIZE:
    #         return

    #     batch = random.sample(self.memory, BATCH_SIZE)
    #     states, actions, rewards, next_states, dones = zip(*batch)
        
    #     # Move data to GPU
    #     states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    #     actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
    #     rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    #     next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    #     dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

    #     # Current Q-values
    #     current_q = self.policy_net(states).gather(1, actions)
        
    #     # Target Q-values
    #     with torch.no_grad():
    #         next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
    #         max_next_q = self.target_net(next_states).gather(1, next_actions)
    #         expected_q = rewards + GAMMA * max_next_q * (1 - dones)

    #     # Compute loss and update
    #     loss = nn.MSELoss()(current_q, expected_q)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    def update_model(self):
        if len(self.memory) < BATCH_SIZE + self.n_steps - 1:  # Đảm bảo đủ dữ liệu cho n bước
            return

        # Tính số lượng mẫu tối đa có thể lấy
        max_idx = len(self.memory) - self.n_steps + 1
        if max_idx < BATCH_SIZE:  # Nếu không đủ mẫu cho BATCH_SIZE, bỏ qua
            return

        # Lấy mẫu ngẫu nhiên các chỉ số bắt đầu từ memory
        start_indices = random.sample(range(max_idx), BATCH_SIZE)

        # Chuẩn bị dữ liệu cho batch
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for idx in start_indices:
            # Tính phần thưởng tích lũy và lấy trạng thái cuối cùng
            multi_step_reward = 0
            for step in range(self.n_steps):
                state, action, reward, next_state, done = self.memory[idx + step]
                multi_step_reward += (GAMMA ** step) * reward
                if done:  # Nếu gặp trạng thái kết thúc, dừng lại
                    break
                
            # Lấy trạng thái ban đầu và trạng thái sau n bước (hoặc khi done)
            states.append(state)
            actions.append(action)
            rewards.append(multi_step_reward)
            next_states.append(self.memory[idx + min(step, self.n_steps - 1)][3])  # next_state cuối cùng
            dones.append(self.memory[idx + min(step, self.n_steps - 1)][4])  # done cuối cùng

        # Chuyển dữ liệu sang tensor và GPU
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions)

        # Target Q-values với Multi-Step DDQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)  # Chọn hành động từ policy_net
            max_next_q = self.target_net(next_states).gather(1, next_actions)  # Đánh giá từ target_net
            expected_q = rewards + (GAMMA ** self.n_steps) * max_next_q * (1 - dones)

        # Compute loss and update
        loss = nn.MSELoss()(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Update target network periodically
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # Learning step
    # def learn(self):
    #     self.update_model()
    #     self.steps_done += 1
    #     if self.steps_done % TARGET_UPDATE_FREQ == 0:
    #         self.update_target_net()

    #     # Decay epsilon
    #     if self.epsilon > EPSILON_MIN:
    #         self.epsilon *= EPSILON_DECAY
    def learn(self):
        self.update_model()
        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.update_target_net()

        # Decay epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    # Add collision discount to the last decision
    def setCollision(self, rb) -> None:
        if len(self.episodeDecisions) > 0:
            state, action_idx, reward = self.episodeDecisions[-1]
            reward += collisionDiscount

            self.episodeDecisions.pop()
            self.episodeDecisions.append((state, action_idx, reward))

            next_state = self.convertState(rb)
            next_state = np.array([next_state[0], next_state[1], -10, -10, -10], dtype=np.float32)
            
            # Mark as terminal state (collision)
            self.store_transition(state, action_idx, reward, next_state, done=True)
            
            # Learn from this experience
            self.learn()
            self.calculateReward()

        # Clear episode decisions
        self.episodeDecisions.clear()

    # Add success reward to the last decision
    def setSuccess(self) -> None:
        # Decay epsilon
        self.epsilon *= EPSILON_DECAY

        if len(self.episodeDecisions) > 0:
            state, action_idx, reward = self.episodeDecisions[-1]
            reward += successReward

            self.episodeDecisions.pop()
            self.episodeDecisions.append((state, action_idx, reward))

            goal_pos = np.array([
                int((self.goal[0] - self.env_padding) / self.cell_size),
                int((self.goal[1] - self.env_padding) / self.cell_size),
                -10, -10, -10
            ], dtype=np.float32)
            
            # Mark as terminal state (success)
            self.store_transition(state, action_idx, reward, goal_pos, done=True)
            
            # Learn from this experience
            self.learn()
            self.calculateReward()

        # Clear episode decisions
        self.episodeDecisions.clear()

    # Calculate rewards for the episode
    def calculateReward(self) -> None:
        sumOfReward = 0
        for state, _, reward in self.episodeDecisions:
            sumOfReward += reward

        self.sumOfRewards.append(sumOfReward)
        self.averageReward.append(sumOfReward / (len(self.episodeDecisions) + 1e-6))

    # Output policy to json file
    def outputPolicy(self, scenario, current_map, run_index) -> None:
        # Create dictionary to store policy
        policy_dict = {}
        grid_size = int(self.env_size / self.cell_size)
        
        # Evaluate for each state
        for i in range(grid_size):
            for j in range(grid_size):
                # Case with no obstacle: (i, j, -10, -10, -10)
                state = np.array([i, j, -10, -10, -10], dtype=np.float32)
                state_tensor = self.state_to_tensor(state)
                
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                best_action_idx = q_values.argmax().item()
                best_action = action_space[best_action_idx]
                
                key = (i, j, -10, -10, -10)
                policy_dict[key] = best_action
                
                # Also evaluate for different obstacle configurations
                for phi in range(3):  # Phi: F, S, D
                    for delta_phi in range(-2, 3):  # DeltaPhi: C, LC, U, LA, A
                        for delta_d in range(-1, 2):  # DeltaD: C, U, A
                            state = np.array([i, j, phi, delta_phi, delta_d], dtype=np.float32)
                            state_tensor = self.state_to_tensor(state)
                            
                            with torch.no_grad():
                                q_values = self.policy_net(state_tensor)
                            best_action_idx = q_values.argmax().item()
                            best_action = action_space[best_action_idx]
                            
                            key = (i, j, phi, delta_phi, delta_d)
                            policy_dict[key] = best_action

        # Đảm bảo thư mục tồn tại
        os.makedirs(f"policy/{scenario}/{current_map}/DeepQL/{run_index}", exist_ok=True)
        
        # Lưu thông tin phần thưởng
        with open(f"policy/{scenario}/{current_map}/DeepQL/{run_index}/sumOfRewards.txt", "w") as outfile:
            outfile.write(str(self.sumOfRewards))
        
        with open(f"policy/{scenario}/{current_map}/DeepQL/{run_index}/averageReward.txt", "w") as outfile:
            outfile.write(str(self.averageReward))
        
        # Lưu tham số mạng policy_net (tách biệt khỏi thiết bị)
        # Đảm bảo sao chép state_dict trước khi chuyển mạng sang CPU
        model_state_dict = self.policy_net.state_dict()  # Lấy state_dict từ mạng hiện tại
        torch.save(model_state_dict, f"policy/{scenario}/{current_map}/DeepQL/{run_index}/policy_net.pth")
        
        # Move models back to device
        self.policy_net.to(device)
        self.target_net.to(device)

    def makeDecision(self, rb) -> tuple:
        # Get state from robot position (without obstacle)
        state_arr = self.convertState(rb)
        state = np.array([state_arr[0], state_arr[1], -10, -10, -10], dtype=np.float32)
        
        # Choose action using epsilon-greedy
        action_idx = self.choose_action(state)
        action = action_space[action_idx]

        # Calculate reward based on distance to goal
        distance = self.calculateDistanceToGoal(state)
        movement = decision_movement[action]
        
        # Calculate next state
        next_state = state.copy()
        next_state[0] += movement[0]
        next_state[1] += movement[1]
        next_distance = self.calculateDistanceToGoal(next_state)

        # Determine weight based on movement direction
        if action in ["up_1", "down_1", "left_1", "right_1"]:
            weight = 1.0
        else:
            weight = 1.0 / np.sqrt(2)

        # Reward is positive if moving closer to goal, negative if moving away
        reward = ((distance - next_distance) / (abs(distance - next_distance) + EPS)) * weight

        # Store experience in replay memory and episode decisions
        self.episodeDecisions.append((state, action_idx, reward))
        self.store_transition(state, action_idx, reward, next_state, done=False)
        
        # Learn from experiences
        self.learn()

        return decision_movement[action]

    def makeObstacleDecision(self, rb, obstacle_position) -> tuple:
        # Get obstacle positions before and after movement
        obstacle_before = obstacle_position[0]
        obstacle_after = obstacle_position[1]

        # Calculate distance to obstacle
        distance_to_obstacle = np.sqrt((rb.pos[0] - obstacle_before[0]) ** 2 + (rb.pos[1] - obstacle_before[1]) ** 2)
        distance_to_obstacle_next = np.sqrt((rb.pos[0] - obstacle_after[0]) ** 2 + (rb.pos[1] - obstacle_after[1]) ** 2)

        # Calculate relative angle
        rb_direction = rb.nextPosition(self.goal)
        phi = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1], 
                    obstacle_before[0] - rb.pos[0], obstacle_before[1] - rb.pos[1])
        phi_next = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1], 
                         obstacle_after[0] - rb.pos[0], obstacle_after[1] - rb.pos[1])

        # Convert to discretized state
        c_phi = convertphi(phi / np.pi * 180)
        c_deltaphi = convertdeltaphi((phi_next - phi) / np.pi * 180)
        c_deltad = convertdeltad((distance_to_obstacle_next - distance_to_obstacle))

        # Get base state from robot position
        base_state = self.convertState(rb)
        # State with obstacle: (i, j, c_phi, c_deltaphi, c_deltad)
        state = np.array([base_state[0], base_state[1], c_phi, c_deltaphi, c_deltad], dtype=np.float32)

        # Choose action using epsilon-greedy
        action_idx = self.choose_action(state)
        action = action_space[action_idx]

        # Calculate reward based on distance to goal
        distance = self.calculateDistanceToGoal(state)
        movement = decision_movement[action]
        
        # Calculate next state and distance
        next_state = state.copy()
        next_state[0] += movement[0]
        next_state[1] += movement[1]
        next_distance = self.calculateDistanceToGoal(next_state)

        # Calculate distance to obstacle after movement
        new_pos = (rb.pos[0] + movement[0] * self.cell_size, rb.pos[1] + movement[1] * self.cell_size)
        distance_to_obstacle_after = np.sqrt(
            (new_pos[0] - obstacle_after[0]) ** 2 + (new_pos[1] - obstacle_after[1]) ** 2)

        # Determine weight based on movement direction
        if action in ["up_1", "down_1", "left_1", "right_1"]:
            weight = 1.0
        else:
            weight = 1.0 / np.sqrt(2)

        # Calculate goal-based reward
        r_goal = ((distance - next_distance) / (abs(distance - next_distance) + EPS)) * weight
        
        # Calculate obstacle-based reward
        r_obstacle = ((distance_to_obstacle_after - distance_to_obstacle) / 
                      (abs(distance_to_obstacle_after - distance_to_obstacle) + EPS)) * weight
        
        # Combine rewards
        reward = r_goal + OBSTACLE_REWARD_FACTOR * r_obstacle

        # Store experience and update
        self.episodeDecisions.append((state, action_idx, reward))
        self.store_transition(state, action_idx, reward, next_state, done=False)
        
        # Learn from experiences
        self.learn()

        return decision_movement[action]