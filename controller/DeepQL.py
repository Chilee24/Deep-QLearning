import json
import os
import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from controller.Controller import Controller, action_space, decision_movement, remap_keys, convertphi, convertdeltaphi, \
    convertdeltad, angle

# Hyperparameters
GAMMA = 0.9
EPSILON = 0.5  # Khởi đầu epsilon
EPSILON_MIN = 0.05  # Giá trị epsilon tối thiểu
EPSILON_DECAY = 0.95

LEARNING_RATE = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 10  # Số bước giữa các lần cập nhật target network

collisionDiscount = -5
successReward = 15
OBSTACLE_REWARD_FACTOR = 0.2
EPS = 1e-6  # Để tránh chia cho 0


# Định nghĩa mạng neural xấp xỉ Q-function
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
    def __init__(self, cell_size, env_size, env_padding, goal):
        super().__init__(cell_size, env_size, env_padding, goal)
        # Ở đây trạng thái có 5 thành phần: (i, j, phi, delta_phi, delta_d)
        # Với trường hợp không phát hiện vật cản, ta dùng (i, j, -10, -10, -10)
        self.state_dim = 5
        self.action_dim = len(action_space)

        self.policy_net = QNetwork(self.state_dim, self.action_dim)
        self.target_net = QNetwork(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.steps_done = 0
        self.episodeRewards = []  # Lưu phần thưởng của mỗi episode

        self.reset()

    def reset(self):
        self.epsilon = EPSILON
        self.episodeRewards.clear()
        self.memory.clear()

    def convertState(self, rb):
        """
        Chuyển đổi vị trí của robot thành trạng thái dạng (i, j, phi, delta_phi, delta_d).
        Ở trường hợp không có vật cản, ta dùng (i, j, -10, -10, -10).
        """
        i = int((rb.pos[0] - self.env_padding) / self.cell_size)
        j = int((rb.pos[1] - self.env_padding) / self.cell_size)
        return np.array([i, j, -10, -10, -10], dtype=np.float32)

    def calculateDistanceToGoal(self, state):
        """
        Tính khoảng cách từ trạng thái (dựa vào chỉ số ô) đến mục tiêu.
        Công thức sử dụng khoảng cách đường chéo.
        """
        # state[0] và state[1] là chỉ số ô
        pos = (self.env_padding + self.cell_size / 2 + state[0] * self.cell_size,
               self.env_padding + self.cell_size / 2 + state[1] * self.cell_size)
        x = self.goal[0] - pos[0]
        y = self.goal[1] - pos[1]
        return np.abs(x - y) + np.sqrt(2) * min(np.abs(x), np.abs(y))

    def choose_action(self, state):
        """
        Chiến lược epsilon-greedy: với xác suất epsilon, chọn hành động ngẫu nhiên,
        ngược lại chọn hành động có giá trị Q cao nhất.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Q-value hiện tại
        current_q = self.policy_net(states).gather(1, actions)
        # Q-value mục tiêu
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q = rewards + GAMMA * max_next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self):
        self.update_model()
        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.update_target_net()

    def outputPolicy(self, scenario, current_map, run_index):
        """
        Xuất policy từ mạng neural ra file JSON bằng cách đánh giá cho từng trạng thái (ô)
        và chọn hành động có Q-value cao nhất.
        """
        policy_dict = {}
        grid_size = int(self.env_size / self.cell_size)
        for i in range(grid_size):
            for j in range(grid_size):
                # Ở trạng thái không có vật cản: (i, j, -10, -10, -10)
                state = np.array([i, j, -10, -10, -10], dtype=np.float32)
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                best_action = action_space[q_values.argmax().item()]
                policy_dict[(i, j)] = best_action

        # Construct the path to save the policy
        directory = f"policy/{scenario}/{current_map}/DeepQL"
        os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist

        with open(f"policy/{scenario}/{current_map}/DeepQL/{run_index}.json", "w") as outfile:
            json.dump(remap_keys(policy_dict), outfile, indent=2)

    def makeDecision(self, rb):
        """
        Quyết định hành động dựa trên trạng thái của robot (không có vật cản).
        Tính phần thưởng theo sự thay đổi khoảng cách đến mục tiêu.
        """
        state = self.convertState(rb)  # state dạng (i, j, -10, -10, -10)
        action_idx = self.choose_action(state)
        action = action_space[action_idx]

        # Tính khoảng cách đến mục tiêu trước và sau khi di chuyển
        distance = self.calculateDistanceToGoal(state)
        movement = decision_movement[action]
        # Tính trạng thái kế tiếp giả định (chỉ cập nhật 2 thành phần đầu)
        next_state = state.copy()
        next_state[0] += movement[0]
        next_state[1] += movement[1]
        next_distance = self.calculateDistanceToGoal(next_state)

        # Xác định trọng số w
        if action in ["up_1", "down_1", "left_1", "right_1"]:
            weight = 1.0
        else:
            weight = 1.0 / np.sqrt(2)

        # Tính phần thưởng: dương nếu tiến gần mục tiêu, âm nếu đi xa
        reward = ((distance - next_distance) / (abs(distance - next_distance) + EPS)) * weight

        # Lưu transition vào replay buffer; done = True nếu đạt được mục tiêu (để minh họa, ta đặt done = False)
        self.store_transition(state, action_idx, reward, next_state, done=False)

        return decision_movement[action]

    def makeObstacleDecision(self, rb, obstacle_position):
        """
        Quyết định hành động khi có vật cản.
        Tính phần thưởng dựa trên sự thay đổi khoảng cách đến mục tiêu và vật cản.
        """
        # Lấy thông tin về vật cản trước và sau khi di chuyển
        obstacle_before = obstacle_position[0]
        obstacle_after = obstacle_position[1]

        # Tính khoảng cách từ robot đến vật cản trước
        distance_to_obstacle = np.sqrt((rb.pos[0] - obstacle_before[0]) ** 2 + (rb.pos[1] - obstacle_before[1]) ** 2)
        # Tính khoảng cách từ robot đến vật cản sau
        distance_to_obstacle_next = np.sqrt((rb.pos[0] - obstacle_after[0]) ** 2 + (rb.pos[1] - obstacle_after[1]) ** 2)

        rb_direction = rb.nextPosition(self.goal)
        phi = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1],
                    obstacle_before[0] - rb.pos[0], obstacle_before[1] - rb.pos[1])
        phi_next = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1],
                         obstacle_after[0] - rb.pos[0], obstacle_after[1] - rb.pos[1])

        # Chuyển đổi góc thành các giá trị rời rạc
        c_phi = convertphi(phi / np.pi * 180)
        c_deltaphi = convertdeltaphi((phi_next - phi) / np.pi * 180)
        c_deltad = convertdeltad((distance_to_obstacle_next - distance_to_obstacle))

        # Lấy trạng thái từ vị trí robot
        base_state = self.convertState(rb)
        # Ở trường hợp có vật cản, trạng thái gồm: (i, j, c_phi, c_deltaphi, c_deltad)
        state = np.array([base_state[0], base_state[1], c_phi, c_deltaphi, c_deltad], dtype=np.float32)

        action_idx = self.choose_action(state)
        action = action_space[action_idx]

        # Tính phần thưởng dựa trên khoảng cách đến mục tiêu
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

        r_goal = ((distance - next_distance) / (abs(distance - next_distance) + EPS)) * weight

        # Tính phần thưởng từ việc thay đổi khoảng cách đến vật cản
        # Robot sau khi di chuyển sẽ có vị trí: rb.pos + movement * cell_size
        new_pos = (rb.pos[0] + movement[0] * self.cell_size, rb.pos[1] + movement[1] * self.cell_size)
        distance_to_obstacle_after = np.sqrt(
            (new_pos[0] - obstacle_after[0]) ** 2 + (new_pos[1] - obstacle_after[1]) ** 2)
        r_obstacle = ((distance_to_obstacle_after - distance_to_obstacle) / (
                    abs(distance_to_obstacle_after - distance_to_obstacle) + EPS)) * weight

        reward = r_goal + OBSTACLE_REWARD_FACTOR * r_obstacle

        self.store_transition(state, action_idx, reward, next_state, done=False)

        return decision_movement[action]
