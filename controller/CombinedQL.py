import json
import numpy as np
import random
from controller.Controller import (Controller, action_space, decision_movement, convertphi, convertdeltaphi,
                                   convertdeltad, angle, remap_keys)

# Hyperparameters
GAMMA = 0.9  # 0.8 to 0.9

EPSILON = 0.5
EPSILON_DECAY = 0.95

ALPHA = 0.9  # 0.2 to 0.9
LEARNING_RATE_DECAY = 1.0

collisionDiscount = -5
successReward = 15

OBSTACLE_REWARD_FACTOR = 0.2


class QLearning(Controller):
    def __init__(self, cell_size, env_size, env_padding, goal):
        # Initialize Qtable and policy
        super().__init__(cell_size, env_size, env_padding, goal)
        self.goal = goal
        self.Qtable = {}
        self.episodeDecisions = []
        self.sumOfRewards = []
        self.averageReward = []

        self.reset()

    def reset(self) -> None:
        global EPSILON
        EPSILON = 0.5

        self.episodeDecisions.clear()
        self.sumOfRewards.clear()
        self.averageReward.clear()

        # Initialize Qtable and policy
        for i in range(int(self.env_size / self.cell_size)):
            for j in range(int(self.env_size / self.cell_size)):
                # Initialize policy to always go to the goal, wherever the robot is
                cell_center = (self.env_padding + self.cell_size / 2 + i * self.cell_size, self.env_padding + self.cell_size / 2 + j * self.cell_size)
                direction = (self.goal[0] - cell_center[0], self.goal[1] - cell_center[1])
                decision = ""

                # A 90-degree region is divided into 3 parts
                # For example, 0 - 90: right, up-right, up
                ratio = np.tan(np.pi / 8)
                if abs(direction[1]) > ratio * abs(direction[0]):
                    if direction[1] > 0:
                        decision += "down"
                    else:
                        decision += "up"

                    if abs(direction[0]) > ratio * abs(direction[1]):
                        if direction[0] > 0:
                            decision += "-right"
                        else:
                            decision += "-left"
                else:
                    if direction[0] > 0:
                        decision += "right"
                    else:
                        decision += "left"

                # If the robot is too far from the goal, add 2 to the decision, else add 1
                distance = self.calculateDistanceToGoal((i, j))
                decision += "_1"

                # Initialize Qtable, value is higher if the robot is closer to the goal
                if distance > 0:
                    ini_value = self.cell_size / distance
                else:
                    ini_value = 0

                for phi in range(3):  # Phi: F, S, D
                    for delta_phi in range(-2, 3):  # DeltaPhi: C, LC, U, LA, A
                        for delta_d in range(-1, 2):  # DeltaD: C, U, A
                            self.policy[(i, j, phi, delta_phi, delta_d)] = decision

                            for action in action_space:
                                self.Qtable[(i, j, phi, delta_phi, delta_d, action)] = ini_value

                # (-10, -10, -10) is the state where no obstacle is detected
                self.policy[(i, j, -10, -10, -10)] = decision

                for action in action_space:
                    self.Qtable[(i, j, -10, -10, -10, action)] = ini_value

    # Add collision discount to the last decision if the robot has collided with an obstacle
    def setCollision(self, rb) -> None:
        if len(self.episodeDecisions) > 0:
            state, decision, reward = self.episodeDecisions[-1]
            reward += collisionDiscount

            self.episodeDecisions.pop()
            self.episodeDecisions.append((state, decision, reward))

            next_state = self.convertState(rb)
            self.episodeDecisions.append(((next_state[0], next_state[1], -10, -10, -10), "", 0))

            # Update Qtable and policy after collision
            self.updateAll()
            self.calculateReward()

        # Clear episode decisions
        self.episodeDecisions.clear()

    # Add success reward to the last decision if the robot has reached the goal
    def setSuccess(self) -> None:
        # Decay epsilon
        global EPSILON
        EPSILON *= EPSILON_DECAY

        if len(self.episodeDecisions) > 0:
            state, decision, reward = self.episodeDecisions[-1]
            reward += successReward

            self.episodeDecisions.pop()
            self.episodeDecisions.append((state, decision, reward))

            goal_pos = (int((self.goal[0] - self.env_padding) / self.cell_size),
                        int((self.goal[1] - self.env_padding) / self.cell_size))
            self.episodeDecisions.append(((goal_pos[0], goal_pos[1], -10, -10, -10), "", 0))

            # Update Qtable and policy after success
            self.updateAll()
            self.calculateReward()

        # Clear episode decisions
        self.episodeDecisions.clear()

    # Calculate the sum of rewards and average reward of the episode after the episode ends
    def calculateReward(self) -> None:
        sumOfReward = 0
        for episodeDecision in self.episodeDecisions:
            sumOfReward += episodeDecision[2]

        self.sumOfRewards.append(sumOfReward)
        self.averageReward.append(sumOfReward / (len(self.episodeDecisions) + 1e-6))

    # Out put policy to json file
    def outputPolicy(self, scenario, current_map, run_index) -> None:
        # Create directory if it doesn't exist
        import os
        if not os.path.exists(f"policy/{scenario}/{current_map}/CombinedQL/{run_index}"):
            os.makedirs(f"policy/{scenario}/{current_map}/CombinedQL/{run_index}", exist_ok=True)
        with open(f"policy/{scenario}/{current_map}/CombinedQL/{run_index}/policy.json", "w") as outfile:
            json.dump(remap_keys(self.policy), outfile, indent=2)

        with open(f"policy/{scenario}/{current_map}/CombinedQL/{run_index}/sumOfRewards.txt", "w") as outfile:
            outfile.write(str(self.sumOfRewards))

        with open(f"policy/{scenario}/{current_map}/CombinedQL/{run_index}/averageReward.txt", "w") as outfile:
            outfile.write(str(self.averageReward))

    def updateQtable(self, state, decision, reward, next_state) -> None:
        # Optimal value of next state
        optimalQnext = max([self.Qtable[(next_state[0], next_state[1], next_state[2], next_state[3],
                                         next_state[4], action)] for action in action_space])

        # Update Qtable
        self.Qtable[(state[0], state[1], state[2], state[3], state[4], decision)] = (1 - ALPHA) * self.Qtable[
            (state[0], state[1], state[2], state[3], state[4], decision)] + ALPHA * (reward + GAMMA * optimalQnext)

    def updatePolicy(self, state) -> None:
        # Update policy
        bestAction = max(action_space, key=lambda action: self.Qtable[(state[0], state[1], state[2], state[3], state[4], action)])

        self.policy[state] = bestAction

    def updateAll(self) -> None:
        if len(self.episodeDecisions) >= 2:
            state, decision, reward = self.episodeDecisions[-2]
            next_state = self.episodeDecisions[-1][0]

            # Update Qtable
            self.updateQtable(state, decision, reward, next_state)

            # Update policy
            self.updatePolicy(state)

    def makeDecision(self, rb) -> tuple:
        self.updateAll()

        state = self.convertState(rb)
        state = (state[0], state[1], -10, -10, -10)

        # Epsilon greedy
        # Randomly choose an action
        if random.random() < EPSILON:
            decision = random.choice(action_space)
        # Choose the best action
        else:
            decision = self.policy[state]

        # Calculate reward
        distance = self.calculateDistanceToGoal(state)

        movement = decision_movement[decision]
        next_state = (state[0] + movement[0] * self.cell_size, state[1] + movement[1] * self.cell_size)
        next_distance = self.calculateDistanceToGoal(next_state)

        if decision in ["up_1", "down_1", "left_1", "right_1"]:
            weight = 1
        else:
            weight = 1 / np.sqrt(2)

        # Reward is the change in distance to the goal (negative reward if the robot is moving away from the goal)
        reward = ((distance - next_distance) / np.abs(distance - next_distance + 1e-6)) * weight

        # Add to episode decisions
        self.episodeDecisions.append((state, decision, reward))

        return decision_movement[decision]

    def makeObstacleDecision(self, rb, obstacle_position) -> tuple:
        self.updateAll()

        # Get the position of the obstacle before and after moving
        # Then calculate the relative position of the obstacle to the robot
        obstacle_before = obstacle_position[0]
        obstacle_after = obstacle_position[1]

        distance_to_obstacle = np.sqrt((rb.pos[0] - obstacle_before[0]) ** 2 + (rb.pos[1] - obstacle_before[1]) ** 2)
        distance_to_obstacle_next = np.sqrt((rb.pos[0] - obstacle_after[0]) ** 2 + (rb.pos[1] - obstacle_after[1]) ** 2)

        rb_direction = rb.nextPosition(self.goal)
        phi = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1], obstacle_before[0] - rb.pos[0],
                    obstacle_before[1] - rb.pos[1])
        phi_next = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1], obstacle_after[0] - rb.pos[0],
                         obstacle_after[1] - rb.pos[1])

        # Convert to state
        c_phi = convertphi(phi / np.pi * 180)
        c_deltaphi = convertdeltaphi((phi_next - phi) / np.pi * 180)
        c_deltad = convertdeltad((distance_to_obstacle_next - distance_to_obstacle))

        state = self.convertState(rb)
        state = (state[0], state[1], c_phi, c_deltaphi, c_deltad)

        # Epsilon greedy
        # Randomly choose an action
        if random.random() < EPSILON:
            decision = random.choice(action_space)
        # Choose the best action
        else:
            decision = self.policy[state]

        # Calculate reward
        distance = self.calculateDistanceToGoal(state)

        movement = decision_movement[decision]
        next_state = (state[0] + movement[0], state[1] + movement[1])
        next_distance = self.calculateDistanceToGoal(next_state)

        distance_to_obstacle_after_movement = np.sqrt((rb.pos[0] + movement[0] * self.cell_size - obstacle_after[0]) ** 2
                                                      + (rb.pos[1] + movement[1] * self.cell_size - obstacle_after[1]) ** 2)

        if decision in ["up_1", "down_1", "left_1", "right_1"]:
            weight = 1
        else:
            weight = 1 / np.sqrt(2)

        # Reward is the change in distance to the goal (negative reward if the robot is moving away from the goal)
        # plus the change in distance to the obstacle (negative reward if the robot is moving closer to the obstacle)
        reward = ((distance - next_distance) / np.abs(distance - next_distance + 1e-6)) * weight
        + OBSTACLE_REWARD_FACTOR * (distance_to_obstacle_after_movement - distance_to_obstacle) / np.abs(distance_to_obstacle_after_movement - distance_to_obstacle + 1e-6) * weight

        # Add to episode decisions
        self.episodeDecisions.append((state, decision, reward))

        return decision_movement[decision]
