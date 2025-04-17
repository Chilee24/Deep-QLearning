import pygame
import time
import numpy as np
import os
import sys
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '\\controller')
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '\\environment')
from pygame.locals import *
from environment.Robot import Robot
from environment.Colors import *
from environment.MapData import maps
from controller.DeepQL import QNetwork

# Choose the version of the algorithm:
version = input("Enter version (1-ClassicalQL, 2-DFQL, 3-CombinedQL, 4-DualQL, 5-DWA, 6-DeepQL): ")
if version == "1":
    from controller.ClassicalQL import QLearning as Controller
    algorithm = "ClassicalQL"
elif version == "2":
    from controller.DFQL import QLearning as Controller
    algorithm = "DFQL"
elif version == "3":
    from controller.CombinedQL import QLearning as Controller
    algorithm = "CombinedQL"
elif version == "4":
    from controller.DualQL import QLearning as Controller
    algorithm = "DualQL"
elif version == "5":
    from controller.DWA import DynamicWindowApproach as Controller
    algorithm = "DWA"
elif version == "6":
    from controller.DeepQL import DeepQLearning as Controller
    algorithm = "DeepQL"
else:
    algorithm = "Unknown"

if version == "4":
    from controller.Controller import ControllerTesterDual as ControllerTester
elif version == "3":
    from controller.Controller import ControllerTesterCombined as ControllerTester
else:
    from controller.Controller import ControllerTester

isTraining = version != "5" and input("Training? (y/n): ") == "y"
scenario = input("Enter scenario (uniform/diverse/complex/map1): ")
input_map = input("Enter map (1/2/3): ")
numsOfRuns = 20 if input("Automatically run 20 times? (y/n): ") == "y" else 1

# Initialize the robot
start = maps[scenario + input_map][0]["Start"]
goal = maps[scenario + input_map][0]["Goal"]
cell_size = 16
env_size = 500
env_padding = int(env_size * 0.02)
path = []
pathLength = 0
distanceToObstacle = []
turningAngle = []
success_counter = 0
consecutive_successes = 0  # Biến đếm số lần thành công liên tiếp
success_threshold = 50  # Ngưỡng để dừng huấn luyện (100 lần thành công liên tiếp)

robot = Robot(start=start, cell_size=cell_size,
              decisionMaker=Controller(cell_size=cell_size, env_size=env_size, env_padding=env_padding, goal=goal))

def draw_target(window, target) -> None:
    pygame.draw.circle(window, RED, target, 8, 0)

def draw_start(window, start) -> None:
    pygame.draw.circle(window, GREEN, start, 6, 0)

def draw_path(window, path, color) -> None:
    for i in range(1, len(path)):
        pygame.draw.line(window, color, path[i - 1], path[i], 2)

def draw_grid(window, cell_size, env_size, env_padding) -> None:
    for i in range(1, int(env_size / cell_size)):
        pygame.draw.line(window, BLACK, (env_padding + i * cell_size, env_padding),
                         (env_padding + i * cell_size, env_padding + env_size), 1)
        pygame.draw.line(window, BLACK, (env_padding, env_padding + i * cell_size),
                         (env_padding + env_size, env_padding + i * cell_size), 1)

def get_sum_turning_angle(path):
    total_angle = 0
    for i in range(1, len(path) - 1):
        vector1 = ([path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]])
        vector2 = ([path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]])
        angle = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6))
        total_angle += np.abs(angle)
    return total_angle

def main(test_map):
    global success_counter, pathLength, consecutive_successes

    env_width = env_height = env_size
    NORTH_PAD, SOUTH_PAD, LEFT_PAD, RIGHT_PAD = env_padding, 3 * env_padding, env_padding, env_padding
    SCREEN_WIDTH = env_width + LEFT_PAD + RIGHT_PAD
    SCREEN_HEIGHT = env_height + NORTH_PAD + SOUTH_PAD

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    pygame.display.set_caption("Q-Learning Path Planning")
    my_font = pygame.font.SysFont("arial", SOUTH_PAD // 5)

    finished = False
    obstacles_list = []
    pause = False
    started = False
    distanceStartGoal = np.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)
    path.append(start)

    robot.resetPosition(start)

    if test_map in maps:
        obstacles_list = maps[test_map][0]["Obstacles"]

    while not finished:
        screen.fill(WHITE)
        button1 = pygame.draw.rect(screen, BLACK, (LEFT_PAD + int(env_width * 0.7), NORTH_PAD * 2 + env_height,
                                                   int(env_width * 0.2), int(SOUTH_PAD * 0.4)), 4)
        button1_text = my_font.render("Start", True, (0, 0, 0))
        button1_rect = button1_text.get_rect(center=button1.center)
        screen.blit(button1_text, button1_rect)

        button2 = pygame.draw.rect(screen, BLACK, (LEFT_PAD + int(env_width * 0.4), NORTH_PAD * 2 + env_height,
                                                   int(env_width * 0.2), int(SOUTH_PAD * 0.4)), 4)
        button2_text = my_font.render("Pause", True, (0, 0, 0))
        button2_rect = button2_text.get_rect(center=button2.center)
        screen.blit(button2_text, button2_rect)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                if button1.collidepoint(mouse_x, mouse_y):
                    started = True
                elif button2.collidepoint(mouse_x, mouse_y):
                    pause = not pause

        if started or numsOfRuns > 1:
            if pause:
                continue
            else:
                for obstacle in obstacles_list:
                    obstacle.move()
                    obstacle.draw(screen, with_past=False)

                robot.makeDecision(obstacles_list)
                robot.move()

                path.append(robot.pos)
                pathLength += np.sqrt((path[-1][0] - path[-2][0]) ** 2 + (path[-1][1] - path[-2][1]) ** 2)
                draw_path(screen, path, ORANGE)

                if robot.checkCollision(obstacles_list) or pathLength > 5 * distanceStartGoal:
                    robot.setDiscount()
                    success_counter = 0
                    consecutive_successes = 0  # Đặt lại nếu thất bại
                    finished = True

                draw_start(screen, start)
                draw_target(screen, goal)
                robot.draw(screen)

                if not isTraining:
                    distanceToObstacle.append(robot.getDistanceToClosestObstacle(obstacles_list))
                    time.sleep(0.02)

                if robot.reach(goal):
                    robot.setSuccess()
                    success_counter += 1
                    consecutive_successes += 1  # Tăng số lần thành công liên tiếp
                    finished = True

        draw_grid(screen, cell_size, env_size, env_padding)
        pygame.draw.rect(screen, BLACK, (LEFT_PAD, NORTH_PAD, env_width, env_height), 3)
        pygame.display.update()

    for obstacle in obstacles_list:
        obstacle.reset()

if __name__ == '__main__':
    epochs = 6000 if version == "6" else 800

    if isTraining:
        for i in range(numsOfRuns):
            print(f"Run {i + 1}")
            if i > 0 and version == "6":
                break
            for j in range(epochs):
                path.clear()
                pathLength = 0
                main(scenario + input_map)

                # Học sớm: Kiểm tra số lần thành công liên tiếp
                if version == "6" and consecutive_successes >= success_threshold and j > 700:
                    print(f"Early stopping at epoch {j + 1}: Achieved {consecutive_successes} consecutive successes")
                    # robot.outputPolicy(scenario, scenario + input_map, i + 1)
                    break  # Thoát vòng lặp epochs

            robot.outputPolicy(scenario, scenario + input_map, i + 1)
            robot.resetController()
    else:
        for i in range(numsOfRuns):
            if version == "6":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                decision_maker = Controller(cell_size=cell_size, env_size=env_size, env_padding=env_padding, goal=goal)
                policy_net = QNetwork(decision_maker.state_dim, decision_maker.action_dim).to(device=device)
                policy_file = f"policy/{scenario}/{scenario + input_map}/DeepQL/{i + 1}/policy_net.pth"
                if os.path.exists(policy_file):
                    policy_net.load_state_dict(torch.load(policy_file, map_location=device))
                    policy_net.eval()
                    decision_maker.policy_net = policy_net
                    robot.decisionMaker = decision_maker
                else:
                    print(f"Error: Policy file {policy_file} not found. Please train the model first.")
                    sys.exit()
            elif version != "5":
                robot.decisionMaker = ControllerTester(cell_size=cell_size, env_size=env_size, env_padding=env_padding,
                                                       goal=goal, scenario=scenario, current_map=input_map,
                                                       algorithm=algorithm, run=i + 1)

            main(scenario + input_map)

            print(f"Run {i + 1}")
            print(f"Path length: {pathLength}")
            print(f"Average turning angle: {get_sum_turning_angle(path)}")
            print(f"Average distance to obstacle: {np.mean(distanceToObstacle)}")
            print("--------------------")

            directory = f"result/{scenario}/{scenario + input_map}/{algorithm}"
            os.makedirs(directory, exist_ok=True)

            with open(f"{directory}/metric.txt", "a") as f:
                f.write(scenario + input_map + ": " + "version" + str(version) + " " + str(pathLength) + " "
                        + str(get_sum_turning_angle(path)) + " " + str(np.mean(distanceToObstacle)))
                if success_counter == 0:
                    f.write(" Fail")
                f.write("\n")

            with open(f"{directory}/path.txt", "a") as f:
                f.write(str(path))
                f.write("\n")

            time.sleep(3)

            path.clear()
            pathLength = 0
            distanceToObstacle.clear()
            success_counter = 0
            consecutive_successes = 0  # Đặt lại sau mỗi run

    pygame.quit()