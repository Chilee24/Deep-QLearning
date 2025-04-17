from environment.Obstacle import Obstacle

uniform = [
    {
        "Start": (54, 262),
        "Goal": (518, 262),
        "Obstacles": [
            Obstacle(158.0, 285.5, 32, 31, False, [0, 1.25], x_bound=(0, 0), y_bound=(116, 116)),
            Obstacle(415.0, 286.0, 32, 30, False, [0, -1.5], x_bound=(0, 0), y_bound=(116, 116)),
            Obstacle(286.5, 285.0, 31, 30, False, [0, 1], x_bound=(0, 0), y_bound=(116, 116)),
            Obstacle(286.0, 94.5, 506, 127, True, [0, 0]),
            Obstacle(285.0, 477.0, 508, 126, True, [0, 0])
        ]}, {
        "Start": (342, 518),
        "Goal": (230, 54),
        "Obstacles": [
            Obstacle(110.5, 461.5, 159, 158, True, [0, 0]),
            Obstacle(461.0, 461.5, 158, 158, True, [0, 0]),
            Obstacle(111.5, 111.5, 159, 158, True, [0, 0]),
            Obstacle(460.5, 111.5, 159, 158, True, [0, 0]),
            Obstacle(94.0, 335.0, 66, 64, False, [1.5, 0], x_bound=[0, 384], y_bound=[0, 0]),
            Obstacle(478.5, 238.5, 63, 63, False, [-1, 0], x_bound=[384, 0], y_bound=[0, 0])
        ]}, {
        "Start": (70, 422),
        "Goal": (502, 118),
        "Obstacles": [
            Obstacle(127.0, 222.0, 192, 64, True, [0, 0]),
            Obstacle(383.0, 461.0, 64, 156, True, [0, 0]),
            Obstacle(380.5, 158.5, 35, 37, False, [1, -1], x_bound=[192, 64], y_bound=[64, 192]),
            Obstacle(350.0, 317.5, 34, 33, False, [0.75, -0.75], x_bound=[172, 64], y_bound=[64, 172])
        ]}
]
diverse = [
    {
        "Start": (54, 262),
        "Goal": (518, 262),
        "Obstacles": [
            Obstacle(286.0, 94.5, 506, 127, True, [0, 0]),
            Obstacle(285.0, 477.0, 508, 126, True, [0, 0]),
            Obstacle(159.0, 239.0, 30, 30, False, [1.25, 0], x_bound=(64, 296), y_bound=(0, 0)),
            Obstacle(416.5, 335.0, 27, 30, False, [-1.5, 0], x_bound=(296, 64), y_bound=(0, 0)),
            Obstacle(285.5, 285.5, 25, 25, False, [0, 1], x_bound=(0, 0), y_bound=(116, 116))
        ]}, {
        "Start": (342, 518),
        "Goal": (230, 54),
        "Obstacles": [
            Obstacle(111.5, 461.0, 159, 158, True, [0, 0]),
            Obstacle(460.5, 462.0, 157, 158, True, [0, 0]),
            Obstacle(111.5, 111.5, 159, 158, True, [0, 0]),
            Obstacle(460.5, 111.5, 159, 158, True, [0, 0]),
            Obstacle(94.0, 335.0, 66, 64, False, [1.5, 0], x_bound=[0, 384], y_bound=[0, 0]),
            Obstacle(254.0, 127.0, 64, 66, False, [0, 1], x_bound=[0, 0], y_bound=[0, 368])
        ]}, {
        "Start": (70, 422),
        "Goal": (502, 118),
        "Obstacles": [
            Obstacle(127.0, 222.0, 192, 64, True, [0, 0]),
            Obstacle(383.0, 461.0, 64, 156, True, [0, 0]),
            Obstacle(302.0, 493.5, 34, 33, False, [0, -1], x_bound=[0, 0], y_bound=[384, 0]),
            Obstacle(173.5, 333.5, 33, 33, False, [1, 0], x_bound=[64, 288], y_bound=[0, 0]),
            Obstacle(397.5, 207.0, 33, 32, False, [1, -1], x_bound=[256, 128], y_bound=[128, 256])
        ]}
]
complex = [
    {
        "Start": (54, 262),
        "Goal": (518, 262),
        "Obstacles": [
            Obstacle(286.0, 94.5, 506, 127, True, [0, 0]),
            Obstacle(285.0, 477.0, 508, 126, True, [0, 0]),
            Obstacle(158.0, 382.0, 32, 32, False, [1, 0],
                     followPath=True, path=[(158.0, 382.0), (157.5, 334.0), (169.0, 292.0), (198.0, 250.5), (230.0, 217.0), (271.5, 193.5),
                                            (325.5, 192.5), (378.0, 190.5), (435.5, 191.0), (474.5, 189.0), (510.0, 190.0)]),
            Obstacle(478.0, 383.5, 32, 31, False, [1, 0],
                     followPath=True, path=[(478.0, 383.5), (414.0, 383.0), (365.0, 382.5), (328.5, 364.5), (320.5, 325.5), (315.0, 273.5),
                                            (309.0, 231.0), (292.0, 206.5), (248.0, 192.5), (196.5, 190.0), (149.5, 192.0), (107.5, 191.5),
                                            (61.0, 190.5)]),
            Obstacle(463.0, 269.5, 34, 33, False, [1, 0],
                     followPath=True, path=[(463.0, 269.5), (396.5, 271.0), (336.0, 270.0), (280.0, 290.0), (249.0, 324.5), (224.5, 356.5),
                                            (187.0, 380.0), (126.0, 382.0), (77.0, 382.5)])
        ]
    }, 

    {
        "Start": (342, 518),
        "Goal": (230, 54),
        "Obstacles": [
            Obstacle(111.5, 461.0, 159, 158, True, [0, 0]),
            Obstacle(460.5, 462.0, 157, 158, True, [0, 0]),
            Obstacle(111.5, 111.5, 159, 158, True, [0, 0]),
            Obstacle(460.5, 111.5, 159, 158, True, [0, 0]),
            Obstacle(494.5, 240.5, 65, 65, False, [1.5, 0],
                     followPath=True, path=[(494.5, 240.5), (407.0, 237.5), (335.5, 254.0), (288.5, 291.0), (252.0, 345.5), (237.0, 396.5),
                                            (237.0, 434.5), (238.0, 493.5)]),
            Obstacle(238.0, 110.0, 66, 64, False, [0.75, 0],
                     followPath=True, path=[(238.0, 110.0), (237.0, 159.0), (239.5, 186.0), (255.5, 218.0), (276.0, 252.0), (300.5, 284.5),
                                            (332.0, 314.5), (373.0, 334.5), (413.5, 335.0), (451.5, 335.0), (493.0, 334.0)]),
        ]
    },

    {
        "Start": (70, 422),
        "Goal": (502, 118),
        "Obstacles": [
            Obstacle(127.0, 222.0, 192, 64, True, [0, 0]),
            Obstacle(383.0, 461.0, 64, 156, True, [0, 0]),
            Obstacle(206.0, 510.5, 32, 33, False, [1, 0],
                     followPath=True, path=[(206.0, 510.5), (222.0, 452.0), (233.0, 387.0), (251.5, 321.5), (254.5, 270.5), (254.5, 212.5),
                                            (243.0, 162.0), (196.0, 122.0), (132.0, 103.0), (78.5, 78.5)]),
            Obstacle(430.5, 62.5, 33, 33, False, [1.15, 0],
                     followPath=True, path=[(430.5, 62.5), (413.5, 117.0), (399.5, 170.5), (375.5, 214.0), (345.5, 254.0), (317.0, 301.0),
                                            (273.0, 340.5), (225.0, 388.0), (184.5, 422.5), (136.5, 470.0), (110.0, 509.5)]),
            Obstacle(494.0, 478.5, 34, 33, False, [0.9, 0],
                     followPath=True, path=[(494.0, 478.5), (469.0, 424.0), (440.5, 377.0), (401.5, 331.0), (334.5, 300.5), (278.0, 289.0),
                           (221.0, 286.0), (157.0, 286.0), (62.0, 284.5)])
        ]
    }
]

map1 = [
    {
    "Start": (50, 50),  # Tọa độ bắt đầu (có thể thay đổi tùy ý)
    "Goal": (450, 450),  # Tọa độ mục tiêu (có thể thay đổi tùy ý)
    "Obstacles": [
        Obstacle(100, 100, 40, 40, True, [0, 0]),  # Không cần thay đổi vì width = height
        Obstacle(200, 100, 40, 40, True, [0, 0]),
        Obstacle(300, 100, 40, 40, True, [0, 0]),
        Obstacle(400, 100, 40, 40, True, [0, 0]),
        Obstacle(100, 200, 40, 40, True, [0, 0]),
        Obstacle(200, 200, 40, 40, True, [0, 0]),
        Obstacle(300, 200, 40, 40, True, [0, 0]),
        Obstacle(400, 200, 40, 40, True, [0, 0]),
        Obstacle(100, 300, 40, 40, True, [0, 0]),
        Obstacle(200, 300, 40, 40, True, [0, 0]),
        Obstacle(300, 300, 40, 40, True, [0, 0]),
        Obstacle(400, 300, 40, 40, True, [0, 0]),
        Obstacle(100, 400, 40, 40, True, [0, 0]),
        Obstacle(200, 400, 40, 40, True, [0, 0]),
        Obstacle(300, 400, 40, 40, True, [0, 0]),
        Obstacle(400, 400, 40, 40, True, [0, 0]),
    ]
}
]

def scale_map_coordinates(file_path, scale_factor):
    scaled_obstacles = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        map_width, map_height = map(int, lines[0].split())  # Kích thước map gốc
        num_obstacles = int(lines[1])  # Số lượng obstacles

        index = 2
        for _ in range(num_obstacles):
            num_vertices = int(lines[index])  # Số đỉnh của obstacle
            index += 1
            vertices = []
            for _ in range(num_vertices):
                x, y = map(int, lines[index].split())
                scaled_x = int(x * scale_factor)
                scaled_y = int(y * scale_factor)
                vertices.append((scaled_x, scaled_y))
                index += 1
            scaled_obstacles.append(vertices)

    return scaled_obstacles

# Scale các tọa độ trong map1.txt
file_path = "c:\\Users\\hieuh\\Downloads\\project\\Deep-QLearning\\map1.txt"
scale_factor = 512 / 500  # Tỷ lệ scale
scaled_obstacles = scale_map_coordinates(file_path, scale_factor)

# Tạo danh sách obstacles cho MapData.py
scaled_map1 = [
    {
    "Start": (int(50 * scale_factor), int(50 * scale_factor)),  # Scale tọa độ Start
    "Goal": (int(450 * scale_factor), int(450 * scale_factor)),  # Scale tọa độ Goal
    "Obstacles": 
        [
        Obstacle(
            x=(vertices[0][0] + vertices[2][0]) // 2,  # Tọa độ trung tâm x
            y=(vertices[0][1] + vertices[2][1]) // 2,  # Tọa độ trung tâm y
            width=abs(vertices[0][0] - vertices[2][0]),  # Chiều rộng
            height=abs(vertices[0][1] - vertices[2][1]),  # Chiều cao
            static=True,
            v=[0, 0]
        )
        for vertices in scaled_obstacles
        ]
    }
]

maps = {}
maps.update({"uniform" + str(i + 1): uniform for i, uniform in enumerate(uniform)})
maps.update({"diverse" + str(i + 1): diverse for i, diverse in enumerate(diverse)})
maps.update({"complex" + str(i + 1): complex for i, complex in enumerate(complex)})
maps.update({"map11": map1})
