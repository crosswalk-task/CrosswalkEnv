import os

from enum import Enum

import pygame

class Penalty(Enum):
    LOW = 100
    MEDIUM = 500
    HIGH = 1000

class CarColorType(Enum):
    GREEN = 0
    YELLOW = 1
    RED = 2

    def __str__(self):
        string_map = {
            CarColorType.GREEN: "GREEN",
            CarColorType.YELLOW: "YELLOW",
            CarColorType.RED: "RED",
        }
        return string_map[self]

class RiskDetail:
    def __init__(self, color_name, color, penalty):
        self.color_name = color_name
        self.color = color
        self.penalty = penalty

RiskDetails = {
    CarColorType.RED: RiskDetail(color_name="Red", color=(255, 0, 0), penalty=Penalty.HIGH),
    CarColorType.YELLOW: RiskDetail(color_name="Yellow", color=(255, 255, 0), penalty=Penalty.MEDIUM),
    CarColorType.GREEN: RiskDetail(color_name="Green", color=(0, 255, 0), penalty=Penalty.LOW),
}

# key = start_row_idx, value = car_heights 
CARS_PER_LANE_PAIR_COMPOSITION = [
    # 2 small cars
    {0: [1], 1: [1]},
    {0: [1], 1: [1]},
    # 3 small cars
    {0: [1], 1: [1, 1]},
    {0: [1, 1], 1: [1]},
    # 1 big car, 1 small car
    {0: [2, 1], 1: []},
    {0: [2], 1: [1]},
    # 1 big car, 2 small cars
    {0: [2, 1], 1: [1]},
    {0: [2, 1], 1: [1]},
]

# key=height, value=(car_type, height:width ratio)
CAR_CANDIDATES = {
    1: [
        ("MICRO", 1.1875),
        ("JEEP", 1.3043),
        ("CIVIC", 1.5556),
        ("VAN", 1.6136),
        ("SUV", 1.8333),
        ("MINIVAN", 1.9070),
        ("WAGON", 1.9545),
        ("HatchBack", 1.9750),
        ("SPORT", 1.9756),
        ("COUPE", 2.0238),
        ("MUSCLECAR", 2.0244),
        ("LUXURY", 2.0698),
        ("SEDAN", 2.0750),
        ("SUPERCAR", 2.2895),
        ("LIMO", 3.2250),
    ],
    2: [
        ("CAMPER", 1.8868),
        ("MEDIUMTRUCK", 1.8983),
        ("BOXTRUCK", 2.0600),
        ("PICKUP", 2.1250),
        # ("BUS", 2.2540), # Too white
    ]
}

def get_max_car_grid_width():
    max_width = 0
    for height in CAR_CANDIDATES:
        for candidate in CAR_CANDIDATES[height]:
            max_width = max(max_width, candidate[1] * height)
    return max_width

def get_max_car_grid_height():
    max_height = 0
    for height in CAR_CANDIDATES:
        max_height = max(max_height, height)
    return max_height

class CarDetails:
    def __init__(self, car_type, color_type, go_right, pix_square_size, car_grid_height, ratio, render_sprite, height_buffer = 0.1):
        self.car_type = car_type
        risk_detail = RiskDetails[color_type]
        self.color = color_type
        self.penalty = risk_detail.penalty
        self.go_right = go_right

        self.car_grid_height = car_grid_height - height_buffer # NOTE: needed to handle the overlap between the lanes
        self.car_grid_width = self.car_grid_height * float(ratio)
        self.car_width = self.car_grid_width * pix_square_size
        self.car_height = self.car_grid_height * pix_square_size

        self.image = None
        if render_sprite:
            direction = "EAST" if go_right else "WEST"
            file_name = f"{risk_detail.color_name}_{car_type}_CLEAN_{direction}_000_cropped.png"
            current_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(current_dir, "../..", "sprites/cars/", file_name)
            object_image = pygame.image.load(image_path)
            self.image = pygame.transform.scale(object_image, (self.car_width, self.car_height))

def load_car_details_dict(pix_square_size, render_sprite):
    car_details_dict = {}
    for height in [1, 2]:
        for i in range(len(CAR_CANDIDATES[height])):
            (car_type, ratio) = CAR_CANDIDATES[height][i]
            car_details_dict[car_type] = {}
            for color_type in RiskDetails.keys():
                car_details_dict[car_type][color_type] = {}
                for go_right in [False, True]:
                    car_details_dict[car_type][color_type][go_right] = CarDetails(car_type, color_type, go_right, pix_square_size, height, ratio, render_sprite)
    return car_details_dict
