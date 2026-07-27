from crosswalk_env.common import is_overlapping, Car, Road, CAR_CANDIDATES, CARS_PER_LANE_PAIR_COMPOSITION

class Cars:
    def __init__(self, agent, elements, road_uid_to_cars_dict):
        self.agent = agent
        self.elements = elements
        self.road_uid_to_cars_dict = road_uid_to_cars_dict

    def update_positions(self, dt):
        for car in self.elements:
            car.update_target(dt)
            car.update_position(dt)

    def render(self, background):
        for car in self.elements:
            car.render(background)

    @staticmethod
    def generate_cars(agent, roads, pix_square_size, map_grid_width, steps_per_second, car_details_dict, random):
        cars = []
        car_uid = 0
        road_uid_to_cars_dict = {}
        for road in roads.elements:
            road_uid_to_cars_dict[road.uid] = []
            road_height = road.bottom_y - road.top_y + 1
            for i in range(0, road_height, Road.COMPOSITION_SIZE):
                going_right = road.going_right[i]
                cars_per_lane_pair = random.choice(CARS_PER_LANE_PAIR_COMPOSITION)
                for j in cars_per_lane_pair.keys():
                    start_row_idx = road.top_y + i + j
                    for height in cars_per_lane_pair[j]:
                        car_uid += 1
                        (car_name, _) = random.choice(CAR_CANDIDATES[height])
                        # NOTE: try to reduce overlap between cars
                        overlapped = False
                        for k in range(3):
                            initial_x = random.integers(0, map_grid_width)
                            car_details = car_details_dict[car_name][road.car_color_type][going_right]
                            new_car = Car(car_uid, initial_x, start_row_idx, height, road, map_grid_width, pix_square_size, steps_per_second, car_details, random)
                            for same_road_car in road_uid_to_cars_dict[road.uid]:
                                overlapped = Cars._check_overlapping(same_road_car, new_car)
                                if overlapped: break
                            if not overlapped: break
                        cars.append(new_car)
                        road_uid_to_cars_dict[road.uid].append(new_car)

        for i in range(len(cars)):
            cur_car = cars[i]
            top_y1, bottom_y1 = cur_car.get_cur_y_pos()
            for j in range(len(cars)):
                if i == j: continue
                other_car = cars[j]
                top_y2, bottom_y2 = other_car.get_cur_y_pos()
                if max(top_y1, top_y2) <= min(bottom_y1, bottom_y2):
                    cur_car.nearby_cars.append(other_car) # overlapping lane
        return Cars(agent, cars, road_uid_to_cars_dict)

    @staticmethod
    def _check_overlapping(car1, car2):
        l1, r1 = car1.get_cur_x_pos()
        t1, b1 = car1.get_cur_y_pos()
        l2, r2 = car2.get_cur_x_pos()
        t2, b2 = car2.get_cur_y_pos()
        if not is_overlapping(l1, r1, l2, r2):
            return False
        if not is_overlapping(t1, b1, t2, b2):
            return False
        return True

    def __str__(self):
        string = ""
        for car in self.elements:
            string += f"{car}\n"
        return string[:-1]
