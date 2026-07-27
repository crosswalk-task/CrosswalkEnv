from crosswalk_env.common.gameobject.car_details import CarColorType, RiskDetails

class Road:
    COMPOSITION_SIZE = 2

    def __init__(self, uid, top_y, bottom_y, going_right_rows, random, crosswalk = None):
        self.uid = uid
        self.top_y = top_y
        self.bottom_y = bottom_y
        self.going_right = going_right_rows
        self.car_color_type = random.choice([CarColorType.RED, CarColorType.YELLOW, CarColorType.GREEN])
        self.risk_detail = RiskDetails[self.car_color_type]
        self.crosswalk = crosswalk

class CrossWalk:
    RATIO = 0.6
    THRESHOLD_DISTANCE = 0.3

    def __init__(self, uid, col, top_row, end_row):
        self.uid = uid
        self.col = col
        self.left_end = col - 1
        self.right_end = col + 1
        self.top_row = top_row
        self.end_row = end_row
        self.is_active = False

    def get_activation_left_right(self):
        left = self.left_end - self.THRESHOLD_DISTANCE
        right = self.right_end + self.THRESHOLD_DISTANCE
        return left, right

    def __str__(self):
        return f"CrossWalk(active={self.is_active}, x=({self.left_end}, {self.right_end}), y=({self.top_row}, {self.end_row}))"
