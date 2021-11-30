import logging
import math
import random
from collections import deque
from typing import Tuple, List, Dict, Optional, Deque

import numpy as np
import sympy
from scipy.spatial import KDTree
from shapely.geometry import Polygon, Point, LineString

import constants

SAMPLE_LIMIT = 1000  # approx. count of sampled points

RANDOM_COUNT = 50  # repeat times of sampling normal distributions
EVALUATE_SAMPLE = 20  # checking whether rolling part inside polygon with fixed interval

EPS = 1e-6


def p2t(p: sympy.Point2D) -> Point:
    return Point(float(p.x), float(p.y))


def sgn(x: float) -> int:
    if math.fabs(x) < EPS:
        return 0
    return 1 if x > 0 else -1


def dot(a: Point, b: Point) -> float:
    return a.x * b.x + a.y * b.y


def det(a: Point, b: Point) -> float:
    return a.x * b.y - a.y * b.x


def cross(s: Point, t: Point, o: Point = Point(0, 0)) -> float:
    return det(s - o, t - o)


def to_numeric_point(p: sympy.Point2D) -> Point:
    return Point(p.x, p.y)


def dist2(p1: Point, p2: Point) -> float:
    return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2


def dist(p1: Point, p2: Point = Point(0, 0)) -> float:
    return math.sqrt(dist2(p1, p2))


def dist_to_line(p: Point, s: Point, t: Point) -> float:
    return math.fabs(cross(s, t, p)) / dist(s - t)


def dist_to_seg(p: Point, s: Point, t: Point) -> float:
    if s == t:
        return dist(p, s)
    vs = p - s
    vt = p - t
    if sgn(dot(t - s, vs)) < 0:
        return dist(vs)
    elif sgn(dot(t - s, vt)) > 0:
        return dist(vt)
    return dist_to_line(p, s, t)


def sample_points_inside_polygon(poly: Polygon) -> Tuple[float, List[Point]]:
    def sample_by_dist(d: float) -> List[Point]:
        l = list()
        xmin, ymin, xmax, ymax = poly.bounds

        for x in np.arange(float(xmin), float(xmax), d):
            for y in np.arange(float(ymin), float(ymax), d):
                p = Point(x, y)
                if poly.contains(p):
                    l.append(p)
        return l

    dist = 20
    while True:
        s = sample_by_dist(dist)
        if len(s) > SAMPLE_LIMIT:
            return dist, s
        dist /= math.sqrt(2)  # increase the number of sampled points by 2


class Player:
    def __init__(self, skill: int, rng: np.random.Generator, logger: logging.Logger) -> None:
        """Initialise the player with given skill.

        Args:
            skill (int): skill of your player
            rng (np.random.Generator): numpy random number generator, use this for same player behvior across run
            logger (logging.Logger): logger use this like logger.info("message")
        """
        self.skill = skill
        self.rng = rng
        self.logger = logger

        self.max_dist = constants.max_dist + self.skill

        self.need_initialization = True

        self.sampled_points = None
        self.sample_dist = None
        self.scores = None

        self.kdt: KDTree = None
        self.golf_map_f = self.target_f = None

    def calc_scores(self, target: Point, max_d: float):
        # naive BFS
        self.sampled_points.append(target)
        self.scores = [-1 for _ in range(len(self.sampled_points))]

        queue: Deque[int] = deque([len(self.sampled_points) - 1])
        self.scores[-1] = 0

        while queue:
            cur = queue.popleft()
            cur_score = self.scores[cur]
            for i in range(len(self.sampled_points)):
                if self.scores[i] != -1 or self.sampled_points[cur].distance(self.sampled_points[i]) >= max_d:
                    continue
                self.scores[i] = cur_score + 1
                queue.append(i)

    def fix_scores(self):
        # encourage landing near target
        for i in range(len(self.scores)):
            self.scores[i] += self.target_f.distance(self.sampled_points[i]) / 1e10

    def initialize(self, golf_map: sympy.Polygon, target: sympy.Point2D):
        self.need_initialization = False

        # for numeric computation
        self.golf_map_f = Polygon(map(p2t, golf_map.vertices))
        self.target_f = p2t(target)

        # sample points
        self.sample_dist, self.sampled_points = sample_points_inside_polygon(self.golf_map_f)
        self.logger.debug(f"# of sampled points: {len(self.sampled_points)}")

        # calculate scores
        self.calc_scores(self.target_f, self.max_dist)
        self.fix_scores()

        # build KD-Tree
        self.kdt = KDTree([(p.x, p.y) for p in self.sampled_points[:-1]])

        self.logger.debug(f"max score: {max(self.scores)}")

    def score(self, p: Point):
        # return the score of nearest point
        d, idx = self.kdt.query([p.x, p.y], k=1)
        if d > self.sample_dist:
            return float('inf')
        return self.scores[idx]

    @staticmethod
    def pos(current_position: Point, distance: float, angle: float) -> Point:
        return Point(current_position.x + distance * math.cos(angle),
                     current_position.y + distance * math.sin(angle))

    def evaluate_putter(self, current_position: Point, distance: float, angle: float) -> Optional[float]:
        # boundary check
        for i in range(EVALUATE_SAMPLE + 1):
            position = self.pos(current_position, distance * (i / EVALUATE_SAMPLE), angle)
            if not self.golf_map_f.contains(position):
                return None

        end = self.pos(current_position, distance, angle)

        # goal
        if dist_to_seg(self.target_f, current_position, end) < constants.target_radius:
            return 0

        return self.score(end)

    def evaluate(self, current_position: Point, distance: float, angle: float) -> Optional[float]:
        return self.evaluate_putter(self.pos(current_position, distance, angle),
                                    distance * constants.extra_roll,
                                    angle)

    def simulate(self, candidates: List[Tuple[float, float]], current_position: Point) -> Tuple[Tuple[float, float], float]:
        min_score = float('inf')

        def get_score(scores: List[float]) -> float:
            n = len(scores)
            valid_scores = list(filter(lambda x: x is not None, scores))
            if not valid_scores:
                return float('inf')
            avg = sum(valid_scores) / len(valid_scores)
            miss_prob = (n - len(valid_scores)) / n
            return avg + miss_prob / (1 - miss_prob)

        def simulate_one(candidate: Tuple[float, float]) -> Optional[float]:
            distance, angle = candidate
            scores = list()
            for counter in range(RANDOM_COUNT):
                # pruning
                if counter == RANDOM_COUNT // 5:
                    if get_score(scores) > min_score * 1.5:
                        return None

                real_distance = np.random.normal(distance, distance / self.skill)
                real_angle = angle + np.random.normal(0, 1 / (2 * self.skill))
                if distance < constants.min_putter_dist:  # putter
                    scores.append(self.evaluate_putter(current_position, real_distance, real_angle))
                else:
                    scores.append(self.evaluate(current_position, real_distance, real_angle))
            return get_score(scores)

        res = None
        for candidate in candidates:
            v = simulate_one(candidate)
            if not v or v > min_score:
                continue
            min_score = v
            res = candidate
        return res, min_score

    def play(self, score: int, golf_map: sympy.Polygon, target: sympy.geometry.Point2D,
             curr_loc: sympy.geometry.Point2D, prev_loc: sympy.geometry.Point2D,
             prev_landing_point: sympy.geometry.Point2D, prev_admissible: bool) -> Tuple[float, float]:
        """Function which based n current game state returns the distance and angle, the shot must be played 

        Args:
            score (int): Your total score including current turn
            golf_map (sympy.Polygon): Golf Map polygon
            target (sympy.geometry.Point2D): Target location
            curr_loc (sympy.geometry.Point2D): Your current location
            prev_loc (sympy.geometry.Point2D): Your previous location. If you haven't played previously then None
            prev_landing_point (sympy.geometry.Point2D): Your previous shot landing location. If you haven't played previously then None
            prev_admissible (bool): Boolean stating if your previous shot was within the polygon limits. If you haven't played previously then None

        Returns:
            Tuple[float, float]: Return a tuple of distance and angle in radians to play the shot
        """
        if self.need_initialization:
            self.initialize(golf_map, target)

        curr_loc = p2t(curr_loc)

        candidates = [
            (distance, angle)
            for distance in list(range(1, 20)) + list(range(20, self.max_dist, 5)) + [self.max_dist]
            for angle in [2 * math.pi * (i / 36) for i in range(36)]
        ]
        random.shuffle(candidates)

        choice, score = self.simulate(candidates, curr_loc)

        # naive method for special case
        distance, _ = choice
        if distance < constants.min_putter_dist:
            return min(distance * 1.1, constants.min_putter_dist), math.atan2(self.target_f.y - curr_loc.y, self.target_f.x - curr_loc.x)

        self.logger.debug(f"last: {p2t(prev_loc) if prev_loc else prev_loc}, target: {target}")
        self.logger.debug(f"choice: {choice}, score: {score}")

        return choice
