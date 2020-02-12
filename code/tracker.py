import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import os.path
import enum

from code.particle_filter import ParticleFilter

class Lines(enum.Enum):
    LEFT_TRY = 0
    LEFT_5M = 1
    LEFT_22M = 2
    LEFT_10M = 3
    HALFWAY = 4
    RIGHT_10M = 5
    RIGHT_22M = 6
    RIGHT_5M = 7
    RIGHT_TRY = 8
    TOP_TOUCH = 9
    TOP_5M = 10
    TOP_15M = 11
    BOTTOM_15M = 12
    BOTTOM_5M = 13
    BOTTOM_TOUCH = 14

class Tracker:

    def __init__(self, video, player_detections, line_annotations):
        self.video = cv2.VideoCapture(video)
        self.detect_file = player_detections
        self.line_file = line_annotations

        self.PHash = cv2.img_hash_PHash().create()
        self.old_hash = None
        # self.pitch
        self.counter = 0
        self.H = None

        self.filters = []

    def get_lines(self, frame):
        hue = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 0]
        vals = np.histogram(hue, bins=180)[0] # Most common hue

        pitch_mask = np.ones_like(hue) * 255
        pitch_mask[hue >= np.argmax(vals) + 5] = 0

        # Mask conversion
        pitch_mask = cv2.morphologyEx(pitch_mask, cv2.MORPH_ERODE, np.ones((20, 20)))
        pitch_mask = cv2.morphologyEx(pitch_mask, cv2.MORPH_CLOSE, np.ones((50, 50))) <= 25

        l = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)[:, :, 0]
        bg = cv2.medianBlur(l, 15)

        edges = l - bg
        edges[edges < 10] = 255
        edges[pitch_mask] = 255
        edges = (edges < 128).astype(float)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5)))

        edges = cv2.Canny(edges.astype(np.uint8), 50, 100, apertureSize=7)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9)))
        lines = cv2.HoughLines(edges, 3, np.pi / 180, 800)

        final = np.zeros((720, 1280))
        if lines is not None:
            for [[x1, y1, x2, y2]] in lines:
                if ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 > 100:
                    cv2.line(final, (x1, y1), (x2, y2), 255, 5)

        final[pitch_mask] = 0
        return final, lines

    def get_equations(self, lines):
        if lines is None:
            return

        out = []

        for [[x1, y1, x2, y2]] in lines:
            if ((x1- x2) ** 2 + (y1 + y2) ** 2) ** 0.5 < 10:
                continue

            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1

            use = True

            for ap, bp, cp, _, _, _, _ in out:
                if (np.isclose(a, ap) or np.isclose(a / ap, 1, 1)) and (np.isclose(b, bp) or np.isclose(b / bp, 1, 1)) and (np.isclose(c, cp) or np.isclose(c / cp, 1, 1)):
                    use = False
                    break

            if use:
                out.append((a, b, c, x1, y1, x2, y2))

        if len(out) > 0:
            return np.vstack(out)

    def find_intersections(self, lines, is_footage):
        intersections = []

        for i in range(len(lines)):
            if i == j:
                continue

            if is_footage:
                t1, a1, b1, c1, _, _, _, _ = lines[i]
                t2, a2, b2, c2, _, _, _, _ = lines[j]
            else:
                try:
                    t1, a1, b1, c1 = lines[i]
                    t2, a2, b2, c2 = lines[j]
                except TypeError:
                    return

            if t1 == t2 or t1 == -1 or t2 == -1:
                continue

            try:
                x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1), (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
            except ZeroDivisionError:
                continue

            x = np.concatenate((np.array([min(t1, t2), max(t1, t2)]), x))
            intersections.append((x))

        if intersections:
            return np.vstack(intersections)
        return

    def draw_lines(self, frame, equations, greyscale=True):
        if equations is None:
            return frame

        for a, b, c, _, _, _, _ in equations:
            try:
                p1 = 0, int(-c / b)
                p2 = 1280, int(-((c + 1280 * a) / b))

                if greyscale:
                    cv2.line(frame, p1, p2, 128, 2)
                else:
                    cv2.line(frame, p1, p2, (0, 255, 0), 4)
            except:
                continue

        return frame

    def get_pitch_equation(self, line):
        # Useful for debugging purposes
        """
        Method used to obtain the pitch equation for the lines provided.
        :param line: A label specifying the line required.
        :return: The equation of the requested line, None if an invalid line
        """
        # Left for reference
        values = {
            0: 'left_try',
            1: 'left_five',
            2: 'left_twenty_two',
            3: 'left_ten',
            4: 'halfway',
            5: 'right_ten',
            6: 'right_twenty_two',
            7: 'right_five',
            8: 'right_try',
            9: 'top_touch',
            10: 'top_5',
            11: 'top_15',
            12: 'bottom_15',
            13: 'bottom_5',
            14: 'bottom_touch',
        }

        if line == 0:
            return (0, 1, 0, -83)
        elif line == 1:
            return (1, 1, 0, -104)
        elif line == 2:
            return (2, 1, 0, -178)
        elif line == 3:
            return (3, 1, 0, -257)
        elif line == 4:
            return (4, 1, 0, -300)
        elif line == 5:
            return (5, 1, 0, -343)
        elif line == 6:
            return (6, 1, 0, -421)
        elif line == 7:
            return (7, 1, 0, -495)
        elif line == 8:
            return (8, 1, 0, -517)
        elif line == 9:
            return (9, 0, 1, -14)
        elif line == 10:
            return (10, 0, 1, -34)
        elif line == 11:
            return (11, 0, 1, -78)
        elif line == 12:
            return (12, 0, 1, -252)
        elif line == 13:
            return (13, 0, 1, -295)
        elif line == 14:
            return (14, 0, 1, -317)

    def get_point_pairs(self, footage_points):
        out = []

        if footage_points is not None:
            for t1, t2, x, y in footage_points:
                res = self.find_intersections([self.get_pitch_equation(t1), self.get_pitch_equation(t2)], False)
                if res is None:
                    continue

                out.append((x, y, res[0, 2], res[0, 3]))

        if len(out) >= 4:
            return True, np.vstack(out)
        return False, None

    def get_detection_positions(self, detections):
        out = np.zeros((len(detections), 2))

        for i, (x1, y1, x2, y2) in enumerate(detections):
            out[i] = (x1 + x2) / 2, (y1 + y2) / 2

        return out.astype(np.float32)

    def translate_points(self, footage_points):
        points = np.zeros((len(footage_points), 2), np.float32)

        for i, r in enumerate(np.insert(footage_points, 2, 1, 1)):
            p_prime = self.H @ r
            p_prime = p_prime[0] / p_prime[2], p_prime[1] / p_prime[2]
            points[i] = p_prime

        return points

    def update_filters(self, scene_changed, points):
        if scene_changed:
            self.filters = []

        if not self.filters:
            for point in points:
                self.filters.append(ParticleFilter(500, point))

            return points
        else:
            dists = np.empty((len(self.filters), len(points)))
            for i, filter in enumerate(self.filters):
                filter.predict()
                dists[i] = np.linalg.norm(points - filter.estimate(), axis=1)

            new_filters = []
            try:
                row, col = linear_sum_assignment(dists)
            except ValueError:
                return self.filters

            for i in range(len(col)):
                if dists[row[i], col[i]] > 25:
                    pass
                    new_filters.append(ParticleFilter(500, points[col[i]]))
                else:
                    filter = self.filters[row[i]]
                    new_filters.append(filter)
                    filter.update(points[col[i]])
                    filter.resample()

            for filter in self.filters:
                if filter not in new_filters and not filter.update_none():
                    new_filters.append(filter)

            self.filters = new_filters

    def get_frame(self, add_lines, add_players, add_translated_points, add_particle_filters):
        ok, frame = self.video.read()

        new_hash = self.PHash.compute(frame)
        changed_scene = False

        if self.old_hash is not None:
            if self.PHash.compare(self.old_hash, new_hash) > 10:
                changed_scene = True

        self.old_hash = new_hash

        #tmp = self.pitch.copy()
