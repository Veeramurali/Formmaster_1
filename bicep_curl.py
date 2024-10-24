import cv2
import numpy as np
import time
from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line

class ProcessFrame:
    def __init__(self, thresholds, flip_frame=False):
        # Set if frame should be flipped or not.
        self.flip_frame = flip_frame

        # self.thresholds
        self.thresholds = thresholds

        # Font type.
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # line type
        self.linetype = cv2.LINE_AA

        # Colors in BGR format.
        self.COLORS = {
            'blue': (0, 127, 255),
            'red': (255, 50, 50),
            'green': (0, 255, 127),
            'light_green': (100, 233, 127),
            'yellow': (255, 255, 0),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'cyan': (0, 255, 255),
            'light_blue': (102, 204, 255)
        }

        self.output_filename = 'output_video.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = None

        # Dictionary to maintain the various landmark features.
        self.dict_features = {}
        self.left_features = {
            'shoulder': 11,
            'elbow': 13,
            'wrist': 15,
        }

        self.right_features = {
            'shoulder': 12,
            'elbow': 14,
            'wrist': 16,
        }

        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0

        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            'state_seq': [],
            'start_inactive_time': time.perf_counter(),
            'start_inactive_time_front': time.perf_counter(),
            'INACTIVE_TIME': 0.0,
            'INACTIVE_TIME_FRONT': 0.0,
            'BICEP_CURL_COUNT': 0,
            'INCORRECT_BICEP_CURL': 0,
            'LOWER_ARMS': False,
            'INCORRECT_POSTURE': False,
            'prev_state': None,
            'curr_state': None
        }

        self.FEEDBACK_ID_MAP = {
            0: ('LOWER YOUR ARMS', 215, (0, 153, 255)),
            1: ('BEND YOUR ELBOWS', 170, (255, 80, 80)),
            2: ('STRAIGHTEN YOUR ARMS', 125, (255, 80, 80))
        }

    def _get_state(self, elbow_angle):
        elbow = None
        if self.thresholds['ELBOW_THRESH'][0] <= elbow_angle <= self.thresholds['ELBOW_THRESH'][1]:
            elbow = 1
        elif self.thresholds['ELBOW_THRESH'][2] <= elbow_angle <= self.thresholds['ELBOW_THRESH'][3]:
            elbow = 2
        return f'e{elbow}' if elbow else None

    def _update_state_sequence(self, state):
        if state == 'e2':
            if (('e1' not in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('e2') == 0)) or \
                    (('e1' in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('e2') == 1)):
                self.state_tracker['state_seq'].append(state)
        elif state == 'e1':
            if (state not in self.state_tracker['state_seq']) and 'e2' in self.state_tracker['state_seq']:
                self.state_tracker['state_seq'].append(state)

    def _show_feedback(self, frame, c_frame, dict_maps, lower_arms_disp):
        if lower_arms_disp:
            draw_text(
                frame,
                'LOWER YOUR ARMS',
                pos=(30, 80),
                text_color=(0, 0, 0),
                font_scale=0.6,
                text_color_bg=(255, 255, 0)
            )

        for idx in np.where(c_frame)[0]:
            draw_text(
                frame,
                dict_maps[idx][0],
                pos=(30, dict_maps[idx][1]),
                text_color=(255, 255, 230),
                font_scale= 0.6,
                text_color_bg=dict_maps[idx][2]
            )

        return frame

    def process(self, frame: np.array, pose):
        play_sound = None

        frame_height, frame_width, _ = frame.shape

        # Process the image.
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks

            nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
            left_shldr_coord, left_elbow_coord, left_wrist_coord = \
                get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
            right_shldr_coord, right_elbow_coord, right_wrist_coord = \
                get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)

            offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord)

            if offset_angle > self.thresholds['OFFSET_THRESH']:
                display_inactivity = False

                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']:
                    self.state_tracker['BICEP_CURL_COUNT'] = 0
                    self.state_tracker['INCORRECT_BICEP_CURL'] = 0
                    display_inactivity = True

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if display_inactivity:
                    # cv2.putText(frame, 'Resetting BICEP_CURL_COUNT due to inactivity!!!', (10, frame_height - 90), self.font, 0.5, self.COLORS['blue'], 2, lineType=self.linetype)
                    play_sound = 'reset_counters'
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['BICEP_CURL_COUNT']),
                    pos=(int(frame_width * 0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['INCORRECT_BICEP_CURL']),
                    pos=(int(frame_width * 0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                )

                draw_text(
                    frame,
                    'CAMERA NOT ALIGNED PROPERLY!!!',
                    pos=(30, frame_height - 60),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                draw_text(
 frame,
                    'OFFSET ANGLE: ' + str(offset_angle),
                    pos=(30, frame_height - 30),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

            else:
                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                dist_l_sh_hip = abs(left_shldr_coord[1] - left_shldr_coord[1])
                dist_r_sh_hip = abs(right_shldr_coord[1] - right_shldr_coord[1])

                shldr_coord = None
                elbow_coord = None
                wrist_coord = None

                if dist_l_sh_hip > dist_r_sh_hip:
                    shldr_coord = left_shldr_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord

                    multiplier = -1

                else:
                    shldr_coord = right_shldr_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord

                    multiplier = 1

                # Join landmarks.
                cv2.line(frame, shldr_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, wrist_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)

                # Plot landmark points
                cv2.circle(frame, shldr_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                cv2.circle(frame, elbow_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                cv2.circle(frame, wrist_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)

                elbow_angle = find_angle(shldr_coord, elbow_coord, wrist_coord)

                current_state = self._get_state(int(elbow_angle))
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)

                if current_state == 'e1':
                    if len(self.state_tracker['state_seq']) == 2 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['BICEP_CURL_COUNT'] += 1
                        play_sound = str(self.state_tracker['BICEP_CURL_COUNT'])

                    elif 'e2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq']) == 1:
                        self.state_tracker['INCORRECT_BICEP_CURL'] += 1
                        play_sound = 'incorrect'

                    elif self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['INCORRECT_BICEP_CURL'] += 1
                        play_sound = 'incorrect'

                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False

                else:
                    if elbow_angle > self.thresholds['ELBOW_THRESH'][1]:
                        self.state_tracker['DISPLAY_TEXT'][0] = True

                    elif elbow_angle < self.thresholds['ELBOW_THRESH'][0] and \
                            self.state_tracker['state_seq'].count('e2') == 1:
                        self.state_tracker['DISPLAY_TEXT'][1] = True

                    if self.thresholds['ELBOW_THRESH'][2] < elbow_angle < self.thresholds['ELBOW_THRESH'][3]:
                        self.state_tracker['LOWER_ARMS'] = True

                    elif elbow_angle > self.thresholds['ELBOW_THRESH'][4]:
                        self.state_tracker['DISPLAY_TEXT'][2] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True

                frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP,
                                          self.state_tracker['LOWER_ARMS'])

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                elbow_text_coord_x = elbow_coord[0] + 15

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    elbow_text_coord_x = frame_width - elbow_coord[0] + 15

                cv2.putText(frame, str(int(elbow_angle)), (elbow_text_coord_x, elbow_coord[1]), self.font, 0.6,
                            self.COLORS['light_green'], 2, lineType=self.linetype)

                draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['BICEP_CURL_COUNT']),
                    pos=(int(frame_width * 0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['INCORRECT_BICEP_CURL']),
                    pos=(int(frame_width * 0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                )

                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0
                self.state_tracker['prev_state'] = current_state

        else:
            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                self.state_tracker['BICEP_CURL_COUNT'] = 0
                self.state_tracker['INCORRECT_BICEP_CURL'] = 0
                # cv2.putText(frame, 'Resetting BICEP_CURL_COUNT due to inactivity!!!', (10, frame_height - 25), self.font, 0.7, self.COLORS['blue'], 2)
                display_inactivity = True

            self.state_tracker['start_inactive_time'] = end_time

            draw_text(
                frame,
                "CORRECT: " + str(self.state_tracker['BICEP_CURL_COUNT']),
                pos=(int(frame_width * 0.68), 30),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(18, 185, 0)
            )

            draw_text(
                frame,
                "INCORRECT: " + str(self.state_tracker['INCORRECT_BICEP_CURL']),
                pos=(int(frame_width * 0.68), 80),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(221, 0, 0),
            )

            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0

            # Reset all other state variables
            self.state_tracker['prev_state'] = None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full((3,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros((3,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()

        return frame, play_sound