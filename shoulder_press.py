import time
import cv2
import numpy as np
from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line

class ProcessShoulderPress:
    def __init__(self, thresholds, flip_frame=False):
        # Set if frame should be flipped or not.
        self.flip_frame = flip_frame
        
        # Set thresholds for angles and inactivity
        self.thresholds = thresholds
        
        # Font type and line type
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.linetype = cv2.LINE_AA
        
        # Define colors in BGR format
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

        # Initialize state tracker for shoulder press
        self.state_tracker = {
            'state_seq': [],
            'start_inactive_time': time.perf_counter(),
            'INACTIVE_TIME': 0.0,
            'DISPLAY_TEXT': np.full((2,), False),  # For feedback
            'COUNT_FRAMES': np.zeros((2,), dtype=np.int64),
            'SHOULDER_PRESS_COUNT': 0,
            'IMPROPER_PRESS': 0,
            'prev_state': None,
            'curr_state': None
        }
        
        # Feedback messages for shoulder press
        self.FEEDBACK_ID_MAP = {
            0: ('RAISE YOUR ARMS HIGHER', 215, (0, 153, 255)),
            1: ('LOWER YOUR ARMS', 170, (255, 80, 80))
        }

    def _get_state(self, shoulder_angle):
        # Determine the state based on shoulder angle
        if shoulder_angle < self.thresholds['SHOULDER_THRESH'][0]:
            return 's1'  # Arms are too low
        elif shoulder_angle > self.thresholds['SHOULDER_THRESH'][1]:
            return 's2'  # Arms are too high
        return None

    def _update_state_sequence(self, state):
        # Update the state sequence for feedback
        if state == 's1' and 's1' not in self.state_tracker['state_seq']:
            self.state_tracker['state_seq'].append(state)
        elif state == 's2' and 's2' not in self.state_tracker['state_seq']:
            self.state_tracker['state_seq'].append(state)

    def _show_feedback(self, frame, c_frame, dict_maps):
        # Display feedback messages on the frame
        for idx in np.where(c_frame)[0]:
            draw_text(
                frame,
                dict_maps[idx][0],
                pos=(30, dict_maps[idx][1]),
                text_color=(255, 255, 230),
                font_scale=0.6,
                text_color_bg=dict_maps[idx][2]
            )
        return frame

    def process(self, frame: np.array, pose):
        play_sound = None
        frame_height, frame_width, _ = frame.shape

        # Process the image to get pose landmarks
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks
            
            # Get coordinates for shoulder, elbow, and wrist
            left_shldr_coord, left_elbow_coord, left_wrist_coord = get_landmark_features(ps_lm.landmark, {'left': { 'shoulder': 11, 'elbow': 13, 'wrist': 15 }}, 'left', frame_width, frame_height)
            right_shldr_coord, right_elbow_coord, right_wrist_coord = get_landmark_features(ps_lm.landmark, {'right': { 'shoulder': 12, 'elbow': 14, 'wrist': 16 }}, 'right', frame_width, frame_height)

            # Calculate the angle at the shoulder
            shoulder_angle = find_angle(left_shldr_coord, left_elbow_coord, right_shldr_coord)

            # Determine the state based on the shoulder angle
            current_state = self._get_state(shoulder_angle)
            self.state_tracker['curr_state'] = current_state
            self._update_state_sequence(current_state)

            # Draw the shoulder angle on the frame
            cv2.ellipse(frame, left_shldr_coord, (30, 30), angle=0, startAngle=-90, endAngle=-90+shoulder_angle, color=self.COLORS['white'], thickness=3, lineType=self.linetype)

            # Draw the arm landmarks
            cv2.line(frame, left_shldr_coord, left_elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
            cv2.line(frame, left_elbow_coord, left_wrist_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
            cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
            cv2.circle(frame, left_elbow_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
            cv2.circle(frame, left_wrist_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)

            # Update the state tracker and display feedback
            if current_state == 's1':
                if len(self.state_tracker['state_seq']) == 2:
                    self.state_tracker['SHOULDER_PRESS_COUNT'] += 1
                    play_sound = str(self.state_tracker['SHOULDER_PRESS_COUNT'])
                elif 's2' in self.state_tracker['state_seq']:
                    self.state_tracker['IMPROPER_PRESS'] += 1
                    play_sound = 'incorrect'
                self.state_tracker['state_seq'] = []
            elif current_state == 's2':
                self.state_tracker['DISPLAY_TEXT'][1] = True

            # Display the shoulder press count and improper press count
            draw_text(
                frame,
                "CORRECT: " + str(self.state_tracker['SHOULDER_PRESS_COUNT']),
                pos=(int(frame_width*0.68), 30),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(18, 185, 0)
            )
            draw_text(
                frame,
                "INCORRECT: " + str(self.state_tracker['IMPROPER_PRESS']),
                pos=(int(frame_width*0.68), 80),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(221, 0, 0)
            )

            # Show feedback messages
            frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP)

        return frame, play_sound