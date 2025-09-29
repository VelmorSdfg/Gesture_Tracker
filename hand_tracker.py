import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from typing import List, Dict, Any, Optional
import time


class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            print("Error: Cannot access camera")
            self.cap = None

        self.last_action_time = 0.0
        self.action_cooldown = 0.5

    @staticmethod
    def _calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        v1 = p1 - p2
        v2 = p3 - p2
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norms == 0:
            return 0.0
        cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return np.degrees(angle_rad)

    def recalculate_coordinates(self, landmarks: List[Any]) -> Optional[Dict[str, Any]]:
        if not landmarks:
            return None

        points_np = np.array([(lm.x, lm.y) for lm in landmarks])
        wrist_point = points_np[0]
        relative_points = points_np - wrist_point

        thumb_direction = "up" if landmarks[4].y < landmarks[2].y else "down"

        angles = {  # все углы
            "angle_thumb_knuckle_1": self._calculate_angle(relative_points[4], relative_points[3], relative_points[2]),
            "angle_thumb_knuckle_2": self._calculate_angle(relative_points[4], relative_points[0], relative_points[5]),
            "angle_index_middle_V": self._calculate_angle(relative_points[8], relative_points[0], relative_points[12]),
            "angle_V_check_1": self._calculate_angle(relative_points[8], relative_points[7], relative_points[6]),
            "angle_V_check_2": self._calculate_angle(relative_points[12], relative_points[11], relative_points[10]),
            "angle_V_check_3": self._calculate_angle(relative_points[16], relative_points[14], relative_points[13]),
            "palm_angle_1": self._calculate_angle(relative_points[20], relative_points[18], relative_points[17]),
            "palm_angle_2": self._calculate_angle(relative_points[8], relative_points[6], relative_points[5]),
            "palm_angle_3": self._calculate_angle(relative_points[12], relative_points[10], relative_points[9]),
            "palm_angle_4": self._calculate_angle(relative_points[16], relative_points[14], relative_points[13]),
        }

        result = {
            "relative_points": relative_points,
            "thumb_direction": thumb_direction,
            **angles,
        }
        return result

    def _control_media(self, gesture: str):
        current_time = time.time()
        if current_time - self.last_action_time < self.action_cooldown:
            return

        action_map = {
            "Thumb": "volumeup",  # Палец вверх -> Повысить громкость
            "Palm": "playpause",  # Ладонь -> Пауза/Воспроизведение
            "Fist": "volumedown",  # Кулак -> Понизить громкость
            "V": "nexttrack"  # Победа -> Следующий трек
        }

        pyautogui.press(action_map.get(gesture))
        print(f"Media Command: {gesture} -> {action_map.get(gesture)}")
        self.last_action_time = current_time

    def process_and_display(self):
        if self.cap is None:
            print("Error: Camera not initialized")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            skeleton = np.zeros(frame.shape, dtype=np.uint8)
            key = cv2.waitKey(1) & 0xFF
            info_lines = []


            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_drawing.draw_landmarks(
                    skeleton, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                data = self.recalculate_coordinates(hand_landmarks.landmark)

                if data:

                    rel_points = data['relative_points']
                    direction = data['thumb_direction']
                    info_lines.append(f"Thumb direction: {rel_points[1]}")

                    palms = [data[f'palm_angle_{i}'] for i in range(1, 5)]
                    info_lines.append(f"Palm Angles: {[f'{p:.1f}' for p in palms]}")
                    info_lines.append(f"Direction: {direction}")

                    current_gesture = ""

                    if ((170 > data['angle_thumb_knuckle_1'] > 130) and (
                            data['angle_thumb_knuckle_2'] > 25) and current_gesture == ""):  # Thumb
                        info_lines.append("Thumb")
                        current_gesture = "Thumb"
                    if all(180 > p > 165 for p in palms) and current_gesture == "":  # Palm
                        info_lines.append("Palm")
                        current_gesture = "Palm"
                    elif ((all(p < 90 for p in palms)) and  (current_gesture == "")):  # Fist
                        info_lines.append("Fist")
                        current_gesture = "Fist"
                    if ((10 < data['angle_index_middle_V'] < 35) and (185 > data['angle_V_check_1'] > 170) and (
                            185 > data['angle_V_check_2'] > 170) and (data['angle_V_check_3'] < 170) and  (current_gesture == "")):  # Victory
                        info_lines.append("Victory")
                        current_gesture = "V"

                    if current_gesture:
                        self._control_media(current_gesture)

            else:
                info_lines.append("No Hand")
            y_pos = 30

            for line in info_lines:
                cv2.putText(frame, line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 25

            cv2.imshow("Webcam", frame)
            cv2.imshow("Skeleton", skeleton)

            if key == ord('q'):
                break

    def cleanup(self):
        if self.cap:
            self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()


def main():
    tracker = HandTracker()
    tracker.process_and_display()
    tracker.cleanup()


if __name__ == "__main__":
    main()