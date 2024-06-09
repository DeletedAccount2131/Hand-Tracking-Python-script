####################################################
#                                                  #
#           Hand Tracking Python Script            #
# ------------------------------------------------ #
#                                                  #
#           Author: Deleted_Account2131            #
#                                                  #
####################################################

import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

    def track_hands(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            x = int(landmark.x * frame.shape[1])
                            y = int(landmark.y * frame.shape[0])
                            if idx == self.mp_hands.HandLandmark.INDEX_FINGER_TIP:
                                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red for index finger tip
                            else:
                                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green for other landmarks
                cv2.imshow('Hand Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.close()

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = HandTracker()
    tracker.track_hands()
