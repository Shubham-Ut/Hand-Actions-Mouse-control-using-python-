import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# ---------------- SETTINGS ----------------
MODEL_PATH = "hand_landmarker.task"

SMOOTH = 5
SENSITIVITY = 1.3
DEAD_ZONE = 3

SCREEN_W, SCREEN_H = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ---------------- MediaPipe Tasks ----------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_x, prev_y = SCREEN_W/2, SCREEN_H/2
last_click = 0

# ---------------- Smooth Function ----------------
def smooth(prev, target, factor):
    return prev + (target - prev) / factor

# ---------------- MAIN ----------------
with HandLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        ts = int(time.time()*1000)
        result = landmarker.detect_for_video(mp_img, ts)

        if result.hand_landmarks:

            lm = result.hand_landmarks[0]

            # -------- INDEX TIP FOR CURSOR --------
            index_tip = lm[8]

            cam_x = index_tip.x * w
            cam_y = index_tip.y * h

            # Map to screen
            screen_x = np.interp(cam_x, [0, w], [0, SCREEN_W])
            screen_y = np.interp(cam_y, [0, h], [0, SCREEN_H])

            # Sensitivity
            screen_x = prev_x + (screen_x - prev_x) * SENSITIVITY
            screen_y = prev_y + (screen_y - prev_y) * SENSITIVITY

            # Dead zone
            if abs(screen_x - prev_x) < DEAD_ZONE:
                screen_x = prev_x
            if abs(screen_y - prev_y) < DEAD_ZONE:
                screen_y = prev_y

            # Smooth
            curr_x = smooth(prev_x, screen_x, SMOOTH)
            curr_y = smooth(prev_y, screen_y, SMOOTH)

            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            # -------- PINCH CLICK --------
            thumb_tip = lm[4]

            pinch_dist = abs(index_tip.x - thumb_tip.x) + abs(index_tip.y - thumb_tip.y)

            if pinch_dist < 0.05 and time.time() - last_click > 0.5:
                pyautogui.click()
                last_click = time.time()

            # -------- TWO FINGER RIGHT CLICK --------
            middle_tip = lm[12]
            pinch2 = abs(middle_tip.x - thumb_tip.x) + abs(middle_tip.y - thumb_tip.y)

            if pinch2 < 0.05 and time.time() - last_click > 0.5:
                pyautogui.rightClick()
                last_click = time.time()

            cv2.circle(frame, (int(cam_x), int(cam_y)), 8, (0,255,0), -1)

        cv2.imshow("Hand Mouse PRO", frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
