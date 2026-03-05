import cv2
import mediapipe as mp
import numpy as np


# ---------- UI CONFIG ----------
WINDOW_NAME = "Air Paint (q to quit, c to clear)"
COLORS = [
    (255, 0, 255),   # purple
    (255, 0, 0),     # blue
    (0, 255, 0),     # green
    (0, 255, 255),   # yellow
    (0, 0, 255),     # red
    (255, 255, 255), # eraser (white)
]
COLOR_NAMES = ["Purple", "Blue", "Green", "Yellow", "Red", "Eraser"]
BRUSH_SIZES = [4, 8, 14, 22]


class HandDetector:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.drawer = mp.solutions.drawing_utils

    def find_landmarks(self, frame):
        """Return list of landmarks (x, y) in pixel coords for first detected hand."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        hand_landmarks = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.drawer.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
            )

        if hand_landmarks is None:
            return None

        h, w, _ = frame.shape
        lm = []
        for p in hand_landmarks.landmark:
            lm.append((int(p.x * w), int(p.y * h)))
        return lm


def fingers_up(lm):
    """Simple heuristic: returns [thumb, index, middle, ring, pinky] as 0/1."""
    if lm is None:
        return [0, 0, 0, 0, 0]

    up = [0, 0, 0, 0, 0]
    # Thumb (x based; rough assumption for mirrored webcam)
    up[0] = 1 if lm[4][0] > lm[3][0] else 0

    # Other fingers (y based)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for i, (t, p) in enumerate(zip(tips, pips), start=1):
        up[i] = 1 if lm[t][1] < lm[p][1] else 0

    return up


def draw_toolbar(frame, selected_color_idx, brush_idx):
    h, w, _ = frame.shape
    toolbar_h = 90

    # Toolbar background
    cv2.rectangle(frame, (0, 0), (w, toolbar_h), (40, 40, 40), -1)

    # Color buttons
    btn_w = max(70, w // (len(COLORS) + 2))
    for i, c in enumerate(COLORS):
        x1 = 10 + i * btn_w
        x2 = x1 + btn_w - 10
        y1, y2 = 12, 50
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, -1)
        if i == selected_color_idx:
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)

    # Brush size buttons
    base_x = 10
    by1, by2 = 58, 82
    for i, size in enumerate(BRUSH_SIZES):
        x1 = base_x + i * 90
        x2 = x1 + 80
        cv2.rectangle(frame, (x1, by1), (x2, by2), (80, 80, 80), -1)
        text = f"{size}px"
        cv2.putText(frame, text, (x1 + 12, by2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        if i == brush_idx:
            cv2.rectangle(frame, (x1 - 2, by1 - 2), (x2 + 2, by2 + 2), (255, 255, 255), 2)

    return toolbar_h


def hit_test_toolbar(x, y, frame_w):
    """Returns (color_idx|None, brush_idx|None)."""
    color_idx = None
    brush_idx = None

    btn_w = max(70, frame_w // (len(COLORS) + 2))

    # Color row
    if 12 <= y <= 50:
        for i in range(len(COLORS)):
            x1 = 10 + i * btn_w
            x2 = x1 + btn_w - 10
            if x1 <= x <= x2:
                color_idx = i
                return color_idx, brush_idx

    # Brush row
    if 58 <= y <= 82:
        for i in range(len(BRUSH_SIZES)):
            x1 = 10 + i * 90
            x2 = x1 + 80
            if x1 <= x <= x2:
                brush_idx = i
                return color_idx, brush_idx

    return color_idx, brush_idx


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру. Проверь доступ к webcam.")

    detector = HandDetector()

    selected_color = 1
    brush_idx = 1
    prev_point = None

    canvas = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

        toolbar_h = draw_toolbar(frame, selected_color, brush_idx)
        lm = detector.find_landmarks(frame)

        if lm:
            idx_tip = lm[8]
            mid_tip = lm[12]
            up = fingers_up(lm)

            # Selection mode: index + middle up
            if up[1] and up[2]:
                prev_point = None
                cv2.circle(frame, idx_tip, 10, (255, 255, 255), cv2.FILLED)

                c_idx, b_idx = hit_test_toolbar(idx_tip[0], idx_tip[1], w)
                if c_idx is not None:
                    selected_color = c_idx
                if b_idx is not None:
                    brush_idx = b_idx

            # Draw mode: only index up
            elif up[1] and not up[2]:
                x, y = idx_tip
                if y > toolbar_h:
                    if prev_point is None:
                        prev_point = (x, y)

                    color = COLORS[selected_color]
                    thickness = BRUSH_SIZES[brush_idx]
                    cv2.line(canvas, prev_point, (x, y), color, thickness)
                    prev_point = (x, y)

                cv2.circle(frame, idx_tip, 8, COLORS[selected_color], cv2.FILLED)
            else:
                prev_point = None

            # Small helper line between index and middle finger
            cv2.line(frame, idx_tip, mid_tip, (180, 180, 180), 1)

        merged = cv2.addWeighted(frame, 0.45, canvas, 0.55, 0)

        cv2.putText(
            merged,
            f"Color: {COLOR_NAMES[selected_color]} | Brush: {BRUSH_SIZES[brush_idx]}px | c: clear",
            (10, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (30, 30, 30),
            2,
        )

        cv2.imshow(WINDOW_NAME, merged)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("c"):
            canvas[:] = 255

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
