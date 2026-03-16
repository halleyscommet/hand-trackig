import cv2
import mediapipe as mp
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

r = None

pairs = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]

FINGER_COLORS = {
    "thumb": (0, 128, 255),  # orange
    "index": (0, 255, 0),  # green
    "middle": (255, 0, 0),  # blue
    "ring": (0, 255, 255),  # yellow
    "pinky": (255, 0, 255),  # magenta
    "palm": (200, 200, 200),  # grey
}

PAIR_COLORS = {
    (0, 1): FINGER_COLORS["thumb"],
    (1, 2): FINGER_COLORS["thumb"],
    (2, 3): FINGER_COLORS["thumb"],
    (3, 4): FINGER_COLORS["thumb"],
    (0, 5): FINGER_COLORS["index"],
    (5, 6): FINGER_COLORS["index"],
    (6, 7): FINGER_COLORS["index"],
    (7, 8): FINGER_COLORS["index"],
    (0, 9): FINGER_COLORS["middle"],
    (9, 10): FINGER_COLORS["middle"],
    (10, 11): FINGER_COLORS["middle"],
    (11, 12): FINGER_COLORS["middle"],
    (0, 13): FINGER_COLORS["ring"],
    (13, 14): FINGER_COLORS["ring"],
    (14, 15): FINGER_COLORS["ring"],
    (15, 16): FINGER_COLORS["ring"],
    (0, 17): FINGER_COLORS["pinky"],
    (17, 18): FINGER_COLORS["pinky"],
    (18, 19): FINGER_COLORS["pinky"],
    (19, 20): FINGER_COLORS["pinky"],
    (5, 9): FINGER_COLORS["palm"],
    (9, 13): FINGER_COLORS["palm"],
    (13, 17): FINGER_COLORS["palm"],
}


def print_result(
    result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    global r
    r = result


def to_px(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))


def distance(landmark1, landmark2):
    return math.sqrt(
        ((landmark2.x - landmark1.x) ** 2) + ((landmark2.y - landmark1.y) ** 2)
    )


def vec3(a, b):
    return (b.x - a.x, b.y - a.y, b.z - a.z)


def norm(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def unit(v):
    n = norm(v)
    if n == 0:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def thumb_extended(hand):
    thumb_seg_1 = unit(vec3(hand[2], hand[3]))
    thumb_seg_2 = unit(vec3(hand[3], hand[4]))
    thumb_dir = unit(vec3(hand[2], hand[4]))

    palm_across = unit(vec3(hand[5], hand[17]))
    palm_forward = unit(vec3(hand[0], hand[9]))

    straight = dot(thumb_seg_1, thumb_seg_2) > 0.75
    out_to_side = abs(dot(thumb_dir, palm_across)) > 0.45
    not_folded_forward = abs(dot(thumb_dir, palm_forward)) < 0.75

    return straight and out_to_side and not_folded_forward


def finger_extended(hand, reference, mcp, tip):
    ref = (hand[reference].x - hand[0].x, hand[reference].y - hand[0].y)
    v = math.sqrt((ref[0] ** 2) + (ref[1] ** 2))
    ref_norm = (ref[0] / v, ref[1] / v)

    mcp_vec = (hand[mcp].x - hand[0].x, hand[mcp].y - hand[0].y)
    tip_vec = (hand[tip].x - hand[0].x, hand[tip].y - hand[0].y)

    mcp_proj = mcp_vec[0] * ref_norm[0] + mcp_vec[1] * ref_norm[1]
    tip_proj = tip_vec[0] * ref_norm[0] + tip_vec[1] * ref_norm[1]

    return tip_proj > mcp_proj


GESTURES = {
    "close": [False, False, False, False, False],
    "open": [True, True, True, True, True],
    "peace": [False, True, True, False, False],
    "thumbs up": [True, False, False, False, False],
}


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=print_result,
)

with HandLandmarker.create_from_options(options) as landmarker:
    cam = cv2.VideoCapture(0)

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, 20, (frame_width, frame_height))

    running = True
    while running:
        ret, frame = cam.read()

        # If the camera drops a frame, frame will be None and everything below
        # will crash — skip the frame instead
        if not ret:
            continue

        frame_timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        if r is not None:
            for hand, handedness in zip(r.hand_landmarks, r.handedness):
                label = handedness[0].display_name
                score = handedness[0].score

                xs = [int(lm.x * frame_width) for lm in hand]
                ys = [int(lm.y * frame_height) for lm in hand]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                padding = 20
                cv2.rectangle(
                    frame,
                    (x1 - padding, y1 - padding),
                    (x2 + padding, y2 + padding),
                    (255, 255, 255),
                    1,
                )

                cv2.putText(
                    frame,
                    f"{label} ({score:.0%})",
                    (x1 - padding, y1 - padding - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

                for (i, j), color in PAIR_COLORS.items():
                    cv2.line(
                        frame,
                        to_px(hand[i], frame_width, frame_height),
                        to_px(hand[j], frame_width, frame_height),
                        color,
                        1,
                    )
                for lm in hand:
                    cv2.circle(
                        frame,
                        to_px(lm, frame_width, frame_height),
                        3,
                        (255, 255, 255),
                        -1,
                    )

                finger_states = [
                    thumb_extended(hand),
                    finger_extended(hand, 9, 6, 8),
                    finger_extended(hand, 9, 10, 12),
                    finger_extended(hand, 9, 14, 16),
                    finger_extended(hand, 9, 18, 20),
                ]

                gesture = None

                for key, check in GESTURES.items():
                    if finger_states == check:
                        cv2.putText(
                            frame,
                            key,
                            (x1 - padding, y2 + padding - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1,
                        )
                        gesture = key

                if gesture is None:
                    cv2.putText(
                        frame,
                        "No Gesture",
                        (x1 - padding, y2 + padding - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )

        out.write(frame)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord("q"):
            running = False

    cam.release()
    out.release()
    cv2.destroyAllWindows()
