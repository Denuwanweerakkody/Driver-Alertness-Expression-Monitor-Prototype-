"""
Driver Alertness & Expression Monitor
======================================
A prototype desktop application that uses a webcam to monitor a driver's
eyes and facial expression in real time, to help flag drowsiness,
unusual/distracted behaviour, and mood.

IMPORTANT SAFETY DISCLAIMER
----------------------------
This is a hobby / research prototype, NOT a certified or medically
validated safety device. It must never be relied upon as the sole means
of preventing an accident. Real in-cabin Driver Monitoring Systems (DMS)
used in production vehicles go through extensive testing and regulatory
approval (e.g. under UN R159 / EU General Safety Regulation 2). Use this
purely for learning, experimentation, or as a starting point for a more
serious engineering project.

Author: Build by Denuwan
"""

import csv
import math
import os
import threading
import time
import tkinter as tk
import urllib.request
from collections import deque
from datetime import datetime
from tkinter import font as tkfont
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python import BaseOptions as mp_BaseOptions
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# The classic `mp.solutions.face_mesh` API has been removed from current
# MediaPipe releases. We use the modern MediaPipe Tasks API instead, which
# needs a small pre-trained model file downloaded once on first run.
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1)
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False


# --------------------------------------------------------------------------
# Landmark index sets (MediaPipe Face Mesh - 468 point model)
# --------------------------------------------------------------------------
LEFT_EYE = [362, 385, 387, 263, 373, 380]      # p1..p6 for EAR formula
RIGHT_EYE = [33, 160, 158, 133, 153, 144]      # p1..p6 for EAR formula

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

LEFT_EYEBROW_INNER = 105
RIGHT_EYEBROW_INNER = 334
LEFT_EYEBROW_TOP = 66
RIGHT_EYEBROW_TOP = 296

NOSE_TIP = 1
CHIN = 152
LEFT_FACE_EDGE = 234
RIGHT_FACE_EDGE = 454

# --------------------------------------------------------------------------
# Tunable thresholds
# --------------------------------------------------------------------------
EAR_THRESHOLD = 0.21          # below this = eye considered closed
EAR_CONSEC_FRAMES_BLINK = 2   # frames below threshold to count as a blink
DROWSY_EYES_CLOSED_SECONDS = 1.5   # eyes closed continuously -> drowsy alert
YAWN_MAR_THRESHOLD = 0.55
YAWN_CONSEC_FRAMES = 8
DISTRACTION_YAW_THRESHOLD = 0.35   # normalised head-turn considered "looking away"
DISTRACTION_SECONDS = 2.0

ALARM_COOLDOWN_SECONDS = 4.0


def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(landmarks, eye_idx):
    p = [landmarks[i] for i in eye_idx]
    vertical_1 = euclidean(p[1], p[5])
    vertical_2 = euclidean(p[2], p[4])
    horizontal = euclidean(p[0], p[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks):
    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]
    vertical = euclidean(top, bottom)
    horizontal = euclidean(left, right)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def classify_mood(blendshapes, ear_avg):
    """
    Mood classifier driven by MediaPipe's FaceLandmarker blendshape scores
    (52 ARKit-style expression coefficients, 0-1, from Google's pre-trained
    model) rather than hand-built landmark geometry. This is meaningfully
    more accurate/robust than a manual heuristic across different faces,
    angles, and lighting, while still being fully offline and fast.
    """
    def g(name):
        return blendshapes.get(name, 0.0)

    smile = (g("mouthSmileLeft") + g("mouthSmileRight")) / 2.0
    frown = (g("mouthFrownLeft") + g("mouthFrownRight")) / 2.0
    brow_down = (g("browDownLeft") + g("browDownRight")) / 2.0
    brow_inner_up = g("browInnerUp")
    brow_outer_up = (g("browOuterUpLeft") + g("browOuterUpRight")) / 2.0
    jaw_open = g("jawOpen")
    eye_blink = (g("eyeBlinkLeft") + g("eyeBlinkRight")) / 2.0
    eye_wide = (g("eyeWideLeft") + g("eyeWideRight")) / 2.0

    # --- Decision logic (order matters) ---
    if (eye_blink > 0.55 or ear_avg < EAR_THRESHOLD) and jaw_open < 0.25:
        return "Tired / Drowsy"
    if jaw_open > 0.45 and (brow_outer_up > 0.25 or eye_wide > 0.25):
        return "Surprised"
    if jaw_open > 0.35:
        return "Yawning"
    if smile > 0.35:
        return "Happy"
    if frown > 0.25 and brow_inner_up > 0.20:
        return "Sad"
    if brow_down > 0.35:
        return "Angry / Frustrated"
    return "Neutral"


def estimate_head_yaw(landmarks):
    """Rough normalised horizontal head-turn estimate using nose position
    relative to the face bounding box (left/right cheek landmarks)."""
    nose = landmarks[NOSE_TIP]
    left_edge = landmarks[LEFT_FACE_EDGE]
    right_edge = landmarks[RIGHT_FACE_EDGE]
    face_width = euclidean(left_edge, right_edge)
    if face_width == 0:
        return 0.0
    center_x = (left_edge[0] + right_edge[0]) / 2.0
    return (nose[0] - center_x) / face_width


class AlarmPlayer:
    """Generates and plays a beep tone in-memory (no external audio file)."""

    def __init__(self):
        self.available = AUDIO_AVAILABLE
        self._sound = None
        if self.available:
            self._sound = self._build_tone()

    @staticmethod
    def _build_tone(freq=1000, duration_ms=350, volume=0.5):
        sample_rate = 44100
        n_samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, n_samples, False)
        tone = np.sin(freq * t * 2 * np.pi)
        # fade in/out to avoid clicks
        fade = min(200, n_samples // 4)
        envelope = np.ones(n_samples)
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)
        audio = (tone * envelope * volume * 32767).astype(np.int16)

        # The mixer may have initialized in stereo even though we asked for
        # mono (this depends on the OS/audio driver), so shape the array to
        # match whatever it actually settled on to avoid a shape mismatch.
        init = pygame.mixer.get_init()
        channels = init[2] if init else 1
        if channels and channels > 1:
            audio = np.column_stack([audio] * channels)

        return pygame.sndarray.make_sound(audio)

    def play(self):
        if self.available and self._sound is not None:
            try:
                self._sound.play()
            except Exception:
                pass
        else:
            # Fallback: terminal bell
            print("\a", end="", flush=True)


class EventLogger:
    """Writes drowsiness / distraction / mood events to a CSV file."""

    def __init__(self, path="driver_monitor_log.csv"):
        self.path = path
        new_file = not os.path.exists(path)
        self._file = open(path, "a", newline="")
        self._writer = csv.writer(self._file)
        if new_file:
            self._writer.writerow(["timestamp", "event_type", "detail"])
            self._file.flush()

    def log(self, event_type, detail=""):
        self._writer.writerow([datetime.now().isoformat(timespec="seconds"), event_type, detail])
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass


class DriverMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Alertness & Expression Monitor (Prototype)")
        self.root.configure(bg="#12161c")
        self.root.geometry("1080x680")
        self.root.minsize(920, 600)

        self.cap = None
        self.running = False
        self.camera_index = 0

        self.alarm = AlarmPlayer()
        self.logger = EventLogger()

        # State tracking
        self.eyes_closed_since = None
        self.distracted_since = None
        self.last_alarm_time = 0
        self.blink_counter = 0
        self.total_blinks = 0
        self.frame_counter_ear = 0
        self.frame_counter_mar = 0
        self.session_start = None
        self.ear_history = deque(maxlen=30)
        self.mood_history = deque(maxlen=15)
        self.no_face_since = None

        self.landmarker = None      # created lazily once the model is downloaded
        self.model_ready = os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 500_000
        self._video_t0 = None
        self._last_ts_ms = -1

        self._build_ui()

        if not MEDIAPIPE_AVAILABLE:
            self._set_status_banner(
                "mediapipe is not installed. Run: pip install mediapipe", "#e05252"
            )

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _create_landmarker(self):
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def _ensure_model_then_start(self):
        """Downloads the face landmark model (~4 MB, one-time) in a
        background thread if needed, then proceeds to open the camera."""
        if self.model_ready and self.landmarker is not None:
            self._open_camera_and_run()
            return

        self._set_status_banner("Preparing face model (first run only)...", "#2d5f8a")
        self.start_btn.config(state="disabled")

        def worker():
            try:
                if not (os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 500_000):
                    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                self._create_landmarker()
                self.model_ready = True
                self.root.after(0, self._open_camera_and_run)
            except Exception as e:
                self.root.after(
                    0,
                    lambda: (
                        self._set_status_banner(f"Model setup failed: {e}", "#e05252"),
                        self.start_btn.config(state="normal"),
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        bold = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        big_bold = tkfont.Font(family="Segoe UI", size=22, weight="bold")
        normal = tkfont.Font(family="Segoe UI", size=11)
        small = tkfont.Font(family="Segoe UI", size=9)

        header = tk.Frame(self.root, bg="#12161c")
        header.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(
            header, text="🚗  Driver Alertness & Expression Monitor",
            font=big_bold, fg="#f5f7fa", bg="#12161c"
        ).pack(side="left")

        self.banner = tk.Label(
            self.root, text="", font=normal, fg="#12161c", bg="#12161c", pady=6
        )
        self.banner.pack(fill="x", padx=16)

        main = tk.Frame(self.root, bg="#12161c")
        main.pack(fill="both", expand=True, padx=16, pady=8)

        # Left: video feed
        left = tk.Frame(main, bg="#1b212b", bd=0)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.video_label = tk.Label(left, bg="#1b212b", text="Camera preview will appear here",
                                     fg="#8b93a1", font=normal)
        self.video_label.pack(fill="both", expand=True, padx=8, pady=8)

        controls = tk.Frame(left, bg="#1b212b")
        controls.pack(fill="x", pady=(0, 10), padx=8)
        self.start_btn = tk.Button(
            controls, text="▶  Start Monitoring", command=self.start_monitoring,
            bg="#2e7d32", fg="white", activebackground="#256428", relief="flat",
            font=bold, padx=14, pady=8, cursor="hand2"
        )
        self.start_btn.pack(side="left")
        self.stop_btn = tk.Button(
            controls, text="■  Stop", command=self.stop_monitoring,
            bg="#8b1e1e", fg="white", activebackground="#6e1717", relief="flat",
            font=bold, padx=14, pady=8, cursor="hand2", state="disabled"
        )
        self.stop_btn.pack(side="left", padx=8)

        self.session_label = tk.Label(controls, text="Session: 00:00:00", font=normal,
                                       fg="#c7ccd4", bg="#1b212b")
        self.session_label.pack(side="right")

        # Right: status panel
        right = tk.Frame(main, bg="#12161c", width=320)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        def card(parent, title):
            frame = tk.Frame(parent, bg="#1b212b", padx=14, pady=12)
            frame.pack(fill="x", pady=(0, 10))
            tk.Label(frame, text=title, font=small, fg="#8b93a1", bg="#1b212b").pack(anchor="w")
            return frame

        mood_card = card(right, "CURRENT MOOD")
        self.mood_value = tk.Label(mood_card, text="—", font=big_bold, fg="#4fc3f7", bg="#1b212b")
        self.mood_value.pack(anchor="w", pady=(4, 0))

        eye_card = card(right, "EYE STATE")
        self.eye_value = tk.Label(eye_card, text="—", font=bold, fg="#f5f7fa", bg="#1b212b")
        self.eye_value.pack(anchor="w", pady=(2, 2))
        self.ear_value = tk.Label(eye_card, text="EAR: —", font=normal, fg="#c7ccd4", bg="#1b212b")
        self.ear_value.pack(anchor="w")
        self.blink_value = tk.Label(eye_card, text="Blinks: 0", font=normal, fg="#c7ccd4", bg="#1b212b")
        self.blink_value.pack(anchor="w")

        alert_card = card(right, "ALERT STATUS")
        self.alert_value = tk.Label(alert_card, text="All clear", font=bold, fg="#66bb6a", bg="#1b212b",
                                     wraplength=270, justify="left")
        self.alert_value.pack(anchor="w", pady=(2, 0))

        stats_card = card(right, "SESSION STATS")
        self.yawn_value = tk.Label(stats_card, text="Yawns: 0", font=normal, fg="#c7ccd4", bg="#1b212b")
        self.yawn_value.pack(anchor="w")
        self.drowsy_value = tk.Label(stats_card, text="Drowsy alerts: 0", font=normal, fg="#c7ccd4", bg="#1b212b")
        self.drowsy_value.pack(anchor="w")
        self.distract_value = tk.Label(stats_card, text="Distraction alerts: 0", font=normal, fg="#c7ccd4", bg="#1b212b")
        self.distract_value.pack(anchor="w")

        log_card = card(right, "LOG FILE")
        tk.Label(log_card, text=self.logger.path, font=small, fg="#c7ccd4", bg="#1b212b",
                 wraplength=270, justify="left").pack(anchor="w")

        disclaimer = tk.Label(
            self.root,
            text="Prototype only — not a certified safety device. Do not rely on this as your sole "
                 "protection against fatigue while driving. Always pull over and rest if you feel drowsy.",
            font=small, fg="#6f7684", bg="#12161c", wraplength=1040, justify="left"
        )
        disclaimer.pack(fill="x", padx=16, pady=(0, 10))

        self.yawn_count = 0
        self.drowsy_count = 0
        self.distraction_count = 0

    def _set_status_banner(self, text, color):
        self.banner.config(text=text, bg=color, fg="white")

    def _clear_status_banner(self):
        self.banner.config(text="", bg="#12161c")

    # ------------------------------------------------------------------
    # Monitoring lifecycle
    # ------------------------------------------------------------------
    def start_monitoring(self):
        if not MEDIAPIPE_AVAILABLE:
            self._set_status_banner("Cannot start: mediapipe is not installed.", "#e05252")
            return
        if self.running:
            return
        self.start_btn.config(state="disabled")
        self._ensure_model_then_start()

    def _open_camera_and_run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self._set_status_banner(
                "Could not open webcam. Check it is connected and not used by another app.",
                "#e05252",
            )
            self.cap = None
            self.start_btn.config(state="normal")
            return

        self.running = True
        self.session_start = time.time()
        self._video_t0 = time.perf_counter()
        self._last_ts_ms = -1
        self.eyes_closed_since = None
        self.distracted_since = None
        self.no_face_since = None
        self._clear_status_banner()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.logger.log("session_start")
        self._update_frame()
        self._update_session_timer()

    def stop_monitoring(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.logger.log("session_end")
        self.video_label.config(image="", text="Camera preview will appear here", fg="#8b93a1")

    def on_close(self):
        self.stop_monitoring()
        self.logger.close()
        self.root.destroy()

    def _update_session_timer(self):
        if not self.running or self.session_start is None:
            return
        elapsed = int(time.time() - self.session_start)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        self.session_label.config(text=f"Session: {h:02d}:{m:02d}:{s:02d}")
        self.root.after(1000, self._update_session_timer)

    # ------------------------------------------------------------------
    # Main per-frame processing
    # ------------------------------------------------------------------
    def _update_frame(self):
        if not self.running or self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.root.after(30, self._update_frame)
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = max(int((time.perf_counter() - self._video_t0) * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        results = self.landmarker.detect_for_video(mp_image, ts_ms)

        if results.face_landmarks:
            self.no_face_since = None
            landmarks_norm = results.face_landmarks[0]
            landmarks = [(lm.x * w, lm.y * h) for lm in landmarks_norm]

            blendshapes = {}
            if results.face_blendshapes:
                blendshapes = {c.category_name: c.score for c in results.face_blendshapes[0]}

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear_avg = (left_ear + right_ear) / 2.0
            self.ear_history.append(ear_avg)

            mar = mouth_aspect_ratio(landmarks)
            yaw = estimate_head_yaw(landmarks)
            mood = classify_mood(blendshapes, ear_avg)
            self.mood_history.append(mood)
            stable_mood = max(set(self.mood_history), key=self.mood_history.count)

            self._process_eye_state(ear_avg)
            self._process_yawn(mar)
            self._process_distraction(yaw)

            self._draw_overlays(frame, landmarks, ear_avg, mar)
            self._refresh_status_panel(ear_avg, stable_mood)
        else:
            self._handle_no_face()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize(self._fit_size(img.size), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk, text="")

        self.root.after(15, self._update_frame)

    def _fit_size(self, size):
        target_w = max(self.video_label.winfo_width(), 480)
        target_h = max(self.video_label.winfo_height(), 360)
        src_w, src_h = size
        scale = min(target_w / src_w, target_h / src_h)
        return (max(1, int(src_w * scale)), max(1, int(src_h * scale)))

    # ------------------------------------------------------------------
    # Detection logic
    # ------------------------------------------------------------------
    def _process_eye_state(self, ear_avg):
        now = time.time()
        if ear_avg < EAR_THRESHOLD:
            self.frame_counter_ear += 1
            if self.eyes_closed_since is None:
                self.eyes_closed_since = now
            closed_duration = now - self.eyes_closed_since
            if closed_duration >= DROWSY_EYES_CLOSED_SECONDS:
                self._trigger_alert(
                    "DROWSINESS DETECTED — eyes closed too long!",
                    "drowsy_eyes_closed", f"{closed_duration:.1f}s",
                )
        else:
            if self.frame_counter_ear >= EAR_CONSEC_FRAMES_BLINK and self.eyes_closed_since is not None:
                self.total_blinks += 1
            self.frame_counter_ear = 0
            self.eyes_closed_since = None

    def _process_yawn(self, mar):
        if mar > YAWN_MAR_THRESHOLD:
            self.frame_counter_mar += 1
            if self.frame_counter_mar == YAWN_CONSEC_FRAMES:
                self.yawn_count += 1
                self.logger.log("yawn_detected", f"MAR={mar:.2f}")
        else:
            self.frame_counter_mar = 0

    def _process_distraction(self, yaw):
        now = time.time()
        if abs(yaw) > DISTRACTION_YAW_THRESHOLD:
            if self.distracted_since is None:
                self.distracted_since = now
            duration = now - self.distracted_since
            if duration >= DISTRACTION_SECONDS:
                self._trigger_alert(
                    "DISTRACTION — driver looking away from the road!",
                    "distraction_head_turn", f"{duration:.1f}s, yaw={yaw:.2f}",
                )
        else:
            self.distracted_since = None

    def _handle_no_face(self):
        now = time.time()
        if self.no_face_since is None:
            self.no_face_since = now
        duration = now - self.no_face_since
        self.mood_value.config(text="No face detected", fg="#8b93a1")
        self.eye_value.config(text="—", fg="#c7ccd4")
        if duration >= DISTRACTION_SECONDS:
            self._trigger_alert(
                "NO FACE DETECTED — check camera position / driver visibility.",
                "no_face", f"{duration:.1f}s",
            )

    def _trigger_alert(self, message, event_type, detail):
        now = time.time()
        self.alert_value.config(text=f"⚠ {message}", fg="#ff5252")
        if event_type == "drowsy_eyes_closed":
            self.drowsy_count += 1
        elif event_type == "distraction_head_turn":
            self.distraction_count += 1
        if now - self.last_alarm_time > ALARM_COOLDOWN_SECONDS:
            self.last_alarm_time = now
            self.alarm.play()
            self.logger.log(event_type, detail)

    def _refresh_status_panel(self, ear_avg, mood):
        eye_open = ear_avg >= EAR_THRESHOLD
        self.eye_value.config(
            text="Open" if eye_open else "CLOSED",
            fg="#66bb6a" if eye_open else "#ff5252",
        )
        self.ear_value.config(text=f"EAR: {ear_avg:.3f}")
        self.blink_value.config(text=f"Blinks: {self.total_blinks}")
        self.mood_value.config(text=mood, fg=self._mood_color(mood))
        self.yawn_value.config(text=f"Yawns: {self.yawn_count}")
        self.drowsy_value.config(text=f"Drowsy alerts: {self.drowsy_count}")
        self.distract_value.config(text=f"Distraction alerts: {self.distraction_count}")

        if self.eyes_closed_since is None and self.distracted_since is None:
            if time.time() - self.last_alarm_time > ALARM_COOLDOWN_SECONDS:
                self.alert_value.config(text="All clear", fg="#66bb6a")

    @staticmethod
    def _mood_color(mood):
        return {
            "Happy": "#66bb6a",
            "Sad": "#5c9ce6",
            "Angry / Frustrated": "#ff7043",
            "Surprised": "#ffca28",
            "Tired / Drowsy": "#ff5252",
            "Yawning": "#ff8a65",
            "Neutral": "#4fc3f7",
        }.get(mood, "#f5f7fa")

    # ------------------------------------------------------------------
    # Drawing overlays on the frame
    # ------------------------------------------------------------------
    def _draw_overlays(self, frame, landmarks, ear_avg, mar):
        eye_color = (80, 220, 100) if ear_avg >= EAR_THRESHOLD else (60, 60, 255)
        for idx in LEFT_EYE + RIGHT_EYE:
            cv2.circle(frame, (int(landmarks[idx][0]), int(landmarks[idx][1])), 2, eye_color, -1)
        mouth_color = (60, 60, 255) if mar > YAWN_MAR_THRESHOLD else (220, 180, 60)
        for idx in [MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT]:
            cv2.circle(frame, (int(landmarks[idx][0]), int(landmarks[idx][1])), 2, mouth_color, -1)


def main():
    root = tk.Tk()
    app = DriverMonitorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
