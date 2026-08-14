## Author - Denuwan weerakkody 
# Driver Alertness & Expression Monitor (Prototype)

A desktop Python app that uses your webcam to track eye state, facial
expression/mood, and head position — aimed at flagging driver drowsiness,
yawning, and distraction ("looking away") in real time, with an on-screen
alert, an audible alarm, and a CSV event log.

> ⚠️ **Safety disclaimer:** This is a learning/hobby prototype, not a
> certified or medically validated safety system. Real automotive Driver
> Monitoring Systems (DMS) go through rigorous testing and regulatory
> approval (e.g. UN R159 / EU General Safety Regulation 2, which now
> requires drowsiness/attention warning systems in new UK/EU vehicle
> types). Do not rely on this software as your only defence against
> fatigue — if you feel drowsy while driving, stop and rest.

## What it does

- **Eye tracking (EAR – Eye Aspect Ratio):** tracks eye landmarks via
  MediaPipe Face Mesh; if eyes stay closed beyond a threshold, it raises
  a drowsiness alert + alarm sound.
- **Yawn detection (MAR – Mouth Aspect Ratio):** counts yawns, an early
  fatigue signal.
- **Mood / expression detection:** a lightweight, geometry-based
  classifier (no extra model download) estimates Neutral / Happy / Sad /
  Angry / Surprised / Tired / Yawning from landmark positions.
- **Distraction detection:** estimates head yaw (left/right turn) from
  landmark geometry; sustained head-turn away from centre triggers a
  distraction alert.
- **No-face detection:** if the driver's face isn't visible for too long
  (camera blocked, driver slumped out of frame, etc.), it alerts too.
- **GUI (Tkinter):** live camera preview with eye/mouth overlay dots,
  a status panel (mood, eye state, EAR value, blink count), alert
  banner, and running session stats.
- **Event logging:** every alert is timestamped and appended to
  `driver_monitor_log.csv` for later review.

## Setup

1. Install Python 3.9+ (3.10–3.12 recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run it:
   ```bash
   python main.py
   ```
4. Click **Start Monitoring**. Make sure your face is well lit and the
   webcam has a clear view of your eyes and mouth.

If you have more than one camera, change `self.camera_index = 0` near
the top of `DriverMonitorApp.__init__` in `main.py` (try `1`, `2`, etc.).

## Tuning it

All the detection thresholds live near the top of `main.py`:

| Constant | Meaning |
|---|---|
| `EAR_THRESHOLD` | Eye-aspect-ratio value below which eyes count as closed |
| `DROWSY_EYES_CLOSED_SECONDS` | How long eyes must stay closed before a drowsiness alert fires |
| `YAWN_MAR_THRESHOLD` | Mouth-aspect-ratio value that counts as a yawn |
| `DISTRACTION_YAW_THRESHOLD` | How far the head can turn before it's flagged as "looking away" |
| `DISTRACTION_SECONDS` | How long that head turn must persist before alerting |
| `ALARM_COOLDOWN_SECONDS` | Minimum gap between repeated alarm sounds |

Everyone's eyes/face geometry and camera angle differ slightly, so it's
worth running a session and adjusting `EAR_THRESHOLD` up/down (e.g.
0.18–0.25) if it's too sensitive or not sensitive enough.

## Notes on accuracy & next steps

- The mood classifier is a **hand-built heuristic** based on facial
  geometry, not a trained deep-learning model — it's a good, fast,
  offline starting point but won't be as accurate as a trained
  classifier. For higher accuracy you could swap in a proper model
  (e.g. the `fer` or `deepface` Python packages, or fine-tune a small
  CNN on a labelled dataset like FER2013/AffectNet), while still reusing
  this app's EAR/MAR/head-yaw logic and GUI.
- Head-yaw "looking away" detection is a simple geometric estimate — a
  real product would typically use full 3D head pose estimation
  (solvePnP with a 3D face model) for more reliable results, especially
  at extreme angles or in poor lighting.
- For an in-vehicle product aimed at real drivers, you'd also want: IR
  camera support for night driving, mounting/vibration robustness,
  offline operation with no cloud dependency (this app is already fully
  offline), and — critically — formal safety validation before it's
  trusted on the road.

## Files

- `main.py` — the full application (GUI + detection logic)
- `requirements.txt` — Python dependencies
- `driver_monitor_log.csv` — created automatically on first run; logs
  every drowsiness/yawn/distraction/no-face event with a timestamp
