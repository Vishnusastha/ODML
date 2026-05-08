import sys, os, time
import cv2
import numpy as np
import torch
from collections import defaultdict, deque
from ultralytics import YOLO

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QComboBox, QSlider, QCheckBox, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFrame, QSizePolicy, QScrollArea, QSplashScreen,
    QGraphicsOpacityEffect
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation,
    QEasingCurve, QSize, QPoint
)
from PyQt6.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter, QPen,
    QBrush, QLinearGradient, QRadialGradient, QPainterPath,
    QPolygon
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ███  SECTION 1: CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOLO_MODEL_FILES = {
    "YOLOv5":  "yolov5su.pt",
    "YOLOv8":  "yolov8m.pt",
    "YOLOv9":  "yolov9c.pt",
    "YOLOv10": "yolov10m.pt",
    "YOLOv11": "yolo11m.pt",
}

DEFAULT_MODEL      = "YOLOv8"
DEFAULT_CONF       = 0.25
DEFAULT_SCALE      = 0.60
TRAIL_LENGTH       = 25
LOW_LIGHT_THRESH   = 75
HUMAN_SPEED_THRESH = 50
VEHICLE_CLASSES    = {"car", "truck", "bus", "motorcycle"}

VIDEO_FPS          = 20
DATASET_DIR        = "auto_dataset"
RECORDINGS_DIR     = "recordings"
SNAPSHOTS_DIR      = "snapshots"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ███  SECTION 2: THEME / PALETTE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

C = {
    "bg":      "#080B10",
    "panel":   "#0D1117",
    "card":    "#111827",
    "border":  "#1C2333",
    "accent":  "#00D4FF",
    "green":   "#00FF9C",
    "warn":    "#FFB547",
    "red":     "#FF3B5C",
    "purple":  "#B06EFF",
    "text":    "#DDE4F0",
    "muted":   "#4A5568",
    "hi":      "#162032",
}

APP_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {C['bg']};
    color: {C['text']};
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
}}
QFrame#panel {{
    background-color: {C['panel']};
    border: 1px solid {C['border']};
    border-radius: 14px;
}}
QFrame#card {{
    background-color: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 10px;
}}
QFrame#hline {{
    background: {C['border']};
    max-height: 1px;
    min-height: 1px;
    border: none;
}}
QPushButton {{
    background-color: {C['card']};
    color: {C['text']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
}}
QPushButton:hover {{
    background: {C['hi']};
    border: 1px solid {C['accent']};
    color: {C['accent']};
}}
QPushButton:disabled {{
    color: {C['muted']};
    border-color: {C['border']};
}}
QPushButton#btnStart {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #0099CC, stop:1 #00D4FF);
    color: #080B10;
    border: none;
    padding: 11px 24px;
    font-size: 13px;
    font-weight: 700;
    border-radius: 10px;
}}
QPushButton#btnStart:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #00BBEE, stop:1 #33DDFF);
}}
QPushButton#btnStop {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #CC1133, stop:1 #FF3B5C);
    color: white;
    border: none;
    padding: 11px 24px;
    font-size: 13px;
    font-weight: 700;
    border-radius: 10px;
}}
QPushButton#btnStop:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #EE1144, stop:1 #FF5577);
}}
QPushButton#btnRec {{
    background: {C['card']};
    color: {C['red']};
    border: 1px solid {C['red']};
    border-radius: 10px;
    padding: 11px 20px;
    font-size: 13px;
    font-weight: 700;
}}
QPushButton#btnRec:hover {{
    background: rgba(255,59,92,0.15);
}}
QPushButton#btnRecActive {{
    background: {C['red']};
    color: white;
    border: none;
    border-radius: 10px;
    padding: 11px 20px;
    font-size: 13px;
    font-weight: 700;
}}
QPushButton#btnSnap {{
    background: {C['card']};
    color: {C['green']};
    border: 1px solid {C['green']};
    border-radius: 10px;
    padding: 11px 20px;
    font-size: 13px;
    font-weight: 700;
}}
QPushButton#btnSnap:hover {{
    background: rgba(0,255,156,0.1);
}}
QComboBox {{
    background: {C['card']};
    color: {C['text']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 12px;
    min-width: 120px;
}}
QComboBox:hover {{ border-color: {C['accent']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {C['card']};
    color: {C['text']};
    border: 1px solid {C['border']};
    selection-background-color: {C['hi']};
    selection-color: {C['accent']};
}}
QSlider::groove:horizontal {{
    height: 3px;
    background: {C['border']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px;
    background: {C['accent']};
    border-radius: 7px;
    margin: -5px 0;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #005588, stop:1 {C['accent']});
    border-radius: 2px;
}}
QCheckBox {{
    color: {C['text']};
    font-size: 12px;
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border-radius: 4px;
    border: 1px solid {C['border']};
    background: {C['card']};
}}
QCheckBox::indicator:checked {{
    background: {C['accent']};
    border-color: {C['accent']};
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background: transparent;
    border: none;
}}
QScrollBar:vertical {{
    background: {C['panel']};
    width: 5px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {C['border']};
    border-radius: 3px;
    min-height: 20px;
}}
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ███  SECTION 3: BACKEND — YOLO DETECTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class YOLODetector:
    """
    Wraps all YOLO detection logic.
    Keeps state for tracking, settings, and video writing.
    """

    def __init__(self):
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"
        self.models       = {}
        self.active_name  = DEFAULT_MODEL
        self.conf         = DEFAULT_CONF
        self.scale        = DEFAULT_SCALE
        self.class_colors = {}

        self.track_history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

        # Feature flags
        self.show_trails    = True
        self.show_predict   = True
        self.enhance_light  = True
        self.save_dataset   = True   # ON by default — auto dataset collection
        self.save_video     = False

        self._video_writer        = None
        self._rec_w               = 0    # stored at start_recording() time
        self._rec_h               = 0
        self._dataset_frame_count = 0   # used to throttle dataset saves
        self._dataset_save_every  = 5   # save one frame every N processed frames

    # ── Model loading ──────────────────────────────────────────

    def load_models(self, progress_cb=None):
        """Load all YOLO models; call progress_cb(name, success)."""
        for name, path in YOLO_MODEL_FILES.items():
            try:
                self.models[name] = YOLO(path)
                if progress_cb:
                    progress_cb(name, True, None)
            except Exception as e:
                if progress_cb:
                    progress_cb(name, False, str(e))

        if not self.models:
            raise RuntimeError("No YOLO models could be loaded.")

        # Build colour map from the first loaded model's class list
        ref = self._ref()
        np.random.seed(42)
        self.class_colors = {
            cls: tuple(int(x) for x in np.random.randint(80, 255, 3))
            for cls in ref.names.values()
        }

    def _ref(self):
        return list(self.models.values())[0]

    def set_model(self, name):
        if name in self.models:
            self.active_name = name

    def set_conf(self, v):   self.conf  = max(0.01, min(0.99, v))
    def set_scale(self, v):  self.scale = max(0.20, min(1.00, v))

    # ── Video writer ───────────────────────────────────────────

    def start_recording(self, w=640, h=480):
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        fname  = f"{RECORDINGS_DIR}/rec_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(fname, fourcc, VIDEO_FPS, (w, h))
        if not self._video_writer.isOpened():
            fname = fname.replace(".mp4", ".avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self._video_writer = cv2.VideoWriter(fname, fourcc, VIDEO_FPS, (w, h))
        # Store dimensions ourselves — VideoWriter.get() returns 0 on most backends
        self._rec_w = w
        self._rec_h = h
        return fname

    def stop_recording(self):
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None

    # ── Utility helpers ────────────────────────────────────────

    @staticmethod
    def _motion_speed(track):
        if len(track) < 2:
            return 0.0
        (x1, y1), (x2, y2) = track[-2], track[-1]
        return ((x2-x1)**2 + (y2-y1)**2) ** 0.5

    @staticmethod
    def _predict_next(track):
        if len(track) < 2:
            return None
        (x1, y1), (x2, y2) = track[-2], track[-1]
        return (x2 + (x2-x1), y2 + (y2-y1))

    @staticmethod
    def _overlap(a, b):
        return not (a[2]<b[0] or a[0]>b[2] or a[3]<b[1] or a[1]>b[3])

    @staticmethod
    def _is_dark(frame):
        small = cv2.resize(frame, (80, 60))
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean() < LOW_LIGHT_THRESH

    @staticmethod
    def _enhance(frame):
        return cv2.convertScaleAbs(frame, alpha=1.5, beta=35)

    @staticmethod
    def _crowd_label(n_persons):
        if n_persons < 3:  return "LOW",    (0, 220, 100)
        if n_persons < 7:  return "MEDIUM", (0, 180, 255)
        return "HIGH", (0, 50, 255)

    # ── Core frame processor ───────────────────────────────────

    def process(self, frame):
        """
        Run detection on one frame.
        Returns (annotated_frame, stats_dict).
        """
        if frame is None:
            return None, {}

        h, w = frame.shape[:2]
        out  = frame.copy()

        low_light = False
        if self.enhance_light and self._is_dark(frame):
            out       = self._enhance(out)
            low_light = True

        # Downscale for speed
        sw, sh = int(w * self.scale), int(h * self.scale)
        small  = cv2.resize(out, (sw, sh))

        model = self.models.get(self.active_name, self._ref())
        ref   = self._ref()

        results = model.predict(small, conf=self.conf,
                                device=self.device, verbose=False)

        counts        = defaultdict(int)
        vehicle_boxes = []
        total         = 0
        events        = set()

        for r in results:
            if r.boxes is None:
                continue

            boxes   = r.boxes.xyxy.cpu().numpy() / self.scale
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs   = r.boxes.conf.cpu().numpy()

            for i, raw_box in enumerate(boxes):
                x1, y1, x2, y2 = (
                    max(0, int(raw_box[0])), max(0, int(raw_box[1])),
                    min(w, int(raw_box[2])), min(h, int(raw_box[3]))
                )
                if x2 <= x1 or y2 <= y1:
                    continue

                cls   = ref.names[cls_ids[i]]
                color = self.class_colors.get(cls, (0, 200, 0))
                conf  = confs[i]
                total += 1
                counts[cls] += 1

                # Bounding box + filled label background
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label = f"{cls} {conf:.2f}"
                (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                cv2.rectangle(out, (x1, y1 - lh - bl - 4), (x1 + lw + 4, y1), color, -1)
                cv2.putText(out, label, (x1 + 2, y1 - bl - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

                # Tracking
                cx, cy = (x1+x2)//2, (y1+y2)//2
                oid    = cx * 10000 + cy          # simple spatial ID
                self.track_history[oid].append((cx, cy))
                track  = self.track_history[oid]

                # Motion trail
                if self.show_trails and len(track) > 1:
                    pts = list(track)
                    for j in range(1, len(pts)):
                        alpha = j / len(pts)
                        tc    = tuple(int(c * alpha) for c in color)
                        cv2.line(out, pts[j-1], pts[j], tc, 2, cv2.LINE_AA)

                # Prediction dot
                if self.show_predict:
                    pred = self._predict_next(track)
                    if pred:
                        cv2.circle(out, (int(pred[0]), int(pred[1])),
                                   5, (255, 80, 255), -1, cv2.LINE_AA)

                # Speed / suspicious motion
                speed = self._motion_speed(track)
                if cls == "person" and speed > HUMAN_SPEED_THRESH:
                    events.add("SUSPICIOUS MOTION")

                # Vehicle collision tracking
                if cls in VEHICLE_CLASSES:
                    vehicle_boxes.append((x1, y1, x2, y2))

                # Dataset auto-save (throttled, full annotated frame + ROI crop)
                if self.save_dataset:
                    self._dataset_frame_count += 1
                    if self._dataset_frame_count % self._dataset_save_every == 0:
                        # Per-class subfolder so dataset is organised
                        cls_dir = os.path.join(DATASET_DIR, "images", cls)
                        os.makedirs(cls_dir, exist_ok=True)
                        ts_ms = int(time.time() * 1000)
                        # Save the ROI crop from the enhanced/annotated frame
                        roi = out[y1:y2, x1:x2]
                        if roi.size > 0:
                            cv2.imwrite(f"{cls_dir}/{cls}_{ts_ms}.jpg", roi)
                        # Also save the full annotated frame (once per throttle tick)
                        full_dir = os.path.join(DATASET_DIR, "full_frames")
                        os.makedirs(full_dir, exist_ok=True)
                        cv2.imwrite(f"{full_dir}/frame_{ts_ms}.jpg", out)

        # Collision detection between vehicles
        for i in range(len(vehicle_boxes)):
            for j in range(i+1, len(vehicle_boxes)):
                if self._overlap(vehicle_boxes[i], vehicle_boxes[j]):
                    events.add("POSSIBLE ACCIDENT")

        # Write to video — every frame so recording is smooth and continuous
        if self.save_video and self._video_writer and self._video_writer.isOpened():
            fh, fw = out.shape[:2]
            if fw == self._rec_w and fh == self._rec_h:
                self._video_writer.write(out)
            elif self._rec_w > 0 and self._rec_h > 0:
                self._video_writer.write(cv2.resize(out, (self._rec_w, self._rec_h)))

        density_label, _ = self._crowd_label(counts.get("person", 0))

        stats = {
            "total":    total,
            "counts":   dict(counts),
            "density":  density_label,
            "events":   list(events),
            "low_light": low_light,
            "model":    self.active_name,
            "device":   self.device,
        }
        return out, stats

    def release(self):
        self.stop_recording()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ███  SECTION 4: WORKER THREADS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LoaderThread(QThread):
    """Loads YOLO models in background so the UI stays responsive."""
    sig_progress = pyqtSignal(str, bool, str)   # name, ok, error_msg
    sig_done     = pyqtSignal()
    sig_error    = pyqtSignal(str)

    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def run(self):
        try:
            self.detector.load_models(
                progress_cb=lambda n, ok, e: self.sig_progress.emit(n, ok, e or "")
            )
            self.sig_done.emit()
        except Exception as ex:
            self.sig_error.emit(str(ex))


class CameraThread(QThread):
    """Captures frames, runs detection, emits results to GUI."""
    sig_frame = pyqtSignal(np.ndarray, dict)
    sig_fps   = pyqtSignal(float)
    sig_error = pyqtSignal(str)

    def __init__(self, detector, cam_id=0):
        super().__init__()
        self.detector = detector
        self.cam_id   = cam_id
        self._running = False
        self._paused  = False

    def run(self):
        cap = cv2.VideoCapture(self.cam_id)
        if not cap.isOpened():
            self.sig_error.emit("Cannot open camera (index 0).")
            return

        self._running = True
        prev = time.perf_counter()
        smooth_fps = 0.0

        while self._running:
            if self._paused:
                time.sleep(0.04)
                continue

            ret, frame = cap.read()
            if not ret:
                self.sig_error.emit("Frame read failed.")
                break

            annotated, stats = self.detector.process(frame)

            now       = time.perf_counter()
            inst_fps  = 1.0 / max(now - prev, 1e-6)
            prev      = now
            smooth_fps = 0.85 * smooth_fps + 0.15 * inst_fps
            self.sig_fps.emit(smooth_fps)

            if annotated is not None:
                self.sig_frame.emit(annotated, stats)

        cap.release()

    def stop(self):
        self._running = False
        self.wait(3000)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ███  SECTION 5: CUSTOM WIDGETS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LED(QLabel):
    """Tiny circular LED indicator widget."""
    def __init__(self, color=C["green"], size=10, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._color = color
        self._on    = False
        self._draw()

    def set_state(self, on: bool, color=None):
        self._on = on
        if color:
            self._color = color
        self._draw()

    def _draw(self):
        px = QPixmap(self.width(), self.height())
        px.fill(Qt.GlobalColor.transparent)
        p  = QPainter(px)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c  = QColor(self._color if self._on else C["muted"])
        if self._on:
            # Glow effect
            g = QRadialGradient(self.width()/2, self.height()/2,
                                self.width()/2)
            g.setColorAt(0.0, c.lighter(160))
            g.setColorAt(1.0, c)
            p.setBrush(QBrush(g))
        else:
            p.setBrush(QBrush(c))
        p.setPen(Qt.PenStyle.NoPen)
        m = 1
        p.drawEllipse(m, m, self.width()-2*m, self.height()-2*m)
        p.end()
        self.setPixmap(px)


class StatCard(QFrame):
    """Compact metric card: title + big number."""
    def __init__(self, title, init_val="—", color=C["accent"], parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._color = color

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)

        self._title = QLabel(title.upper())
        self._title.setFont(QFont("Courier New", 8, QFont.Weight.Bold))
        self._title.setStyleSheet(f"color: {C['muted']}; letter-spacing: 1.5px;")

        self._val = QLabel(str(init_val))
        self._val.setFont(QFont("Courier New", 24, QFont.Weight.Bold))
        self._val.setStyleSheet(f"color: {color};")

        lay.addWidget(self._title)
        lay.addWidget(self._val)

    def update_value(self, v):
        self._val.setText(str(v))


class TagBadge(QLabel):
    """Small coloured pill badge for events."""
    def __init__(self, text, bg=C["red"], parent=None):
        super().__init__(text, parent)
        self.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.setStyleSheet(
            f"background:{bg}; color:white; border-radius:6px;"
            f"padding:3px 10px; letter-spacing:0.5px;"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


def hline():
    f = QFrame()
    f.setObjectName("hline")
    return f


def section_label(text):
    lbl = QLabel(text)
    lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
    lbl.setStyleSheet(f"color:{C['muted']}; letter-spacing:2px; margin-top:6px;")
    return lbl


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ███  SECTION 6: SPLASH SCREEN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(480, 280)

        # Centre on screen
        screen = QApplication.primaryScreen().geometry()
        self.move(
            (screen.width()  - self.width())  // 2,
            (screen.height() - self.height()) // 2,
        )

        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 40, 40, 40)
        lay.setSpacing(12)

        title = QLabel("YOLO VISION")
        title.setFont(QFont("Courier New", 28, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {C['accent']}; letter-spacing: 6px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sub = QLabel("AI Object Detection System")
        sub.setFont(QFont("Segoe UI", 11))
        sub.setStyleSheet(f"color: {C['muted']};")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_lbl = QLabel("Initialising…")
        self.status_lbl.setFont(QFont("Courier New", 10))
        self.status_lbl.setStyleSheet(f"color: {C['text']};")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.log_lbl = QLabel("")
        self.log_lbl.setFont(QFont("Courier New", 9))
        self.log_lbl.setStyleSheet(f"color: {C['muted']};")
        self.log_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lay.addStretch()
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addSpacing(16)
        lay.addWidget(self.status_lbl)
        lay.addWidget(self.log_lbl)
        lay.addStretch()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(QColor(C["panel"])))
        p.setPen(QPen(QColor(C["border"]), 1))
        p.drawRoundedRect(self.rect().adjusted(1,1,-1,-1), 16, 16)
        p.end()

    def set_status(self, txt):
        self.status_lbl.setText(txt)

    def set_log(self, txt):
        self.log_lbl.setText(txt)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ███  SECTION 7: MAIN WINDOW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Vision — AI Object Detection")
        self.setMinimumSize(1200, 740)
        self.resize(1380, 840)
        self.setStyleSheet(APP_STYLE)

        self.detector     = YOLODetector()
        self.cam_thread   = None
        self._recording   = False
        self._frame_count = 0
        self._last_frame  = None   # for snapshot
        self._log_rows    = []
        self._ds_crops    = 0      # dataset crop counter
        self._ds_frames   = 0      # dataset full-frame counter

        self._build_ui()

    # ─────────────────────────────────────────────
    # BUILD UI
    # ─────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        # LEFT column (camera + controls)
        left_col = QVBoxLayout()
        left_col.setSpacing(10)
        left_col.addLayout(self._build_header())
        left_col.addWidget(self._build_camera_panel(), stretch=1)
        left_col.addWidget(self._build_event_bar())
        left_col.addLayout(self._build_controls())

        # RIGHT column (stats + settings)
        right_col = self._build_right_panel()

        outer.addLayout(left_col, stretch=3)
        outer.addWidget(right_col, stretch=1)

    # ── Header ────────────────────────────────────

    def _build_header(self):
        lay = QHBoxLayout()
        lay.setSpacing(14)

        logo = QLabel("◈  YOLO VISION")
        logo.setFont(QFont("Courier New", 17, QFont.Weight.Bold))
        logo.setStyleSheet(f"color:{C['accent']}; letter-spacing:4px;")

        sub = QLabel("Multi-Model Real-Time Object Detection")
        sub.setFont(QFont("Segoe UI", 10))
        sub.setStyleSheet(f"color:{C['muted']};")

        lc = QVBoxLayout()
        lc.setSpacing(1)
        lc.addWidget(logo)
        lc.addWidget(sub)

        lay.addLayout(lc)
        lay.addStretch()

        # Live badge
        self.led_live = LED(C["green"], 12)
        self.lbl_fps  = QLabel("— FPS")
        self.lbl_fps.setFont(QFont("Courier New", 13, QFont.Weight.Bold))
        self.lbl_fps.setStyleSheet(f"color:{C['green']};")

        vsep = QFrame()
        vsep.setFrameShape(QFrame.Shape.VLine)
        vsep.setStyleSheet(f"color:{C['border']};")

        self.lbl_device = QLabel("DEVICE: CPU")
        self.lbl_device.setFont(QFont("Courier New", 10))
        self.lbl_device.setStyleSheet(f"color:{C['muted']};")

        badge = QHBoxLayout()
        badge.setSpacing(8)
        badge.addWidget(self.led_live)
        badge.addWidget(self.lbl_fps)
        badge.addWidget(vsep)
        badge.addWidget(self.lbl_device)

        lay.addLayout(badge)
        return lay

    # ── Camera panel ──────────────────────────────

    def _build_camera_panel(self):
        self.cam_lbl = QLabel()
        self.cam_lbl.setObjectName("panel")
        self.cam_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.cam_lbl.setMinimumSize(820, 480)
        self.cam_lbl.setStyleSheet(
            f"color:{C['muted']}; background:{C['panel']}; border-radius:14px;")
        self.cam_lbl.setText("● MODELS LOADING…")
        self.cam_lbl.setFont(QFont("Courier New", 14))
        return self.cam_lbl

    # ── Event bar ─────────────────────────────────

    def _build_event_bar(self):
        frame = QFrame()
        frame.setObjectName("card")
        frame.setFixedHeight(46)

        lay = QHBoxLayout(frame)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(10)

        self.lbl_event_dot = QLabel("○")
        self.lbl_event_dot.setFont(QFont("Courier New", 14, QFont.Weight.Bold))
        self.lbl_event_dot.setStyleSheet(f"color:{C['muted']};")

        self.lbl_event_txt = QLabel("System ready — no events detected")
        self.lbl_event_txt.setFont(QFont("Segoe UI", 11))
        self.lbl_event_txt.setStyleSheet(f"color:{C['muted']};")

        lay.addWidget(self.lbl_event_dot)
        lay.addWidget(self.lbl_event_txt)
        lay.addStretch()

        self.lbl_density = QLabel("DENSITY: —")
        self.lbl_density.setFont(QFont("Courier New", 11, QFont.Weight.Bold))
        self.lbl_density.setStyleSheet(f"color:{C['muted']};")
        lay.addWidget(self.lbl_density)

        self.lbl_ll = QLabel("")  # Low-light indicator
        self.lbl_ll.setFont(QFont("Courier New", 10))
        self.lbl_ll.setStyleSheet(f"color:{C['warn']};")
        lay.addWidget(self.lbl_ll)

        return frame

    # ── Controls ──────────────────────────────────

    def _build_controls(self):
        lay = QHBoxLayout()
        lay.setSpacing(8)

        self.btn_start = QPushButton("▶  START")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setFixedHeight(46)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._start_camera)

        self.btn_stop = QPushButton("■  STOP")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setFixedHeight(46)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_camera)

        self.btn_snap = QPushButton("⌾  SNAPSHOT")
        self.btn_snap.setObjectName("btnSnap")
        self.btn_snap.setFixedHeight(46)
        self.btn_snap.setEnabled(False)
        self.btn_snap.clicked.connect(self._snapshot)

        self.btn_rec = QPushButton("⏺  RECORD")
        self.btn_rec.setObjectName("btnRec")
        self.btn_rec.setFixedHeight(46)
        self.btn_rec.setEnabled(False)
        self.btn_rec.clicked.connect(self._toggle_record)

        for b in [self.btn_start, self.btn_stop,
                  self.btn_snap, self.btn_rec]:
            lay.addWidget(b)

        lay.addStretch()

        # Model selector
        col = QVBoxLayout()
        col.setSpacing(3)
        col.addWidget(section_label("MODEL"))
        self.combo_model = QComboBox()
        self.combo_model.setFixedHeight(40)
        self.combo_model.currentTextChanged.connect(self.detector.set_model)
        col.addWidget(self.combo_model)
        lay.addLayout(col)

        # Camera ID
        col2 = QVBoxLayout()
        col2.setSpacing(3)
        col2.addWidget(section_label("CAMERA"))
        self.combo_cam = QComboBox()
        self.combo_cam.addItems(["0 — Default", "1", "2", "3"])
        self.combo_cam.setFixedHeight(40)
        col2.addWidget(self.combo_cam)
        lay.addLayout(col2)

        return lay

    # ── Right panel (stats + settings) ────────────

    def _build_right_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(310)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 0, 8, 0)
        lay.setSpacing(10)

        # ── Stats cards
        lay.addWidget(section_label("LIVE STATISTICS"))

        grid = QGridLayout()
        grid.setSpacing(8)
        self.card_total   = StatCard("Objects",  "0",   C["accent"])
        self.card_persons = StatCard("Persons",  "0",   C["green"])
        self.card_vehicles= StatCard("Vehicles", "0",   C["warn"])
        self.card_fps     = StatCard("FPS",      "0",   C["purple"])
        grid.addWidget(self.card_total,    0, 0)
        grid.addWidget(self.card_persons,  0, 1)
        grid.addWidget(self.card_vehicles, 1, 0)
        grid.addWidget(self.card_fps,      1, 1)
        lay.addLayout(grid)

        # ── Class breakdown
        lay.addWidget(hline())
        lay.addWidget(section_label("CLASS BREAKDOWN"))

        self.breakdown_frame = QFrame()
        self.breakdown_frame.setObjectName("card")
        self.breakdown_lay   = QVBoxLayout(self.breakdown_frame)
        self.breakdown_lay.setContentsMargins(12, 8, 12, 8)
        self.breakdown_lay.setSpacing(4)
        self._breakdown_lbl  = QLabel("No detections yet")
        self._breakdown_lbl.setFont(QFont("Courier New", 9))
        self._breakdown_lbl.setStyleSheet(f"color:{C['muted']};")
        self._breakdown_lbl.setWordWrap(True)
        self.breakdown_lay.addWidget(self._breakdown_lbl)
        lay.addWidget(self.breakdown_frame)

        # ── Settings
        lay.addWidget(hline())
        lay.addWidget(section_label("SETTINGS"))

        # Confidence
        rc = QHBoxLayout()
        rc.addWidget(section_label("CONFIDENCE"))
        rc.addStretch()
        self.lbl_conf = QLabel("0.25")
        self.lbl_conf.setFont(QFont("Courier New", 11, QFont.Weight.Bold))
        self.lbl_conf.setStyleSheet(f"color:{C['accent']};")
        rc.addWidget(self.lbl_conf)
        lay.addLayout(rc)

        self.sld_conf = QSlider(Qt.Orientation.Horizontal)
        self.sld_conf.setRange(5, 95)
        self.sld_conf.setValue(25)
        self.sld_conf.valueChanged.connect(self._conf_changed)
        lay.addWidget(self.sld_conf)

        # Scale
        rs = QHBoxLayout()
        rs.addWidget(section_label("DETECT SCALE"))
        rs.addStretch()
        self.lbl_scale = QLabel("0.60")
        self.lbl_scale.setFont(QFont("Courier New", 11, QFont.Weight.Bold))
        self.lbl_scale.setStyleSheet(f"color:{C['accent']};")
        rs.addWidget(self.lbl_scale)
        lay.addLayout(rs)

        self.sld_scale = QSlider(Qt.Orientation.Horizontal)
        self.sld_scale.setRange(20, 100)
        self.sld_scale.setValue(60)
        self.sld_scale.valueChanged.connect(self._scale_changed)
        lay.addWidget(self.sld_scale)

        # Feature toggles
        lay.addWidget(hline())
        lay.addWidget(section_label("FEATURES"))

        toggles = [
            ("Motion Trails",           "show_trails",    True),
            ("Predict Next Position",   "show_predict",   True),
            ("Low-Light Enhancement",   "enhance_light",  True),
        ]
        for label, attr, default in toggles:
            chk = QCheckBox(label)
            chk.setChecked(default)
            chk.stateChanged.connect(
                lambda s, a=attr: setattr(self.detector, a, bool(s))
            )
            lay.addWidget(chk)

        # Dataset toggle — ON by default, with live save counter
        self.chk_dataset = QCheckBox("Auto Save Dataset")
        self.chk_dataset.setChecked(True)
        self.chk_dataset.setStyleSheet(f"color:{C['green']}; font-weight:600;")
        self.chk_dataset.stateChanged.connect(
            lambda s: setattr(self.detector, 'save_dataset', bool(s))
        )
        lay.addWidget(self.chk_dataset)

        self.lbl_dataset_count = QLabel("  Saved: 0 crops  |  0 frames")
        self.lbl_dataset_count.setFont(QFont("Courier New", 8))
        self.lbl_dataset_count.setStyleSheet(f"color:{C['muted']}; margin-left:24px;")
        lay.addWidget(self.lbl_dataset_count)

        # ── Log
        lay.addWidget(hline())
        lay.addWidget(section_label("DETECTION LOG"))

        self.log_frame = QFrame()
        self.log_frame.setObjectName("card")
        self.log_inner = QVBoxLayout(self.log_frame)
        self.log_inner.setContentsMargins(10, 8, 10, 8)
        self.log_inner.setSpacing(3)
        lay.addWidget(self.log_frame)

        lay.addStretch()
        scroll.setWidget(w)
        return scroll

    # ─────────────────────────────────────────────
    # MODEL LOADING
    # ─────────────────────────────────────────────

    def start_loading(self, splash=None):
        self._splash = splash
        self._loader = LoaderThread(self.detector)
        self._loader.sig_progress.connect(self._on_model_progress)
        self._loader.sig_done.connect(self._on_models_ready)
        self._loader.sig_error.connect(self._on_model_error)
        self._loader.start()

    def _on_model_progress(self, name, ok, err):
        icon = "✔" if ok else "✘"
        col  = C["green"] if ok else C["red"]
        msg  = f"{icon} {name}" + (f" — {err}" if err else "")
        if self._splash:
            self._splash.set_log(msg)
        if ok:
            self.combo_model.addItem(name)

    def _on_models_ready(self):
        if self._splash:
            self._splash.set_status("Ready!")
            QTimer.singleShot(600, self._splash.close)

        self.cam_lbl.setText("✔  Models loaded.  Press  ▶ START  to begin.")
        self.cam_lbl.setStyleSheet(
            f"color:{C['green']}; background:{C['panel']}; border-radius:14px;")
        self.btn_start.setEnabled(True)

        # Select default model
        idx = self.combo_model.findText(DEFAULT_MODEL)
        if idx >= 0:
            self.combo_model.setCurrentIndex(idx)

        self.lbl_device.setText(f"DEVICE: {self.detector.device.upper()}")

    def _on_model_error(self, msg):
        self.cam_lbl.setText(f"⚠  Error: {msg}")
        if self._splash:
            self._splash.close()

    # ─────────────────────────────────────────────
    # CAMERA CONTROL
    # ─────────────────────────────────────────────

    def _start_camera(self):
        cam_id = int(self.combo_cam.currentText().split("—")[0].strip())
        self.cam_thread = CameraThread(self.detector, cam_id)
        self.cam_thread.sig_frame.connect(self._on_frame)
        self.cam_thread.sig_fps.connect(self._on_fps)
        self.cam_thread.sig_error.connect(self._on_cam_error)
        self.cam_thread.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_snap.setEnabled(True)
        self.btn_rec.setEnabled(True)
        self.led_live.set_state(True, C["green"])

    def _stop_camera(self):
        if self.cam_thread:
            self.cam_thread.stop()
            self.cam_thread = None

        if self._recording:
            self.detector.stop_recording()
            self.detector.save_video = False
            self._recording = False
            self.btn_rec.setObjectName("btnRec")
            self.btn_rec.setText("⏺  RECORD")
            self.btn_rec.setStyle(self.btn_rec.style())

        self.cam_lbl.setPixmap(QPixmap())
        self.cam_lbl.setText("■  Camera stopped.")
        self.cam_lbl.setStyleSheet(
            f"color:{C['muted']}; background:{C['panel']}; border-radius:14px;")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_snap.setEnabled(False)
        self.btn_rec.setEnabled(False)
        self.led_live.set_state(False)
        self.lbl_fps.setText("— FPS")

    def _on_cam_error(self, msg):
        self._flash(f"⚠  {msg}", C["red"])
        self._stop_camera()

    # ─────────────────────────────────────────────
    # FRAME DISPLAY
    # ─────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray, stats: dict):
        self._frame_count += 1
        self._last_frame   = frame

        # Convert BGR → RGB → QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
        px  = QPixmap.fromImage(img)
        px  = px.scaled(self.cam_lbl.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
        self.cam_lbl.setPixmap(px)

        # Stat cards
        total    = stats.get("total", 0)
        counts   = stats.get("counts", {})
        density  = stats.get("density", "—")
        events   = stats.get("events", [])
        ll       = stats.get("low_light", False)

        persons  = counts.get("person", 0)
        vehicles = sum(counts.get(v, 0) for v in VEHICLE_CLASSES)

        self.card_total.update_value(total)
        self.card_persons.update_value(persons)
        self.card_vehicles.update_value(vehicles)

        # Density
        d_col = {
            "LOW": C["green"], "MEDIUM": C["warn"], "HIGH": C["red"]
        }.get(density, C["muted"])
        self.lbl_density.setText(f"DENSITY: {density}")
        self.lbl_density.setStyleSheet(
            f"color:{d_col}; font-family:'Courier New'; font-weight:bold;")

        # Low light
        self.lbl_ll.setText("◑ LOW LIGHT" if ll else "")

        # Events
        if events:
            self.lbl_event_dot.setText("●")
            self.lbl_event_dot.setStyleSheet(f"color:{C['red']};")
            self.lbl_event_txt.setText("  ·  ".join(events))
            self.lbl_event_txt.setStyleSheet(
                f"color:{C['red']}; font-weight:bold;")
        else:
            self.lbl_event_dot.setText("○")
            self.lbl_event_dot.setStyleSheet(f"color:{C['muted']};")
            self.lbl_event_txt.setText("No events detected")
            self.lbl_event_txt.setStyleSheet(f"color:{C['muted']};")

        # Class breakdown (update every 10 frames)
        if self._frame_count % 10 == 0:
            self._update_breakdown(counts)

        # Dataset counter (update every 30 frames)
        if self._frame_count % 30 == 0 and self.detector.save_dataset:
            self._update_dataset_counter()

        # Log entry (every 20 frames)
        if self._frame_count % 20 == 0 and counts:
            self._add_log(counts, events)

    def _on_fps(self, fps):
        self.lbl_fps.setText(f"{fps:.1f} FPS")
        self.card_fps.update_value(f"{fps:.0f}")

    def _update_breakdown(self, counts):
        if not counts:
            self._breakdown_lbl.setText("No detections")
            return
        lines = []
        for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(n, 10)
            lines.append(f"{cls:<12} {n:>3}  {bar}")
        self._breakdown_lbl.setText("\n".join(lines))
        self._breakdown_lbl.setFont(QFont("Courier New", 9))
        self._breakdown_lbl.setStyleSheet(f"color:{C['text']};")

    def _update_dataset_counter(self):
        """Count saved files on disk and update the label."""
        try:
            img_dir   = os.path.join(DATASET_DIR, "images")
            full_dir  = os.path.join(DATASET_DIR, "full_frames")
            crops = sum(
                len(files)
                for _, _, files in os.walk(img_dir)
            ) if os.path.exists(img_dir) else 0
            frames = len(os.listdir(full_dir)) if os.path.exists(full_dir) else 0
            self.lbl_dataset_count.setText(
                f"  Saved: {crops} crops  |  {frames} frames"
            )
            color = C["green"] if crops > 0 else C["muted"]
            self.lbl_dataset_count.setStyleSheet(
                f"color:{color}; margin-left:24px;"
            )
        except Exception:
            pass
    # ─────────────────────────────────────────────
    # LOG
    # ─────────────────────────────────────────────

    def _add_log(self, counts, events):
        ts    = time.strftime("%H:%M:%S")
        items = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()) if v)
        ev    = f"  ⚠ {', '.join(events)}" if events else ""
        text  = f"[{ts}] {items}{ev}"

        lbl = QLabel(text)
        lbl.setFont(QFont("Courier New", 8))
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color:{C['red'] if events else C['muted']};")
        self.log_inner.insertWidget(0, lbl)
        self._log_rows.append(lbl)

        if len(self._log_rows) > 14:
            old = self._log_rows.pop()
            old.deleteLater()

    # ─────────────────────────────────────────────
    # SLIDER HANDLERS
    # ─────────────────────────────────────────────

    def _conf_changed(self, v):
        val = v / 100
        self.lbl_conf.setText(f"{val:.2f}")
        self.detector.set_conf(val)

    def _scale_changed(self, v):
        val = v / 100
        self.lbl_scale.setText(f"{val:.2f}")
        self.detector.set_scale(val)

    # ─────────────────────────────────────────────
    # ACTIONS
    # ─────────────────────────────────────────────

    def _snapshot(self):
        if self._last_frame is None:
            return
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
        fname = f"{SNAPSHOTS_DIR}/snap_{int(time.time())}.jpg"
        cv2.imwrite(fname, self._last_frame)
        self._flash(f"Snapshot → {fname}", C["green"])

    def _toggle_record(self):
        if not self._recording:
            # frame.shape = (height, width, channels) — must pass (width, height) to writer
            if self._last_frame is not None:
                frame_h, frame_w = self._last_frame.shape[:2]
            else:
                frame_w, frame_h = 640, 480
            fname = self.detector.start_recording(frame_w, frame_h)
            self.detector.save_video = True
            self._recording = True
            self.btn_rec.setObjectName("btnRecActive")
            self.btn_rec.setText("⏹  STOP REC")
            self.btn_rec.setStyle(self.btn_rec.style())
            self._flash(f"Recording → {fname}", C["red"])
        else:
            self.detector.stop_recording()
            self.detector.save_video = False
            self._recording = False
            self.btn_rec.setObjectName("btnRec")
            self.btn_rec.setText("⏺  RECORD")
            self.btn_rec.setStyle(self.btn_rec.style())
            self._flash("Recording saved.", C["accent"])

    def _flash(self, msg, color=C["accent"]):
        self.lbl_event_txt.setText(msg)
        self.lbl_event_txt.setStyleSheet(f"color:{color}; font-weight:bold;")
        QTimer.singleShot(3500, lambda: (
            self.lbl_event_txt.setText("No events detected"),
            self.lbl_event_txt.setStyleSheet(f"color:{C['muted']};"),
        ))

    # ─────────────────────────────────────────────
    # CLOSE
    # ─────────────────────────────────────────────

    def closeEvent(self, e):
        if self.cam_thread:
            self.cam_thread.stop()
        self.detector.release()
        e.accept()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ███  SECTION 8: ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Vision")

    # Show splash
    splash = SplashScreen()
    splash.show()
    splash.set_status("Loading YOLO models…")
    app.processEvents()

    # Build main window (hidden for now)
    win = MainWindow()

    # Wire splash → loader → show window
    def on_ready():
        splash.close()
        win.show()

    win._on_models_ready_original = win._on_models_ready
    def patched_ready():
        win._on_models_ready_original()
        win.show()
    win._on_models_ready = patched_ready
    win._loader_patch    = True

    # Kick off model loading (splash stays visible during this)
    win.start_loading(splash)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()