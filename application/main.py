# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tkinter as tk
# from tkinter import ttk, messagebox
# import threading
# import time
# import numpy as np
# import tensorflow as tf
# import cv2
# import pyautogui
# from plyer import notification  # For system notifications

# class PopupGuardGUI:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("PopupGuard Pro v2.0")
#         self.root.geometry("500x350")
        
#         # Status variables
#         self.running = True
#         self.detection_active = False
#         self.current_status = "Initializing"
        
#         # Thresholds
#         self.REAL_POPUP_THRESHOLD = 0.85  # Confidence threshold for real popups
#         self.FAKE_POPUP_THRESHOLD = 0.65   # Confidence threshold for fake popups
        
#         # Create GUI elements
#         self.create_widgets()
        
#         # Initialize detector
#         self.detector = None
#         self.init_detector()

#     def create_widgets(self):
#         """Create all GUI components"""
#         # Status Frame
#         status_frame = ttk.LabelFrame(self.root, text="Detection Status", padding=10)
#         status_frame.pack(pady=10, padx=10, fill="x")
        
#         self.status_label = ttk.Label(
#             status_frame, 
#             text=self.current_status, 
#             font=('Helvetica', 12)
#         )
#         self.status_label.pack()
        
#         self.type_label = ttk.Label(
#             status_frame,
#             text="Type: None",
#             font=('Helvetica', 10)
#         )
#         self.type_label.pack()
        
#         self.confidence_label = ttk.Label(
#             status_frame,
#             text="Confidence: N/A",
#             font=('Helvetica', 10)
#         )
#         self.confidence_label.pack()
        
#         # Controls Frame
#         controls_frame = ttk.Frame(self.root)
#         controls_frame.pack(pady=10)
        
#         self.start_button = ttk.Button(
#             controls_frame,
#             text="Start Detection",
#             command=self.start_detection
#         )
#         self.start_button.pack(side="left", padx=5)
        
#         self.stop_button = ttk.Button(
#             controls_frame,
#             text="Stop Detection",
#             command=self.stop_detection,
#             state="disabled"
#         )
#         self.stop_button.pack(side="left", padx=5)
        
#         # Settings Frame
#         settings_frame = ttk.LabelFrame(self.root, text="Detection Settings", padding=10)
#         settings_frame.pack(pady=5, padx=10, fill="x")
        
#         ttk.Label(settings_frame, text="Real Popup Threshold:").pack(side="left")
#         self.real_thresh = ttk.Scale(
#             settings_frame,
#             from_=0.5,
#             to=1.0,
#             value=self.REAL_POPUP_THRESHOLD,
#             command=lambda v: setattr(self, 'REAL_POPUP_THRESHOLD', float(v))
#         )
#         self.real_thresh.pack(side="left", padx=5, fill="x", expand=True)
        
#         ttk.Label(settings_frame, text="Fake Popup Threshold:").pack(side="left")
#         self.fake_thresh = ttk.Scale(
#             settings_frame,
#             from_=0.3,
#             to=0.8,
#             value=self.FAKE_POPUP_THRESHOLD,
#             command=lambda v: setattr(self, 'FAKE_POPUP_THRESHOLD', float(v))
#         )
#         self.fake_thresh.pack(side="left", padx=5, fill="x", expand=True)
        
#         # Log Frame
#         log_frame = ttk.LabelFrame(self.root, text="Activity Log", padding=10)
#         log_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
#         self.log_text = tk.Text(log_frame, height=10, wrap="word")
#         self.log_text.pack(fill="both", expand=True)
        
#         scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
#         scrollbar.pack(side="right", fill="y")
#         self.log_text.config(yscrollcommand=scrollbar.set)

#     def init_detector(self):
#         """Initialize the detector in background"""
#         def initialize():
#             try:
#                 self.log("Loading AI model...")
#                 self.detector = PopupDetector()
#                 self.update_status("Ready")
#                 self.log("System ready! Click Start Detection")
#                 self.start_button.config(state="normal")
#             except Exception as e:
#                 self.log(f"Initialization failed: {str(e)}", error=True)
#                 messagebox.showerror("Error", f"Failed to initialize:\n{str(e)}")
        
#         threading.Thread(target=initialize, daemon=True).start()

#     def start_detection(self):
#         """Start the detection process"""
#         if not self.detector:
#             messagebox.showerror("Error", "Detector not initialized")
#             return
            
#         self.detection_active = True
#         self.start_button.config(state="disabled")
#         self.stop_button.config(state="normal")
#         self.update_status("Scanning...")
#         self.log("Detection started")
        
#         def detection_loop():
#             while self.running and self.detection_active:
#                 try:
#                     result = self.detector.detect()
#                     if result["detected"]:
#                         popup_type, action = self.classify_popup(result['confidence'])
#                         self.update_status(f"{popup_type} Detected!", "red")
#                         self.type_label.config(text=f"Type: {popup_type}")
#                         self.confidence_label.config(text=f"Confidence: {result['confidence']:.2f}")
#                         self.log(f"Detected {popup_type.lower()} popup! Confidence: {result['confidence']:.2f}")
#                         self.show_notification(popup_type, result['confidence'])
#                     else:
#                         self.update_status("Scanning...", "blue")
#                         self.type_label.config(text="Type: None")
#                 except Exception as e:
#                     self.log(f"Detection error: {str(e)}", error=True)
#                 time.sleep(1)
        
#         threading.Thread(target=detection_loop, daemon=True).start()

#     def classify_popup(self, confidence):
#         """Classify popup as real or fake"""
#         if confidence >= self.REAL_POPUP_THRESHOLD:
#             return "REAL", "block"
#         elif confidence >= self.FAKE_POPUP_THRESHOLD:
#             return "FAKE", "warn"
#         else:
#             return "UNKNOWN", "ignore"

#     def show_notification(self, popup_type, confidence):
#         """Show system notification"""
#         try:
#             notification.notify(
#                 title=f"PopupGuard: {popup_type} Popup Detected",
#                 message=f"Confidence: {confidence:.2f}\nAction: {self.classify_popup(confidence)[1].upper()}",
#                 app_name="PopupGuard Pro",
#                 timeout=5
#             )
#         except Exception as e:
#             self.log(f"Notification failed: {str(e)}", error=True)

#     def update_status(self, text, color="black"):
#         """Update status label"""
#         self.current_status = text
#         self.status_label.config(text=text, foreground=color)

#     def stop_detection(self):
#         """Stop the detection process"""
#         self.detection_active = False
#         self.start_button.config(state="normal")
#         self.stop_button.config(state="disabled")
#         self.update_status("Ready", "green")
#         self.log("Detection stopped")

#     def log(self, message, error=False):
#         """Add message to log with timestamp"""
#         timestamp = time.strftime("%H:%M:%S")
#         tag = "ERROR" if error else "INFO"
#         self.log_text.insert("end", f"[{timestamp}] {tag}: {message}\n")
#         self.log_text.see("end")
#         if error:
#             self.log_text.tag_add("error", "end-1l", "end")
#             self.log_text.tag_config("error", foreground="red")

#     def shutdown(self):
#         """Proper shutdown procedure"""
#         self.running = False
#         self.root.destroy()

# class PopupDetector:
#     """Core detection logic"""
#     def __init__(self):
#         self.model = self.load_model()
#         self.qr_detector = cv2.QRCodeDetector()
        
#     def load_model(self):
#         """Load the trained model"""
#         model_path = os.path.join(
#             os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#             "system",
#             "model",
#             "popup_classifier.h5"
#         )
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model file not found at {model_path}")
#         return tf.keras.models.load_model(model_path)
    
#     def detect(self):
#         """Perform one detection cycle"""
#         screenshot = pyautogui.screenshot()
#         image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
#         # QR detection
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         has_qr, _ = self.qr_detector.detect(gray)
        
#         if has_qr:
#             # Popup classification
#             img = cv2.resize(image, (224, 224))
#             img_array = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
#             confidence = float(self.model.predict(img_array, verbose=0)[0][0])
#             return {"detected": True, "confidence": confidence}
#         return {"detected": False, "confidence": 0.0}

# if __name__ == "__main__":
#     # Install plyer if not available: pip install plyer
#     root = tk.Tk()
#     app = PopupGuardGUI(root)
#     root.protocol("WM_DELETE_WINDOW", app.shutdown)
#     root.mainloop()


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np
import tensorflow as tf
import cv2
import pyautogui
from plyer import notification
import hashlib
from collections import deque

class PopupGuardGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PopupGuard Pro v4.0")
        self.root.geometry("600x400")
        
        # Detection settings
        self.detection_active = False
        self.current_popups = {}
        self.popup_history = deque(maxlen=20)
        self.SCAN_INTERVAL = 0.5  # Seconds between full scans
        self.DETECTION_PAUSE = 0.1  # Seconds between quick checks
        
        # Create GUI
        self.create_widgets()
        self.detector = PopupDetector()
        self.log("System initialized. Ready to scan.")

    def create_widgets(self):
        """Initialize all GUI components"""
        # Status Frame
        status_frame = ttk.LabelFrame(self.root, text="Detection Status", padding=10)
        status_frame.pack(pady=10, padx=10, fill="x")
        
        self.status_label = ttk.Label(
            status_frame, 
            text="Ready to scan", 
            font=('Helvetica', 12)
        )
        self.status_label.pack()
        
        self.stats_label = ttk.Label(
            status_frame,
            text="Stats: 0 scans | 0 popups",
            font=('Helvetica', 9)
        )
        self.stats_label.pack()
        
        # Controls
        controls_frame = ttk.Frame(self.root)
        controls_frame.pack(pady=10)
        
        self.start_button = ttk.Button(
            controls_frame,
            text="Start Detection",
            command=self.start_detection
        )
        self.start_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(
            controls_frame,
            text="Stop Detection",
            command=self.stop_detection,
            state="disabled"
        )
        self.stop_button.pack(side="left", padx=5)
        
        # Log
        log_frame = ttk.LabelFrame(self.root, text="Detection Log", padding=10)
        log_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.log_text = tk.Text(log_frame, height=12, wrap="word")
        self.log_text.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

    def start_detection(self):
        """Start the scanning process"""
        self.detection_active = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Scanning...", foreground="blue")
        self.log("Starting detection system")
        
        def scan_loop():
            scan_count = 0
            last_full_scan = 0
            
            while self.detection_active:
                try:
                    current_time = time.time()
                    
                    # Full scan every SCAN_INTERVAL seconds
                    if current_time - last_full_scan >= self.SCAN_INTERVAL:
                        screenshot = pyautogui.screenshot()
                        detected_popups = self.detector.detect_popups(screenshot)
                        scan_count += 1
                        last_full_scan = current_time
                        
                        # Process new popups
                        new_popups = {}
                        for popup_hash, (popup_type, confidence) in detected_popups.items():
                            if popup_hash not in self.current_popups:
                                self.handle_new_popup(popup_hash, popup_type, confidence)
                            new_popups[popup_hash] = (popup_type, confidence, current_time)
                        
                        # Check disappeared popups
                        expired = set(self.current_popups.keys()) - set(new_popups.keys())
                        for popup_hash in expired:
                            self.log(f"Popup disappeared: {self.current_popups[popup_hash][0]}")
                        
                        self.current_popups = new_popups
                        self.stats_label.config(text=f"Stats: {scan_count} scans | {len(self.current_popups)} popups")
                    
                    time.sleep(self.DETECTION_PAUSE)
                    
                except Exception as e:
                    self.log(f"Scan error: {str(e)}", error=True)
                    time.sleep(1)
        
        threading.Thread(target=scan_loop, daemon=True).start()

    def handle_new_popup(self, popup_hash, popup_type, confidence):
        """Handle a newly detected popup"""
        self.popup_history.append((time.time(), popup_hash, popup_type, confidence))
        self.log(f"New {popup_type} popup (Confidence: {confidence:.2f})")
        
        # Show notification after verification delay
        threading.Timer(0.3, lambda: self._verify_and_notify(popup_hash, popup_type, confidence)).start()

    def _verify_and_notify(self, popup_hash, popup_type, confidence):
        """Verify popup still exists and show notification"""
        if self.detection_active and popup_hash in self.current_popups:
            self.show_notification(popup_type, confidence)
            self.status_label.config(
                text=f"{popup_type} popup detected!",
                foreground="red" if popup_type == "REAL" else "orange"
            )

    def show_notification(self, popup_type, confidence):
        """Show system notification"""
        try:
            notification.notify(
                title=f"{popup_type} Popup Detected",
                message=f"Confidence: {confidence:.2f}",
                timeout=3
            )
        except Exception as e:
            self.log(f"Notification failed: {str(e)}", error=True)

    def stop_detection(self):
        """Stop the scanning process"""
        self.detection_active = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Ready to scan", foreground="green")
        self.log("Detection stopped")

    def log(self, message, error=False):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        tag = "ERROR" if error else "INFO"
        self.log_text.insert("end", f"[{timestamp}] {tag}: {message}\n")
        self.log_text.see("end")
        if error:
            self.log_text.tag_add("error", "end-1l", "end")
            self.log_text.tag_config("error", foreground="red")

    def shutdown(self):
        """Clean shutdown"""
        self.detection_active = False
        self.root.destroy()

class PopupDetector:
    """Robust popup detection system"""
    def __init__(self):
        self.model = self.load_model()
        self.qr_detector = cv2.QRCodeDetector()
        self.last_scan_time = 0
        self.scan_count = 0
        
        print("Popup detector initialized")

    def load_model(self):
        """Load the trained model"""
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "system",
            "model",
            "popup_classifier.h5"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded. Input shape: {model.input_shape}")
        return model

    def detect_popups(self, screenshot):
        """Detect and classify popups in screenshot"""
        try:
            image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect QR codes
            retval, _, points, _ = self.qr_detector.detectAndDecodeMulti(gray)
            current_popups = {}
            
            if retval:
                for i in range(len(points)):
                    if len(points[i]) > 0:
                        try:
                            # Extract QR region
                            x, y, w, h = cv2.boundingRect(points[i].astype(int))
                            qr_img = gray[y:y+h, x:x+w]
                            
                            # Classify
                            popup_type, confidence = self.classify_popup(qr_img)
                            popup_hash = hashlib.md5(qr_img.tobytes()).hexdigest()
                            current_popups[popup_hash] = (popup_type, confidence)
                            
                            print(f"Detected {popup_type} popup at ({x},{y})")
                        except Exception as e:
                            print(f"Error processing QR: {str(e)}")
            
            self.scan_count += 1
            return current_popups
            
        except Exception as e:
            print(f"Detection error: {str(e)}")
            return {}

    def classify_popup(self, qr_img):
        """Classify a single popup"""
        # Convert to RGB if grayscale
        if len(qr_img.shape) == 2:
            qr_img = cv2.cvtColor(qr_img, cv2.COLOR_GRAY2RGB)
        
        # Prepare image for model
        img = cv2.resize(qr_img, (224, 224))
        img_array = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        
        # Verify shape
        if img_array.shape != (1, 224, 224, 3):
            print(f"WARNING: Reshaping image from {img_array.shape}")
            img_array = img_array.reshape(1, 224, 224, 3)
        
        # Get prediction
        confidence = float(self.model.predict(img_array, verbose=0)[0][0])
        
        # Classify
        if confidence >= 0.85:
            return "REAL", confidence
        elif confidence >= 0.60:
            return "FAKE", confidence
        else:
            return "UNKNOWN", confidence

if __name__ == "__main__":
    # Create required folders
    os.makedirs('system/model', exist_ok=True)
    
    root = tk.Tk()
    app = PopupGuardGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.shutdown)
    root.mainloop()


















