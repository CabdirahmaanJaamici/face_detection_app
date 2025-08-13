import cv2
import os
import datetime
import time
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk, messagebox, Canvas, Scrollbar, filedialog
import threading
import queue
from collections import deque
import pygame
import csv

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Face Detection System")
        self.root.geometry("1200x800")
        
        # Theme variables
        self.themes = {
            "light": {
                "bg": "#e6f2ff",
                "frame_bg": "#cce6ff",
                "title_bg": "#4d94ff",
                "text": "#333333",
                "status_text": "#0066cc",
                "button": {
                    "start": "#00cc66",
                    "stop": "#ff6666",
                    "gallery": "#9966ff",
                    "clear": "#ffcc66",
                    "exit": "#999999"
                }
            },
            "dark": {
                "bg": "#2c3e50",
                "frame_bg": "#34495e",
                "title_bg": "#1a252f",
                "text": "#ecf0f1",
                "status_text": "#3498db",
                "button": {
                    "start": "#27ae60",
                    "stop": "#e74c3c",
                    "gallery": "#8e44ad",
                    "clear": "#f39c12",
                    "exit": "#7f8c8d"
                }
            }
        }
        self.current_theme = "light"
        
        # Initialize pygame for sound
        pygame.mixer.init()
        self.detection_sound = None
        try:
            # Try to load a sound file
            self.detection_sound = pygame.mixer.Sound("detection.wav")
        except:
            print("Sound file not found. Sound notifications disabled.")
        
        # Set initial theme
        self.set_theme(self.current_theme)
        
        # Create directories
        self.SAVE_DIR = "detected_faces"
        self.KNOWN_FACES_DIR = "known_faces"
        self.TEMP_DIR = "temp_faces"
        self.ATTENDANCE_FILE = "attendance.csv"
        self.SCREENSHOTS_DIR = "screenshots"
        self.RECORDINGS_DIR = "recordings"
        
        for dir_path in [self.SAVE_DIR, self.KNOWN_FACES_DIR, self.TEMP_DIR, 
                        self.SCREENSHOTS_DIR, self.RECORDINGS_DIR]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # Initialize attendance file
        if not os.path.exists(self.ATTENDANCE_FILE):
            with open(self.ATTENDANCE_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Timestamp", "Status"])
        
        # Load pre-trained classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Try to load face landmark detector
        self.landmarks_available = False
        try:
            # For OpenCV 4.x, we need to use contrib modules
            if hasattr(cv2, 'face'):
                self.face_landmark_detector = cv2.face.createFacemarkLBF()
                # Try to load the model (you'll need to download lbfmodel.yaml)
                try:
                    self.face_landmark_detector.loadModel("lbfmodel.yaml")
                    self.landmarks_available = True
                    print("Face landmark detector loaded successfully")
                except:
                    print("lbfmodel.yaml not found. Landmarks will not be shown.")
            else:
                print("OpenCV contrib modules not available. Landmarks will not be shown.")
        except Exception as e:
            print(f"Error loading face landmark detector: {e}")
        
        # Initialize variables
        self.cap = None
        self.running = False
        self.recording = False
        self.video_writer = None
        self.frame_queue = deque(maxlen=5)
        self.status_queue = queue.Queue()
        
        # Tracking variables
        self.saved_faces = []
        self.SAVE_INTERVAL = 3.0
        self.DISTANCE_THRESHOLD = 50
        
        # Face recognition variables
        self.known_faces = []
        self.known_face_names = []
        self.next_person_id = 1
        self.unknown_faces_buffer = []
        self.BUFFER_SIZE = 3
        self.PROCESS_INTERVAL = 5.0
        self.last_process_time = time.time()
        
        # Performance optimization variables
        self.frame_skip = 1
        self.frame_count = 0
        self.MIN_FACE_SIZE = 60
        self.MIN_RECOGNITION_SIZE = 100
        self.MATCH_THRESHOLD = 0.65
        
        # Resolution settings
        self.resolutions = {
            "640x480": (640, 480),
            "800x600": (800, 600),
            "1024x768": (1024, 768),
            "1280x720": (1280, 720)
        }
        self.current_resolution = "640x480"
        
        # Feature toggles
        self.show_emotions = tk.BooleanVar(value=True)
        self.show_landmarks = tk.BooleanVar(value=True)
        self.blur_faces = tk.BooleanVar(value=False)
        self.play_sounds = tk.BooleanVar(value=True)
        self.show_eye_tracking = tk.BooleanVar(value=True)
        self.show_smile_detection = tk.BooleanVar(value=True)
        
        # Camera error handling
        self.camera_error_count = 0
        self.max_camera_errors = 5
        
        # Load known faces
        self.load_known_faces()
        
        # Create GUI elements
        self.create_widgets()
        
        # Start GUI update loop
        self.update_gui()
        
        # Initialize gallery
        self.gallery_window = None
        self.gallery_images = []
    
    def set_theme(self, theme_name):
        theme = self.themes[theme_name]
        self.root.configure(bg=theme["bg"])
        self.current_theme = theme_name
    
    def create_widgets(self):
        theme = self.themes[self.current_theme]
        
        # Create main container with scrollbar
        main_container = tk.Frame(self.root, bg=theme["bg"])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create canvas for scrolling
        self.canvas = Canvas(main_container, bg=theme["bg"], highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create scrollbar
        scrollbar = Scrollbar(main_container, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure canvas
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        # Create scrollable frame
        self.scrollable_frame = tk.Frame(self.canvas, bg=theme["bg"])
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Title
        title_frame = tk.Frame(self.scrollable_frame, bg=theme["title_bg"], relief=tk.RAISED, bd=2)
        title_frame.pack(fill=tk.X, padx=10, pady=5)
        
        title_label = tk.Label(title_frame, text="Advanced Face Detection System", 
                              font=('Arial', 24, 'bold'), bg=theme["title_bg"], fg='white')
        title_label.pack(pady=10)
        
        # Content container
        content_container = tk.Frame(self.scrollable_frame, bg=theme["bg"])
        content_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Video and controls
        left_panel = tk.Frame(content_container, bg=theme["bg"])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Video frame
        video_frame = tk.Frame(left_panel, bg=theme["frame_bg"], relief=tk.RAISED, bd=2)
        video_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Status frame
        status_frame = tk.Frame(left_panel, bg=theme["frame_bg"], relief=tk.RAISED, bd=2)
        status_frame.pack(pady=10, padx=10, fill=tk.X)
        
        self.status_label = tk.Label(status_frame, text="Status: Ready", 
                                    font=('Arial', 14), bg=theme["frame_bg"], fg=theme["status_text"])
        self.status_label.pack(pady=5)
        
        self.faces_label = tk.Label(status_frame, text=f"Known Faces: {len(self.known_face_names)}", 
                                   font=('Arial', 14), bg=theme["frame_bg"], fg='#009933')
        self.faces_label.pack(pady=5)
        
        self.fps_label = tk.Label(status_frame, text="FPS: 0", 
                                 font=('Arial', 14), bg=theme["frame_bg"], fg='#cc6600')
        self.fps_label.pack(pady=5)
        
        # Control buttons frame
        control_frame = tk.Frame(left_panel, bg=theme["bg"])
        control_frame.pack(pady=10)
        
        # First row of buttons
        button_frame1 = tk.Frame(control_frame, bg=theme["bg"])
        button_frame1.pack(pady=5)
        
        self.start_button = tk.Button(button_frame1, text="Start Detection", 
                                     command=self.start_detection, 
                                     bg=theme["button"]["start"], fg='white', 
                                     font=('Arial', 12, 'bold'),
                                     width=15, height=2,
                                     relief=tk.RAISED, bd=3)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(button_frame1, text="Stop Detection", 
                                    command=self.stop_detection, 
                                    bg=theme["button"]["stop"], fg='white', 
                                    font=('Arial', 12, 'bold'),
                                    width=15, height=2,
                                    relief=tk.RAISED, bd=3, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.record_button = tk.Button(button_frame1, text="Start Recording", 
                                      command=self.toggle_recording, 
                                      bg=theme["button"]["gallery"], fg='white', 
                                      font=('Arial', 12, 'bold'),
                                      width=15, height=2,
                                      relief=tk.RAISED, bd=3, state=tk.DISABLED)
        self.record_button.pack(side=tk.LEFT, padx=5)
        
        # Second row of buttons
        button_frame2 = tk.Frame(control_frame, bg=theme["bg"])
        button_frame2.pack(pady=5)
        
        self.screenshot_button = tk.Button(button_frame2, text="Screenshot", 
                                          command=self.take_screenshot, 
                                          bg=theme["button"]["clear"], fg='white', 
                                          font=('Arial', 12, 'bold'),
                                          width=15, height=2,
                                          relief=tk.RAISED, bd=3)
        self.screenshot_button.pack(side=tk.LEFT, padx=5)
        
        self.gallery_button = tk.Button(button_frame2, text="View Gallery", 
                                       command=self.open_gallery, 
                                       bg=theme["button"]["gallery"], fg='white', 
                                       font=('Arial', 12, 'bold'),
                                       width=15, height=2,
                                       relief=tk.RAISED, bd=3)
        self.gallery_button.pack(side=tk.LEFT, padx=5)
        
        # Third row of buttons
        button_frame3 = tk.Frame(control_frame, bg=theme["bg"])
        button_frame3.pack(pady=5)
        
        self.clear_button = tk.Button(button_frame3, text="Clear Known Faces", 
                                    command=self.clear_known_faces, 
                                    bg=theme["button"]["clear"], fg='white', 
                                    font=('Arial', 12, 'bold'),
                                    width=15, height=2,
                                    relief=tk.RAISED, bd=3)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        self.theme_button = tk.Button(button_frame3, text="Toggle Theme", 
                                    command=self.toggle_theme, 
                                    bg=theme["button"]["exit"], fg='white', 
                                    font=('Arial', 12, 'bold'),
                                    width=15, height=2,
                                    relief=tk.RAISED, bd=3)
        self.theme_button.pack(side=tk.LEFT, padx=5)
        
        self.exit_button = tk.Button(button_frame3, text="Exit", 
                                    command=self.exit_app, 
                                    bg=theme["button"]["exit"], fg='white', 
                                    font=('Arial', 12, 'bold'),
                                    width=15, height=2,
                                    relief=tk.RAISED, bd=3)
        self.exit_button.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Settings and resolution
        right_panel = tk.Frame(content_container, bg=theme["bg"], width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        right_panel.pack_propagate(False)
        
        # Settings frame
        settings_frame = tk.LabelFrame(right_panel, text="Settings", 
                                     font=('Arial', 16, 'bold'), 
                                     bg=theme["frame_bg"], fg=theme["status_text"],
                                     relief=tk.RAISED, bd=2)
        settings_frame.pack(pady=10, padx=5, fill=tk.X)
        
        # Save interval
        tk.Label(settings_frame, text="Save Interval (s):", 
                font=('Arial', 12), bg=theme["frame_bg"], fg=theme["text"]).pack(pady=5)
        self.save_interval_var = tk.DoubleVar(value=self.SAVE_INTERVAL)
        save_interval_scale = tk.Scale(settings_frame, from_=1, to=10, 
                                      orient=tk.HORIZONTAL, 
                                      variable=self.save_interval_var,
                                      command=self.update_save_interval,
                                      bg=theme["frame_bg"], fg=theme["text"],
                                      length=200)
        save_interval_scale.pack(pady=5)
        
        # Match threshold
        tk.Label(settings_frame, text="Match Threshold:", 
                font=('Arial', 12), bg=theme["frame_bg"], fg=theme["text"]).pack(pady=5)
        self.match_threshold_var = tk.DoubleVar(value=self.MATCH_THRESHOLD)
        match_threshold_scale = tk.Scale(settings_frame, from_=0.5, to=0.9, 
                                       resolution=0.05,
                                       orient=tk.HORIZONTAL, 
                                       variable=self.match_threshold_var,
                                       command=self.update_match_threshold,
                                       bg=theme["frame_bg"], fg=theme["text"],
                                       length=200)
        match_threshold_scale.pack(pady=5)
        
        # Features frame
        features_frame = tk.LabelFrame(right_panel, text="Features", 
                                     font=('Arial', 16, 'bold'), 
                                     bg=theme["frame_bg"], fg=theme["status_text"],
                                     relief=tk.RAISED, bd=2)
        features_frame.pack(pady=10, padx=5, fill=tk.X)
        
        # Feature checkboxes
        tk.Checkbutton(features_frame, text="Show Emotions (Smile Detection)", 
                      variable=self.show_emotions,
                      font=('Arial', 12), 
                      bg=theme["frame_bg"], fg=theme["text"],
                      selectcolor=theme["frame_bg"]).pack(anchor=tk.W, padx=20, pady=2)
        
        tk.Checkbutton(features_frame, text="Show Landmarks", 
                      variable=self.show_landmarks,
                      font=('Arial', 12), 
                      bg=theme["frame_bg"], fg=theme["text"],
                      selectcolor=theme["frame_bg"]).pack(anchor=tk.W, padx=20, pady=2)
        
        tk.Checkbutton(features_frame, text="Eye Tracking", 
                      variable=self.show_eye_tracking,
                      font=('Arial', 12), 
                      bg=theme["frame_bg"], fg=theme["text"],
                      selectcolor=theme["frame_bg"]).pack(anchor=tk.W, padx=20, pady=2)
        
        tk.Checkbutton(features_frame, text="Smile Detection", 
                      variable=self.show_smile_detection,
                      font=('Arial', 12), 
                      bg=theme["frame_bg"], fg=theme["text"],
                      selectcolor=theme["frame_bg"]).pack(anchor=tk.W, padx=20, pady=2)
        
        tk.Checkbutton(features_frame, text="Blur Faces", 
                      variable=self.blur_faces,
                      font=('Arial', 12), 
                      bg=theme["frame_bg"], fg=theme["text"],
                      selectcolor=theme["frame_bg"]).pack(anchor=tk.W, padx=20, pady=2)
        
        tk.Checkbutton(features_frame, text="Play Sounds", 
                      variable=self.play_sounds,
                      font=('Arial', 12), 
                      bg=theme["frame_bg"], fg=theme["text"],
                      selectcolor=theme["frame_bg"]).pack(anchor=tk.W, padx=20, pady=2)
        
        # Resolution frame
        resolution_frame = tk.LabelFrame(right_panel, text="Camera Resolution", 
                                       font=('Arial', 16, 'bold'), 
                                       bg=theme["frame_bg"], fg=theme["status_text"],
                                       relief=tk.RAISED, bd=2)
        resolution_frame.pack(pady=10, padx=5, fill=tk.X)
        
        self.resolution_var = tk.StringVar(value=self.current_resolution)
        for res in self.resolutions:
            rb = tk.Radiobutton(resolution_frame, text=res, 
                               variable=self.resolution_var, 
                               value=res,
                               command=self.change_resolution,
                               font=('Arial', 12), 
                               bg=theme["frame_bg"], fg=theme["text"],
                               selectcolor=theme["frame_bg"])
            rb.pack(anchor=tk.W, padx=20, pady=2)
        
        # Statistics frame
        stats_frame = tk.LabelFrame(right_panel, text="Statistics", 
                                   font=('Arial', 16, 'bold'), 
                                   bg=theme["frame_bg"], fg=theme["status_text"],
                                   relief=tk.RAISED, bd=2)
        stats_frame.pack(pady=10, padx=5, fill=tk.X)
        
        self.stats_text = tk.Text(stats_frame, height=10, width=30, 
                                 font=('Arial', 10), 
                                 bg='white', fg=theme["text"],
                                 relief=tk.SUNKEN, bd=2)
        self.stats_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        
        # Add scrollbar to statistics
        stats_scrollbar = Scrollbar(stats_frame, command=self.stats_text.yview)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_text.config(yscrollcommand=stats_scrollbar.set)
        
        # Update scrollregion
        self.scrollable_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def toggle_theme(self):
        # Switch theme
        if self.current_theme == "light":
            self.set_theme("dark")
        else:
            self.set_theme("light")
        
        # Recreate widgets with new theme
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.create_widgets()
    
    def take_screenshot(self):
        if self.frame_queue:
            # Get the latest frame
            frame_rgb = self.frame_queue[-1]
            
            # Convert to BGR for saving
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.SCREENSHOTS_DIR}/screenshot_{timestamp}.jpg"
            
            # Save the image
            cv2.imwrite(filename, frame_bgr)
            
            # Notify user
            self.status_queue.put(f"Screenshot saved: {filename}")
            
            # Play sound if enabled
            if self.play_sounds.get() and self.detection_sound:
                self.detection_sound.play()
    
    def toggle_recording(self):
        if not self.recording:
            # Start recording
            self.recording = True
            self.record_button.config(text="Stop Recording")
            
            # Get current resolution
            width, height = self.resolutions[self.current_resolution]
            
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.RECORDINGS_DIR}/recording_{timestamp}.avi"
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
            
            # Notify user
            self.status_queue.put(f"Recording started: {filename}")
        else:
            # Stop recording
            self.recording = False
            self.record_button.config(text="Start Recording")
            
            # Release video writer
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            # Notify user
            self.status_queue.put("Recording stopped")
    
    def update_save_interval(self, value):
        self.SAVE_INTERVAL = float(value)
    
    def update_match_threshold(self, value):
        self.MATCH_THRESHOLD = float(value)
    
    def change_resolution(self):
        # Store the new resolution
        new_resolution = self.resolution_var.get()
        
        # If the resolution hasn't changed, do nothing
        if new_resolution == self.current_resolution:
            return
        
        # If detection is running, stop it first
        was_running = self.running
        if was_running:
            self.stop_detection()
            # Wait a moment for the camera to release
            time.sleep(0.5)
        
        # Update the current resolution
        self.current_resolution = new_resolution
        
        # If detection was running, restart it
        if was_running:
            self.start_detection()
        
        # Notify user
        self.status_queue.put(f"Resolution changed to {self.current_resolution}")
    
    def load_known_faces(self):
        image_paths = [os.path.join(self.KNOWN_FACES_DIR, f) 
                      for f in os.listdir(self.KNOWN_FACES_DIR) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_path in image_paths:
            pil_image = Image.open(image_path).convert('L')
            image_np = np.array(pil_image, 'uint8')
            name = os.path.splitext(os.path.basename(image_path))[0]
            
            faces = self.face_cascade.detectMultiScale(image_np, scaleFactor=1.2, minNeighbors=4, minSize=(60, 60))
            
            for (x, y, w, h) in faces:
                face_roi = image_np[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (80, 80))
                self.known_faces.append(face_roi)
                self.known_face_names.append(name)
        
        # Set next_person_id based on existing Person_ names
        for name in self.known_face_names:
            if name.startswith("Person_"):
                try:
                    person_id = int(name.split("_")[1])
                    if person_id >= self.next_person_id:
                        self.next_person_id = person_id + 1
                except:
                    pass
        
        print(f"Loaded {len(self.known_face_names)} known faces: {self.known_face_names}")
        print(f"Next person ID: {self.next_person_id}")
    
    def start_detection(self):
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.record_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Running", fg='#00cc66')
            
            # Initialize camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_queue.put("Error: Could not open camera")
                self.stop_detection()
                return
            
            # Set resolution
            width, height = self.resolutions[self.current_resolution]
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Verify the resolution was set correctly
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width != width or actual_height != height:
                self.status_queue.put(f"Warning: Requested {width}x{height}, got {actual_width}x{actual_height}")
            
            # Reset error counter
            self.camera_error_count = 0
            
            # Start detection thread
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            # Start FPS calculation
            self.last_frame_time = time.time()
            self.fps_frame_count = 0
    
    def stop_detection(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.record_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped", fg='#ff6666')
        
        # Stop recording if active
        if self.recording:
            self.toggle_recording()
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def clear_known_faces(self):
        result = messagebox.askyesno("Clear Known Faces", 
                                    "Are you sure you want to clear all known faces?")
        if result:
            # Delete all files in known_faces directory
            for filename in os.listdir(self.KNOWN_FACES_DIR):
                file_path = os.path.join(self.KNOWN_FACES_DIR, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            
            # Clear memory
            self.known_faces = []
            self.known_face_names = []
            self.next_person_id = 1
            
            # Update GUI
            self.faces_label.config(text=f"Known Faces: {len(self.known_face_names)}")
            self.status_queue.put("Cleared all known faces")
    
    def exit_app(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.gallery_window:
            self.gallery_window.destroy()
        self.root.quit()
        self.root.destroy()
    
    def open_gallery(self):
        if self.gallery_window is None or not self.gallery_window.winfo_exists():
            self.gallery_window = tk.Toplevel(self.root)
            self.gallery_window.title("Face Gallery")
            self.gallery_window.geometry("1000x700")
            theme = self.themes[self.current_theme]
            self.gallery_window.configure(bg=theme["bg"])
            
            # Create gallery frame
            gallery_frame = tk.Frame(self.gallery_window, bg=theme["bg"])
            gallery_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create canvas and scrollbar
            self.gallery_canvas = Canvas(gallery_frame, bg=theme["bg"])
            gallery_scrollbar = Scrollbar(gallery_frame, orient="vertical", command=self.gallery_canvas.yview)
            self.gallery_canvas.configure(yscrollcommand=gallery_scrollbar.set)
            
            # Create scrollable frame
            self.scrollable_frame = tk.Frame(self.gallery_canvas, bg=theme["bg"])
            self.scrollable_frame.bind(
                "<Configure>",
                lambda e: self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
            )
            
            self.gallery_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
            self.gallery_canvas.pack(side="left", fill="both", expand=True)
            gallery_scrollbar.pack(side="right", fill="y")
            
            # Load images
            self.load_gallery_images()
            
            # Close button
            close_button = tk.Button(self.gallery_window, text="Close Gallery", 
                                   command=self.gallery_window.destroy,
                                   bg=theme["button"]["stop"], fg='white', 
                                   font=('Arial', 12, 'bold'),
                                   width=15, height=2)
            close_button.pack(pady=10)
    
    def load_gallery_images(self):
        # Clear existing images
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Load images from both directories
        image_paths = []
        
        # Known faces
        for filename in os.listdir(self.KNOWN_FACES_DIR):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append((os.path.join(self.KNOWN_FACES_DIR, filename), "Known"))
        
        # Detected faces
        for filename in os.listdir(self.SAVE_DIR):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append((os.path.join(self.SAVE_DIR, filename), "Detected"))
        
        # Screenshots
        for filename in os.listdir(self.SCREENSHOTS_DIR):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append((os.path.join(self.SCREENSHOTS_DIR, filename), "Screenshot"))
        
        # Create image grid
        row, col = 0, 0
        max_cols = 4
        theme = self.themes[self.current_theme]
        
        for image_path, category in image_paths:
            try:
                # Load and resize image
                img = Image.open(image_path)
                img = img.resize((200, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Create frame for image and label
                img_frame = tk.Frame(self.scrollable_frame, bg=theme["frame_bg"], relief=tk.RAISED, bd=2)
                img_frame.grid(row=row, column=col, padx=10, pady=10)
                
                # Image label
                img_label = tk.Label(img_frame, image=photo, bg=theme["frame_bg"])
                img_label.image = photo  # Keep a reference
                img_label.pack(pady=5)
                
                # Filename label
                filename = os.path.basename(image_path)
                name_label = tk.Label(img_frame, text=filename, 
                                     font=('Arial', 10), bg=theme["frame_bg"], fg=theme["text"],
                                     wraplength=180)
                name_label.pack(pady=2)
                
                # Category label
                category_label = tk.Label(img_frame, text=category, 
                                        font=('Arial', 10, 'bold'), 
                                        bg=theme["frame_bg"], fg=theme["status_text"])
                category_label.pack(pady=2)
                
                # Update grid position
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
                    
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
        
        # Update scrollregion
        self.scrollable_frame.update_idletasks()
        self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
    
    def detection_loop(self):
        last_time = time.time()
        fps_update_time = time.time()
        fps_count = 0
        
        while self.running:
            try:
                # Check if camera is still open
                if self.cap is None or not self.cap.isOpened():
                    self.status_queue.put("Camera disconnected. Attempting to reconnect...")
                    self.cap = cv2.VideoCapture(0)
                    if not self.cap.isOpened():
                        self.camera_error_count += 1
                        if self.camera_error_count > self.max_camera_errors:
                            self.status_queue.put("Camera error. Stopping detection.")
                            self.stop_detection()
                            return
                        time.sleep(1)
                        continue
                    else:
                        # Reset camera settings
                        width, height = self.resolutions[self.current_resolution]
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        self.camera_error_count = 0
                        self.status_queue.put("Camera reconnected.")
                
                ret, frame = self.cap.read()
                if not ret:
                    self.camera_error_count += 1
                    if self.camera_error_count > self.max_camera_errors:
                        self.status_queue.put("Camera error. Stopping detection.")
                        self.stop_detection()
                        return
                    # Try to reconnect
                    self.cap.release()
                    self.cap = cv2.VideoCapture(0)
                    if self.cap.isOpened():
                        width, height = self.resolutions[self.current_resolution]
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        self.camera_error_count = 0
                    continue
                
                # Reset error counter on successful frame capture
                self.camera_error_count = 0
                
                current_time = time.time()
                
                # Calculate FPS
                fps_count += 1
                if current_time - fps_update_time >= 1.0:
                    fps = fps_count / (current_time - fps_update_time)
                    self.status_queue.put(f"FPS: {fps:.1f}")
                    fps_count = 0
                    fps_update_time = current_time
                
                # Process every frame for smoother display
                self.frame_count += 1
                
                # Create a copy of the frame for processing
                process_frame = frame.copy()
                
                # Reduce resolution for processing
                small_frame = cv2.resize(process_frame, (0, 0), fx=0.75, fy=0.75)
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=4,
                    minSize=(self.MIN_FACE_SIZE, self.MIN_FACE_SIZE)
                )
                
                # Process unknown faces periodically
                if current_time - self.last_process_time > self.PROCESS_INTERVAL:
                    self.process_unknown_faces()
                    self.last_process_time = current_time
                
                # Scale back to original frame size
                scale_x = frame.shape[1] / small_frame.shape[1]
                scale_y = frame.shape[0] / small_frame.shape[0]
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Scale coordinates back to original size
                    x = int(x * scale_x)
                    y = int(y * scale_y)
                    w = int(w * scale_x)
                    h = int(h * scale_y)
                    
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    save_face = True
                    
                    # Check against previously saved faces
                    for (saved_x, saved_y, saved_time) in self.saved_faces:
                        distance = ((center_x - saved_x)**2 + (center_y - saved_y)**2)**0.5
                        
                        if distance < self.DISTANCE_THRESHOLD and (current_time - saved_time) < self.SAVE_INTERVAL:
                            save_face = False
                            break
                    
                    # Blur face if enabled
                    if self.blur_faces.get():
                        # Extract the face ROI
                        face_roi = process_frame[y:y+h, x:x+w]
                        # Apply Gaussian blur
                        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                        # Put the blurred face back
                        process_frame[y:y+h, x:x+w] = blurred_face
                    
                    # Draw face rectangle with rounded corners
                    if save_face:
                        self.draw_rounded_rectangle(process_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    else:
                        self.draw_rounded_rectangle(process_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    
                    # Face recognition
                    name = "Unknown"
                    emotion = "Neutral"
                    smile_detected = False
                    
                    if w >= self.MIN_RECOGNITION_SIZE and h >= self.MIN_RECOGNITION_SIZE and len(self.known_faces) > 0:
                        face_roi = gray[int(y/scale_y):int((y+h)/scale_y), int(x/scale_x):int((x+w)/scale_y)]
                        face_roi = cv2.resize(face_roi, (80, 80))
                        
                        best_match_score = 0
                        best_match_name = "Unknown"
                        
                        for i, known_face in enumerate(self.known_faces):
                            result = cv2.matchTemplate(face_roi, known_face, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, _ = cv2.minMaxLoc(result)
                            
                            if max_val > best_match_score:
                                best_match_score = max_val
                                best_match_name = self.known_face_names[i]
                        
                        if best_match_score > self.MATCH_THRESHOLD:
                            name = best_match_name
                    
                    # Extract face for emotion analysis
                    if w >= self.MIN_RECOGNITION_SIZE and h >= self.MIN_RECOGNITION_SIZE:
                        face_img = frame[y:y+h, x:x+w]
                        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                        
                        # Eye tracking
                        if self.show_eye_tracking.get():
                            eyes = self.eye_cascade.detectMultiScale(
                                face_gray,
                                scaleFactor=1.1,
                                minNeighbors=5,
                                minSize=(15, 15)
                            )
                            
                            # Draw eye rectangles
                            for (ex, ey, ew, eh) in eyes:
                                # Scale eye coordinates back to original frame
                                ex = int(x + ex * scale_x)
                                ey = int(y + ey * scale_y)
                                ew = int(ew * scale_x)
                                eh = int(eh * scale_y)
                                cv2.rectangle(process_frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                        
                        # Smile detection (simple emotion detection)
                        if self.show_emotions.get() and self.show_smile_detection.get():
                            smiles = self.smile_cascade.detectMultiScale(
                                face_gray,
                                scaleFactor=1.7,
                                minNeighbors=20,
                                minSize=(15, 15)
                            )
                            
                            if len(smiles) > 0:
                                emotion = "Happy"
                                smile_detected = True
                                
                                # Draw smile rectangle
                                for (sx, sy, sw, sh) in smiles:
                                    # Scale smile coordinates back to original frame
                                    sx = int(x + sx * scale_x)
                                    sy = int(y + sy * scale_y)
                                    sw = int(sw * scale_x)
                                    sh = int(sh * scale_y)
                                    cv2.rectangle(process_frame, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 2)
                        
                        # Draw landmarks if enabled
                        if self.show_landmarks.get() and self.landmarks_available:
                            try:
                                # Detect landmarks
                                _, landmarks = self.face_landmark_detector.fit(face_gray, np.array([(x, y, w, h)]))
                                if landmarks:
                                    for landmark in landmarks[0]:
                                        # Scale landmarks back to original frame
                                        lx = int(landmark[0] * scale_x)
                                        ly = int(landmark[1] * scale_y)
                                        # Draw landmark point
                                        cv2.circle(process_frame, (lx, ly), 2, (0, 0, 255), -1)
                            except Exception as e:
                                # If landmark detection fails, just continue
                                pass
                    
                    # Display name with background
                    self.draw_text_with_background(process_frame, name, (x, y-10), 
                                                font_scale=0.7, thickness=2,
                                                bg_color=(0, 0, 0), fg_color=(255, 255, 255))
                    
                    # Display emotion if enabled
                    if self.show_emotions.get() and emotion != "Neutral":
                        self.draw_text_with_background(process_frame, emotion, (x, y+h+25), 
                                                    font_scale=0.6, thickness=2,
                                                    bg_color=(0, 0, 255), fg_color=(255, 255, 255))
                    
                    # Save detected face if it's a new detection and unknown
                    if save_face and name == "Unknown":
                        face_img = frame[y:y+h, x:x+w]
                        self.unknown_faces_buffer.append(face_img)
                        self.saved_faces.append((center_x, center_y, current_time))
                        
                        # Log attendance if known
                        if name != "Unknown":
                            self.log_attendance(name, emotion)
                        
                        # Play sound if enabled
                        if self.play_sounds.get() and self.detection_sound:
                            self.detection_sound.play()
                        
                        self.draw_text_with_background(process_frame, 'PROCESSING', (x, y+h+75), 
                                                    font_scale=0.6, thickness=2,
                                                    bg_color=(0, 0, 255), fg_color=(255, 255, 255))
                    elif save_face:
                        self.saved_faces.append((center_x, center_y, current_time))
                        
                        # Log attendance if known
                        if name != "Unknown":
                            self.log_attendance(name, emotion)
                        
                        self.draw_text_with_background(process_frame, 'SAVED', (x, y+h+75), 
                                                    font_scale=0.6, thickness=2,
                                                    bg_color=(0, 0, 255), fg_color=(255, 255, 255))
                
                # Clean up old saved faces
                self.saved_faces = [(x, y, t) for x, y, t in self.saved_faces if (current_time - t) < self.SAVE_INTERVAL]
                
                # Convert frame to RGB and put in queue
                frame_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                if len(self.frame_queue) < self.frame_queue.maxlen:
                    self.frame_queue.append(frame_rgb)
                
                # Write frame to video if recording
                if self.recording and self.video_writer:
                    self.video_writer.write(process_frame)
                
                # Control frame rate
                elapsed = time.time() - last_time
                if elapsed < 1.0 / 30.0:  # Target 30 FPS
                    time.sleep(1.0 / 30.0 - elapsed)
                last_time = time.time()
                
            except Exception as e:
                self.status_queue.put(f"Error in detection loop: {str(e)}")
                self.camera_error_count += 1
                if self.camera_error_count > self.max_camera_errors:
                    self.status_queue.put("Too many errors. Stopping detection.")
                    self.stop_detection()
                    return
                time.sleep(0.5)
    
    def log_attendance(self, name, emotion):
        try:
            with open(self.ATTENDANCE_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([name, timestamp, emotion])
        except Exception as e:
            print(f"Error logging attendance: {e}")
    
    def draw_rounded_rectangle(self, img, pt1, pt2, color, thickness, radius=10):
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw the main rectangle
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw the corners
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    
    def draw_text_with_background(self, img, text, position, font_scale=0.7, thickness=2, 
                                 bg_color=(0, 0, 0), fg_color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        x, y = position
        cv2.rectangle(img, (x, y - text_height - baseline), 
                     (x + text_width, y + baseline), 
                     bg_color, -1)
        cv2.putText(img, text, (x, y), font, font_scale, fg_color, thickness)
    
    def process_unknown_faces(self):
        if len(self.unknown_faces_buffer) < self.BUFFER_SIZE:
            return
        
        groups = []
        for i, face_img in enumerate(self.unknown_faces_buffer):
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (80, 80))
            
            matched = False
            for group in groups:
                result = cv2.matchTemplate(gray_face, group[0], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > 0.65:
                    group.append(gray_face)
                    matched = True
                    break
            
            if not matched:
                groups.append([gray_face])
        
        for group in groups:
            if len(group) >= 2:
                best_face = None
                best_size = 0
                
                for face in group:
                    h, w = face.shape
                    size = h * w
                    if size > best_size:
                        best_size = size
                        best_face = face
                
                best_face_bgr = cv2.cvtColor(best_face, cv2.COLOR_GRAY2BGR)
                self.add_new_person(best_face_bgr)
        
        self.unknown_faces_buffer = []
    
    def add_new_person(self, face_img):
        new_name = f"Person_{self.next_person_id}"
        self.next_person_id += 1
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.KNOWN_FACES_DIR}/{new_name}_{timestamp}.jpg"
        cv2.imwrite(filename, face_img)
        
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (80, 80))
        self.known_faces.append(gray_face)
        self.known_face_names.append(new_name)
        
        # Update GUI
        self.faces_label.config(text=f"Known Faces: {len(self.known_face_names)}")
        self.status_queue.put(f"Added new person: {new_name}")
        
        # Update statistics
        self.update_statistics()
    
    def update_statistics(self):
        stats = f"=== Face Detection Statistics ===\n\n"
        stats += f"Total Known Faces: {len(self.known_face_names)}\n"
        stats += f"Next Person ID: {self.next_person_id}\n"
        stats += f"Save Interval: {self.SAVE_INTERVAL}s\n"
        stats += f"Match Threshold: {self.MATCH_THRESHOLD}\n"
        stats += f"Current Resolution: {self.current_resolution}\n\n"
        
        stats += "=== Known Faces ===\n"
        for name in self.known_face_names:
            stats += f"- {name}\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats)
    
    def update_gui(self):
        # Update video feed
        try:
            if self.frame_queue:
                frame_rgb = self.frame_queue.popleft()
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=photo)
                self.video_label.image = photo
        except IndexError:
            pass
        
        # Update status messages
        try:
            while not self.status_queue.empty():
                message = self.status_queue.get_nowait()
                if message.startswith("FPS:"):
                    self.fps_label.config(text=message)
                else:
                    self.status_label.config(text=f"Status: {message}", fg='#ff9900')
                    self.root.after(3000, lambda: self.status_label.config(text="Status: Running", fg='#00cc66'))
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(15, self.update_gui)  # ~66 FPS for smoother display

def main():
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()