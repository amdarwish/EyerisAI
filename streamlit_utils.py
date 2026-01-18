import streamlit as st
import cv2
import numpy as np
import threading
import time
from datetime import datetime
import queue
import base64
from PIL import Image
import io

class CameraManager:
    """Manages camera operations for Streamlit interface"""
    
    def __init__(self):
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.running = False
        self.thread = None
        
    def start_camera(self, source):
        """Start camera capture"""
        if self.running:
            self.stop_camera()
            
        try:
            if isinstance(source, str):  # IP camera
                self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            else:  # Local camera
                self.cap = cv2.VideoCapture(source)
                
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera source: {source}")
                
            self.running = True
            self.thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.thread.start()
            
            return True
            
        except Exception as e:
            st.error(f"Failed to start camera: {str(e)}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        
        # Clear the frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def _capture_frames(self):
        """Capture frames in separate thread"""
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Add frame to queue (remove old frames if queue is full)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()  # Remove oldest frame
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
            time.sleep(0.03)  # ~30 FPS
    
    def get_latest_frame(self):
        """Get the latest captured frame"""
        if not self.frame_queue.empty():
            try:
                return self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        return None
    
    def is_running(self):
        """Check if camera is running"""
        return self.running and self.cap and self.cap.isOpened()

class MotionDetector:
    """Motion detection with Streamlit integration"""
    
    def __init__(self, config):
        self.config = config
        self.previous_frame = None
        self.last_detection_time = 0
        
    def detect_motion(self, frame):
        """Detect motion in frame"""
        if frame is None:
            return False, [], frame
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.config['motion_detection']['blur_size'], 0)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return False, [], frame
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_diff, self.config['motion_detection']['threshold'], 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        significant_contours = [
            cnt for cnt in contours 
            if cv2.contourArea(cnt) > self.config['motion_detection']['min_area']
        ]
        
        # Check cooldown
        current_time = time.time()
        motion_detected = (
            len(significant_contours) > 0 and 
            (current_time - self.last_detection_time) > self.config['motion_detection']['cooldown']
        )
        
        if motion_detected:
            self.last_detection_time = current_time
        
        # Draw visualization
        annotated_frame = self.draw_visualization(frame, significant_contours)
        
        # Update previous frame
        self.previous_frame = gray
        
        return motion_detected, significant_contours, annotated_frame
    
    def draw_visualization(self, frame, contours):
        """Draw motion detection visualization"""
        annotated = frame.copy()
        
        if self.config['visualization']['draw_contours']:
            cv2.drawContours(
                annotated, 
                contours, 
                -1, 
                self.config['visualization']['contour_color'], 
                self.config['visualization']['contour_thickness']
            )
        
        if self.config['visualization']['draw_timestamp']:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                annotated, 
                timestamp, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                self.config['visualization']['timestamp_color'], 
                self.config['visualization']['contour_thickness']
            )
        
        return annotated

class EventLogger:
    """Event logging for Streamlit session"""
    
    @staticmethod
    def log_motion_event(description, image_path=None, alert_sent=False):
        """Log a motion detection event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': 'motion',
            'description': description,
            'alert_sent': alert_sent
        }
        
        if image_path:
            event['image_path'] = image_path
        
        if 'events_log' not in st.session_state:
            st.session_state.events_log = []
        
        st.session_state.events_log.append(event)
        
        # Keep only last 1000 events
        if len(st.session_state.events_log) > 1000:
            st.session_state.events_log = st.session_state.events_log[-1000:]
    
    @staticmethod
    def log_counting_event(item_name, count, image_path=None):
        """Log an item counting event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': 'item_count',
            'item': item_name,
            'count': count,
            'description': f"Counted {count} {item_name}"
        }
        
        if image_path:
            event['image_path'] = image_path
        
        if 'events_log' not in st.session_state:
            st.session_state.events_log = []
        
        st.session_state.events_log.append(event)

def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string"""
    if frame is None:
        return None
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

def save_frame(frame, filename):
    """Save frame to file"""
    try:
        cv2.imwrite(filename, frame)
        return filename
    except Exception as e:
        st.error(f"Failed to save frame: {str(e)}")
        return None

def create_status_indicator(status, label):
    """Create a colored status indicator"""
    colors = {
        'active': '#28a745',
        'inactive': '#6c757d', 
        'warning': '#ffc107',
        'error': '#dc3545'
    }
    
    color = colors.get(status.lower(), '#6c757d')
    
    return f"""
    <div style="
        display: inline-block;
        padding: 4px 8px;
        background-color: {color};
        color: white;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        margin: 2px;
    ">
        {label}
    </div>
    """

def format_timestamp(timestamp_str):
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str