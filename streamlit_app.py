import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import time
import threading
from datetime import datetime
from pathlib import Path
import base64
from PIL import Image
import io
import configparser
import plotly.express as px
import plotly.graph_objects as go
import os
import requests

# Import utility modules first
try:
    from streamlit_utils import (
        CameraManager, MotionDetector, EventLogger,
        frame_to_base64, save_frame, create_status_indicator, format_timestamp
    )
except ImportError:
    st.error("streamlit_utils.py not found. Please ensure all files are in the same directory.")
    st.stop()

# Safe import of EyerisAI core functions
def safe_import_eyeris_functions():
    try:
        from EyerisAI import describe_image, describe_frames, count_items, detect_motion, load_config
        return describe_image, describe_frames, count_items, detect_motion, load_config
    except Exception as e:
        st.error(f"Error importing EyerisAI functions: {str(e)}")
        return None, None, None, None, None

# Try to import agent functions
def safe_import_agents():
    try:
        from my_agent import run_motion_agent, run_qa_agent, send_email_alert_tool
        return run_motion_agent, run_qa_agent, send_email_alert_tool
    except Exception as e:
        st.error(f"Error importing agent functions: {str(e)}")
        # Return dummy functions
        def dummy_agent(*args, **kwargs):
            return False
        return dummy_agent, dummy_agent, dummy_agent

# Initialize with safe imports
describe_image, describe_frames, count_items, detect_motion, load_config = safe_import_eyeris_functions()
run_motion_agent, run_qa_agent, send_email_alert_tool = safe_import_agents()

# Load config using EyerisAI's load_config function
try:
    CONFIG = load_config() if load_config else None
except Exception as e:
    st.error(f"Failed to load EyerisAI config: {str(e)}")
    CONFIG = None

# Helper functions for the UI
# describe_image_ui removed - using EyerisAI.describe_image instead

# detect_motion_ui removed - using EyerisAI.detect_motion instead

# All backend functions removed - using EyerisAI.py and my_agent.py directly
st.set_page_config(
    page_title="EyerisAI - Intelligent Surveillance System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class StreamlitEyerisAI:
    def __init__(self):
        self.config = CONFIG
        # Only initialize camera manager if streamlit_utils is available
        try:
            self.camera_manager = CameraManager()
            self.motion_detector = MotionDetector(self.config)
        except:
            self.camera_manager = None
            self.motion_detector = None
        
        # Initialize session state
        if 'motion_running' not in st.session_state:
            st.session_state.motion_running = False
        if 'events_log' not in st.session_state:
            st.session_state.events_log = []
        if 'config_updated' not in st.session_state:
            st.session_state.config_updated = False
        if 'camera_source' not in st.session_state:
            st.session_state.camera_source = None

    def main(self):
        """Main application interface"""
        st.markdown('<h1 class="main-header">👁️ EyerisAI Control Center</h1>', unsafe_allow_html=True)
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigate to:",
            ["🏠 Dashboard", "🎥 Motion Detection", "🔍 Item Counter", "📹 Video Analysis", "⚙️ Configuration", "📊 Analytics", "📋 Event Logs"]
        )
        
        # Route to appropriate page
        if page == "🏠 Dashboard":
            self.dashboard_page()
        elif page == "🎥 Motion Detection":
            self.motion_detection_page()
        elif page == "🔍 Item Counter":
            self.item_counter_page()
        elif page == "📹 Video Analysis":
            self.video_analysis_page()
        elif page == "⚙️ Configuration":
            self.configuration_page()
        elif page == "📊 Analytics":
            self.analytics_page()
        elif page == "📋 Event Logs":
            self.event_logs_page()

    def dashboard_page(self):
        """Main dashboard with system overview"""
        st.header("System Overview")
        
        # System status metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Motion Detection",
                "Active" if st.session_state.motion_running else "Inactive",
                delta="Running" if st.session_state.motion_running else "Stopped"
            )
        
        with col2:
            model_display = f"{self.config['ai']['model']}"
            if self.config['ai'].get('agent_model') != self.config['ai']['model']:
                model_display += f" / {self.config['ai']['agent_model']}"
            st.metric("AI Model", model_display)
        
        with col3:
            st.metric("Events Today", len(st.session_state.events_log))
        
        with col4:
            st.metric(
                "Email Alerts", 
                "Enabled" if self.config['email']['enabled'] else "Disabled"
            )
        
        # Quick actions
        st.subheader("Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🎥 Start Motion Detection", use_container_width=True):
                st.switch_page("Motion Detection")
        
        with col2:
            if st.button("🔍 Count Items", use_container_width=True):
                st.switch_page("Item Counter")
                
        with col3:
            if st.button("📹 Analyze Video", use_container_width=True):
                st.switch_page("Video Analysis")
                
        with col4:
            if st.button("⚙️ Configure System", use_container_width=True):
                st.switch_page("Configuration")
        
        # Recent events
        if st.session_state.events_log:
            st.subheader("Recent Events")
            recent_events = st.session_state.events_log[-5:]
            for event in reversed(recent_events):
                with st.expander(f"Event at {event.get('timestamp', 'Unknown')}"):
                    st.write(f"**Description:** {event.get('description', 'No description')}")
                    st.write(f"**Type:** {event.get('type', 'Motion Detection')}")
                    if 'image_path' in event:
                        try:
                            img = Image.open(event['image_path'])
                            st.image(img, caption="Captured Frame", width=300)
                        except:
                            st.write("Image not available")

    def motion_detection_page(self):
        """Motion detection interface with live monitoring"""
        st.header("🎥 Motion Detection System")
        
        # Control panel
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Camera Controls")
            
            # Camera source selection
            camera_source = st.radio(
                "Select Camera Source:",
                ["Local Camera", "IP Camera"],
                horizontal=True
            )
            
            if camera_source == "IP Camera":
                ip_url = st.text_input("IP Camera URL:", value=self.config['camera'].get('ip_url', ''))
                source = ip_url
            else:
                device_id = st.number_input("Device ID:", min_value=0, max_value=10, value=0)
                source = device_id
                
            st.session_state.camera_source = source
        
        with col2:
            st.subheader("Detection Settings")
            min_area = st.slider("Minimum Motion Area:", 100, 2000, self.config['motion_detection']['min_area'])
            threshold = st.slider("Motion Threshold:", 10, 100, self.config['motion_detection']['threshold'])
            cooldown = st.slider("Cooldown (seconds):", 10, 300, self.config['motion_detection']['cooldown'])
            
            # Update config with new settings
            self.config['motion_detection']['min_area'] = min_area
            self.config['motion_detection']['threshold'] = threshold
            self.config['motion_detection']['cooldown'] = cooldown
        
        # Start/Stop controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Start Detection", use_container_width=True, type="primary"):
                if not st.session_state.motion_running:
                    if self.camera_manager:
                        success = self.camera_manager.start_camera(source)
                        if success:
                            st.session_state.motion_running = True
                            st.success("Motion detection started!")
                            st.rerun()
                        else:
                            st.error("Failed to start camera!")
                    else:
                        st.error("Camera utilities not available!")
        
        with col2:
            if st.button("⏹️ Stop Detection", use_container_width=True):
                if st.session_state.motion_running:
                    if self.camera_manager:
                        self.camera_manager.stop_camera()
                    st.session_state.motion_running = False
                    st.success("Motion detection stopped!")
                    st.rerun()
        
        with col3:
            if st.button("📸 Test Camera", use_container_width=True):
                self.test_camera_connection()
        
        # Status indicators
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            camera_status = "Active" if (self.camera_manager and self.camera_manager.is_running()) else "Inactive"
            st.markdown(create_status_indicator(camera_status.lower(), f"Camera: {camera_status}"), unsafe_allow_html=True)
        
        with status_col2:
            detection_status = "Running" if st.session_state.motion_running else "Stopped"
            st.markdown(create_status_indicator(detection_status.lower(), f"Detection: {detection_status}"), unsafe_allow_html=True)
        
        with status_col3:
            ai_status = "Connected" if self.config['ai']['base_url'] else "Not Configured"
            st.markdown(create_status_indicator("active" if ai_status == "Connected" else "warning", f"AI: {ai_status}"), unsafe_allow_html=True)
        
        # Live feed display with motion detection
        if st.session_state.motion_running:
            if self.camera_manager and self.camera_manager.is_running():
                st.subheader("Live Camera Feed with Motion Detection")
                
                # Create placeholders for live updates
                frame_placeholder = st.empty()
                info_placeholder = st.empty()
                
                # Get latest frame and process it
                frame = self.camera_manager.get_latest_frame()
                
                if frame is not None:
                    # Perform motion detection using EyerisAI function
                    if hasattr(self, 'previous_frame') and self.previous_frame is not None and detect_motion:
                        motion_detected, contours = detect_motion(self.previous_frame, frame)
                    else:
                        motion_detected, contours = False, []
                        
                    self.previous_frame = frame.copy()
                    
                    # Draw visualization
                    annotated_frame = frame.copy()
                    if motion_detected and self.config['visualization']['draw_contours']:
                        cv2.drawContours(
                            annotated_frame, 
                            contours, 
                            -1, 
                            self.config['visualization']['contour_color'], 
                            self.config['visualization']['contour_thickness']
                        )
                    
                    # Add timestamp
                    if self.config['visualization']['draw_timestamp']:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(
                            annotated_frame, 
                            timestamp, 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            self.config['visualization']['timestamp_color'], 
                            2
                        )
                    
                    # Convert frame for display
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(
                        frame_rgb, 
                        caption=f"Live Feed - {'🔴 Motion Detected!' if motion_detected else '🟢 Monitoring'}", 
                        use_column_width=True
                    )
                    
                    # Show motion info
                    if motion_detected:
                        with info_placeholder.container():
                            st.warning(f"🚨 Motion detected! {len(contours)} moving objects found.")
                            
                            # Process with LangGraph motion agent if available
                            with st.spinner("Analyzing with LangGraph Agent..."):
                                try:
                                    # Get frame data for agent
                                    _, img_bytes = cv2.imencode('.jpg', frame)
                                    
                                    # Create temporary image path
                                    temp_dir = Path("/tmp")
                                    temp_dir.mkdir(exist_ok=True)
                                    temp_path = temp_dir / f"motion_{int(time.time())}.jpg"
                                    cv2.imwrite(str(temp_path), frame)
                                    
                                    # Use LangGraph motion agent
                                    alert_sent = run_motion_agent(
                                        "Motion detected in surveillance camera", 
                                        [img_bytes.tobytes()], 
                                        str(temp_path), 
                                        datetime.now().isoformat()
                                    )
                                    
                                    st.info("Motion analysis completed by LangGraph agent")
                                    
                                    if alert_sent:
                                        st.success("📧 Alert email sent by agent!")
                                    else:
                                        st.info("Agent determined no alert needed")
                                    
                                    # Log event
                                    EventLogger.log_motion_event("Motion detected and processed by LangGraph agent")
                                    
                                except Exception as e:
                                    st.error(f"LangGraph agent analysis failed: {str(e)}")
                    else:
                        info_placeholder.empty()
                else:
                    frame_placeholder.info("Waiting for camera feed...")
                
                # Auto-refresh every 100ms for smooth video
                time.sleep(0.1)
                st.rerun()
            else:
                st.warning("Camera utilities not available or camera not running")
        elif st.session_state.motion_running:
            st.warning("Motion detection is enabled but camera is not running. Please check camera connection.")
        
        # Motion detection statistics
        motion_events = [e for e in st.session_state.events_log if e.get('type') == 'motion']
        if motion_events:
            st.subheader("Detection Statistics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Detections", len(motion_events))
            with col2:
                today_events = [e for e in motion_events if 'timestamp' in e and 
                              datetime.fromisoformat(e['timestamp']).date() == datetime.now().date()]
                st.metric("Events Today", len(today_events))
            with col3:
                alerts_sent = [e for e in motion_events if e.get('alert_sent', False)]
                st.metric("Alerts Sent", len(alerts_sent))

    def item_counter_page(self):
        """Item counting interface"""
        st.header("🔍 Intelligent Item Counter")
        
        # File upload section
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to count specific items"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.subheader("Counting Parameters")
                
                # Item to count
                item_name = st.text_input(
                    "What would you like to count?",
                    placeholder="e.g., people, cars, bottles, etc."
                )
                
                # Custom prompt (optional)
                custom_prompt = st.text_area(
                    "Custom Instructions (Optional):",
                    placeholder="Additional instructions for the AI..."
                )
                
                # Count button
                if st.button("🔢 Start Counting", use_container_width=True, type="primary"):
                    if item_name:
                        with st.spinner(f"Counting {item_name} in the image..."):
                            try:
                                # Save uploaded image temporarily
                                temp_path = f"/tmp/{uploaded_file.name}"
                                image.save(temp_path)
                                
                                # Count items using EyerisAI function directly
                                if count_items:
                                    # EyerisAI.count_items prints results to console
                                    count_items(temp_path, item_name)
                                    
                                    st.success(f"Item counting completed for '{item_name}'. Check console output for results.")
                                    
                                    # Log the event (generic since count_items doesn't return values)
                                    event = {
                                        'timestamp': datetime.now().isoformat(),
                                        'type': 'item_count',
                                        'item': item_name,
                                        'count': 'See console output',
                                        'image_path': temp_path,
                                        'description': f"Item counting performed for '{item_name}' using EyerisAI"
                                    }
                                    st.session_state.events_log.append(event)
                                    
                                else:
                                    st.error("EyerisAI count_items function not available.")
                                    
                            except Exception as e:
                                st.error(f"Error during counting: {str(e)}")
                    else:
                        st.warning("Please specify what you want to count!")
        
        # Recent counting results
        counting_events = [e for e in st.session_state.events_log if e.get('type') == 'item_count']
        if counting_events:
            st.subheader("Recent Counting Results")
            
            for event in reversed(counting_events[-5:]):
                with st.expander(f"{event.get('item', 'Unknown')} - {event.get('timestamp', 'Unknown time')}"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Item:** {event.get('item', 'Unknown')}")
                        st.write(f"**Count:** {event.get('count', 0)}")
                        st.write(f"**Description:** {event.get('description', 'No description')}")
                    with col2:
                        if 'image_path' in event:
                            try:
                                img = Image.open(event['image_path'])
                                st.image(img, width=200)
                            except:
                                st.write("Image not available")

    def configuration_page(self):
        """Configuration display interface - Read-only view of EyerisAI config"""
        st.header("⚙️ System Configuration")
        
        st.info("📋 **Read-Only Configuration Display** - To modify settings, edit the `config.ini` file directly and restart the application.")
        
        if CONFIG:
            # Display current configuration in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 AI Settings", "📷 Camera", "🚨 Motion Detection", "📧 Email", "🎨 Visualization"])
            
            with tab1:
                st.subheader("AI Model Configuration")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.text_input("API Base URL:", value=CONFIG['ai']['base_url'], disabled=True)
                    st.text_input("Model Name:", value=CONFIG['ai']['model'], disabled=True)
                    st.text_input("Agent Model:", value=CONFIG['ai'].get('agent_model', ''), disabled=True)
                
                with col2:
                    st.text_input("API Key:", value="***" if CONFIG['ai'].get('api_key') else "", disabled=True, type="password")
                    st.number_input("Max Tokens:", value=CONFIG['ai'].get('max_tokens', 300), disabled=True)
                
                st.text_area("Default Prompt:", value=CONFIG['ai'].get('prompt', ''), height=100, disabled=True)
                st.text_area("Motion Analysis Prompt:", value=CONFIG['ai'].get('motion_prompt', ''), height=100, disabled=True)
            
            with tab2:
                st.subheader("Camera Configuration")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.checkbox("Use IP Camera", value=CONFIG['camera'].get('use_ip', False), disabled=True)
                    if CONFIG['camera'].get('use_ip', False):
                        st.text_input("IP Camera URL:", value=CONFIG['camera'].get('ip_url', ''), disabled=True)
                    else:
                        st.number_input("Device ID:", value=CONFIG['camera'].get('device_id', 0), disabled=True)
                
                with col2:
                    st.number_input("Width:", value=CONFIG['camera'].get('width', 640), disabled=True)
                    st.number_input("Height:", value=CONFIG['camera'].get('height', 480), disabled=True)
                    st.slider("Auto Exposure:", 0.0, 1.0, CONFIG['camera'].get('auto_exposure', 0.75), disabled=True)
            
            with tab3:
                st.subheader("Motion Detection Settings")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.number_input("Minimum Area:", value=CONFIG['motion_detection'].get('min_area', 700), disabled=True)
                    st.number_input("Threshold:", value=CONFIG['motion_detection'].get('threshold', 50), disabled=True)
                    st.number_input("Cooldown (seconds):", value=CONFIG['motion_detection'].get('cooldown', 60), disabled=True)
                
                with col2:
                    blur_size = CONFIG['motion_detection'].get('blur_size', (21, 21))
                    st.number_input("Blur Size X:", value=blur_size[0], disabled=True)
                    st.number_input("Blur Size Y:", value=blur_size[1], disabled=True)
                    st.number_input("Frames to Capture:", value=CONFIG['motion_detection'].get('n_frames', 2), disabled=True)
            
            with tab4:
                st.subheader("Email Alert Configuration")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.checkbox("Enable Email Alerts", value=CONFIG['email'].get('enabled', False), disabled=True)
                    st.text_input("SMTP Server:", value=CONFIG['email'].get('smtp_server', ''), disabled=True)
                    st.number_input("SMTP Port:", value=CONFIG['email'].get('smtp_port', 587), disabled=True)
                    st.checkbox("Use TLS", value=CONFIG['email'].get('use_tls', True), disabled=True)
                
                with col2:
                    st.text_input("From Address:", value=CONFIG['email'].get('from_address', ''), disabled=True)
                    st.text_input("To Address:", value=CONFIG['email'].get('to_address', ''), disabled=True)
                    st.text_input("SMTP Username:", value=CONFIG['email'].get('smtp_username', ''), disabled=True)
                    st.text_input("SMTP Password:", value="***" if CONFIG['email'].get('smtp_password') else "", disabled=True, type="password")
            
            with tab5:
                st.subheader("Visualization Settings")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.checkbox("Draw Contours", value=CONFIG['visualization'].get('draw_contours', True), disabled=True)
                    st.checkbox("Draw Timestamp", value=CONFIG['visualization'].get('draw_timestamp', True), disabled=True)
                
                with col2:
                    contour_color = CONFIG['visualization'].get('contour_color', (232, 8, 255))
                    st.text_input("Contour Color (RGB):", value=f"{contour_color}", disabled=True)
                    timestamp_color = CONFIG['visualization'].get('timestamp_color', (0, 255, 0))
                    st.text_input("Timestamp Color (RGB):", value=f"{timestamp_color}", disabled=True)
                    st.slider("Contour Thickness:", 1, 10, CONFIG['visualization'].get('contour_thickness', 2), disabled=True)
            
            st.markdown("---")
            st.info("💡 **To modify these settings:** Edit the `config.ini` file in the EyerisAI directory and restart the application.")
            
        else:
            st.error("❌ Unable to load EyerisAI configuration. Please check that config.ini exists and is properly formatted.")

    def analytics_page(self):
        """Analytics and reporting interface"""
        st.header("📊 Analytics Dashboard")
        
        if not st.session_state.events_log:
            st.info("No events recorded yet. Start using the system to see analytics!")
            return
        
        # Create DataFrame from events
        df = pd.DataFrame(st.session_state.events_log)
        
        # Time-based analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Event Timeline")
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                hourly_counts = df.groupby('hour').size()
                
                fig_timeline = px.line(
                    x=hourly_counts.index, 
                    y=hourly_counts.values,
                    title="Events by Hour of Day",
                    labels={'x': 'Hour', 'y': 'Event Count'}
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col2:
            st.subheader("Event Types")
            if 'type' in df.columns:
                type_counts = df['type'].value_counts()
                
                fig_types = px.pie(
                    values=type_counts.values, 
                    names=type_counts.index,
                    title="Distribution of Event Types"
                )
                st.plotly_chart(fig_types, use_container_width=True)
        
        # Item counting analytics
        counting_events = df[df['type'] == 'item_count'] if 'type' in df.columns else pd.DataFrame()
        if not counting_events.empty:
            st.subheader("Item Counting Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                item_summary = counting_events.groupby('item')['count'].sum().sort_values(ascending=False)
                st.bar_chart(item_summary)
            
            with col2:
                st.write("**Top Counted Items:**")
                for item, total in item_summary.head(5).items():
                    st.write(f"• {item}: {total}")
        
        # Video analysis analytics
        video_events = df[df['type'] == 'video_analysis'] if 'type' in df.columns else pd.DataFrame()
        if not video_events.empty:
            st.subheader("Video Analysis Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Videos Analyzed", len(video_events))
            
            with col2:
                total_segments = video_events['segments_analyzed'].sum() if 'segments_analyzed' in video_events.columns else 0
                st.metric("Total Segments", total_segments)
            
            with col3:
                if 'analysis_type' in video_events.columns:
                    most_common = video_events['analysis_type'].mode().iloc[0] if not video_events['analysis_type'].mode().empty else "N/A"
                    st.metric("Most Used Analysis", most_common)
        
        # Motion detection analytics
        motion_events = df[df['type'] == 'motion'] if 'type' in df.columns else pd.DataFrame()
        if not motion_events.empty:
            st.subheader("Motion Detection Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Motion Events", len(motion_events))
            
            with col2:
                alerts_sent = motion_events.get('alert_sent', pd.Series()).sum()
                st.metric("Alerts Sent", alerts_sent if pd.notna(alerts_sent) else 0)
            
            with col3:
                if 'timestamp' in motion_events.columns:
                    today_events = motion_events[motion_events['timestamp'].dt.date == datetime.now().date()]
                    st.metric("Events Today", len(today_events))

    def video_analysis_page(self):
        """Video analysis interface for quality assurance and defect detection"""
        st.header("📹 Video Analysis & Quality Control")
        
        st.markdown("""
        Upload a video file to analyze it frame by frame. This is useful for:
        - Quality control in production lines
        - Defect detection in manufacturing
        - Safety monitoring
        - Process analysis
        """)
        
        # Video upload section
        st.subheader("Upload Video File")
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Upload a video file for frame-by-frame analysis"
        )
        
        if uploaded_video is not None:
            # Save uploaded video temporarily
            temp_video_path = f"/tmp/{uploaded_video.name}"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.read())
            
            # Video info and parameters
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display video info
                try:
                    cap = cv2.VideoCapture(temp_video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps if fps > 0 else 0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    
                    st.info(f"""
                    **Video Information:**
                    - Duration: {duration:.2f} seconds
                    - FPS: {fps:.1f}
                    - Resolution: {width}x{height}
                    - Total Frames: {total_frames}
                    """)
                    
                except Exception as e:
                    st.error(f"Could not read video file: {str(e)}")
                    return
            
            with col2:
                st.subheader("Analysis Parameters")
                
                # Number of frames to extract
                n_frames = st.slider(
                    "Frames to Extract per Segment:",
                    min_value=1,
                    max_value=10,
                    value=4,
                    help="Number of frames to extract from each time segment"
                )
                
                # Time interval
                interval_options = {
                    "Entire video": 0.0,
                    "Every 5 seconds": 5.0,
                    "Every 10 seconds": 10.0,
                    "Every 30 seconds": 30.0,
                    "Every minute": 60.0,
                    "Custom": -1
                }
                
                interval_choice = st.selectbox(
                    "Analysis Interval:",
                    list(interval_options.keys()),
                    help="How often to extract frames for analysis"
                )
                
                if interval_choice == "Custom":
                    interval_sec = st.number_input(
                        "Custom interval (seconds):",
                        min_value=0.1,
                        max_value=duration,
                        value=10.0
                    )
                else:
                    interval_sec = interval_options[interval_choice]
                
                # Analysis prompt
                st.subheader("Analysis Instructions")
                
                # Predefined prompts
                prompt_templates = {
                    "Quality Control": """
                    Analyze these video frames for quality control. Look for:
                    - Product defects or irregularities
                    - Items in wrong positions
                    - Process anomalies
                    Report findings in JSON format with frame numbers as keys.
                    """,
                    "Safety Monitoring": """
                    Analyze these frames for safety issues:
                    - Unsafe conditions
                    - Equipment malfunctions
                    - Personnel safety violations
                    Report any safety concerns in JSON format.
                    """,
                    "Production Line": """
                    Monitor this production line footage:
                    - Items moving correctly on conveyor
                    - No jams or blockages
                    - Proper item positioning
                    Report status in JSON format with frame analysis.
                    """,
                    "Custom": ""
                }
                
                prompt_choice = st.selectbox(
                    "Analysis Type:",
                    list(prompt_templates.keys())
                )
                
                # Always allow editing of the prompt, but pre-populate with template if selected
                if prompt_choice == "Custom":
                    initial_prompt = ""
                    placeholder_text = "Enter your specific analysis instructions here..."
                else:
                    initial_prompt = prompt_templates[prompt_choice]
                    placeholder_text = "You can edit this template or write your own instructions..."
                
                analysis_prompt = st.text_area(
                    "Analysis Instructions:",
                    value=initial_prompt,
                    height=100,
                    placeholder=placeholder_text,
                    help="Edit this prompt to customize how the AI analyzes your video frames"
                )
            
            # Analysis button
            if st.button("🎬 Start Video Analysis", use_container_width=True, type="primary"):
                if not analysis_prompt.strip():
                    st.warning("Please provide analysis instructions!")
                    return
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()
                
                try:
                    # Process video
                    results = self.process_video_analysis(
                        temp_video_path, 
                        n_frames, 
                        interval_sec, 
                        analysis_prompt,
                        progress_bar,
                        status_text
                    )
                    
                    # Display results
                    with results_container:
                        st.subheader("Analysis Results")
                        
                        if results:
                            for i, result in enumerate(results):
                                with st.expander(f"Segment {i+1}: {result.get('timerange', 'Unknown time')}"):
                                    col1, col2 = st.columns([2, 1])
                                    
                                    with col1:
                                        st.write("**AI Analysis:**")
                                        st.write(result.get('description', 'No analysis available'))
                                        
                                        # Show QA agent result if available
                                        if result.get('qa_result'):
                                            if result['qa_result']:
                                                st.error("⚠️ Issue detected by QA agent!")
                                            else:
                                                st.success("✅ No issues detected")
                                    
                                    with col2:
                                        # Display extracted frames
                                        if 'frames' in result:
                                            for j, frame_path in enumerate(result['frames']):
                                                try:
                                                    img = Image.open(frame_path)
                                                    st.image(img, caption=f"Frame {j+1}", width=200)
                                                except:
                                                    st.write(f"Frame {j+1}: Not available")
                            
                            # Log the video analysis event
                            event = {
                                'timestamp': datetime.now().isoformat(),
                                'type': 'video_analysis',
                                'video_file': uploaded_video.name,
                                'segments_analyzed': len(results),
                                'analysis_type': prompt_choice,
                                'description': f"Analyzed {len(results)} segments from {uploaded_video.name}"
                            }
                            st.session_state.events_log.append(event)
                        else:
                            st.warning("No results generated from video analysis")
                
                except Exception as e:
                    st.error(f"Video analysis failed: {str(e)}")
                finally:
                    progress_bar.empty()
                    status_text.empty()

    def process_video_analysis(self, video_path, n_frames, interval_sec, prompt, progress_bar, status_text):
        """Process video analysis similar to the command line version"""
        results = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Failed to open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            if fps <= 0 or total_frames <= 0:
                raise Exception("Invalid video file or unable to read FPS/frames")
            
            # If interval is 0.0, use entire video duration
            if interval_sec == 0.0:
                interval_sec = duration
            
            segment = 0
            start_time = 0.0
            total_segments = max(1, int(duration / interval_sec))
            
            while start_time < duration:
                # Update progress
                progress = min(segment / total_segments, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing segment {segment + 1}/{total_segments}...")
                
                # Calculate frame indices for this segment
                if n_frames > 1:
                    frame_indices = [
                        int((start_time + i * interval_sec / (n_frames - 1)) * fps) 
                        for i in range(n_frames)
                    ]
                    frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]
                else:
                    frame_indices = [int(start_time * fps)]
                
                # Extract frames
                frames_bytes = []
                frame_paths = []
                
                for frame_num, idx in enumerate(frame_indices):
                    if idx >= total_frames:
                        continue
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        # Encode frame
                        success, jpg = cv2.imencode('.jpg', frame)
                        if success:
                            frames_bytes.append(jpg.tobytes())
                            
                            # Save frame
                            save_dir = Path(CONFIG.get('save_directory', 'captures'))
                            save_dir.mkdir(exist_ok=True)
                            frame_filename = save_dir / f"video_segment{segment+1}_frame{frame_num+1}_idx{idx}.jpg"
                            cv2.imwrite(str(frame_filename), frame)
                            frame_paths.append(str(frame_filename))
                
                # Analyze frames if we have any
                if frames_bytes:
                    try:
                        # Get AI description using EyerisAI function
                        if describe_frames:
                            description = describe_frames(frames_bytes, prompt)
                        else:
                            description = "AI analysis not available"
                        
                        # Try QA analysis with LangGraph agent if available
                        qa_result = None
                        try:
                            qa_result = run_qa_agent(
                                description, 
                                frames_bytes, 
                                frame_paths[0] if frame_paths else video_path, 
                                f"segment_{segment+1}"
                            )
                            print(f"QA agent result for segment {segment+1}: {qa_result}")
                        except Exception as e:
                            print(f"QA agent error for segment {segment+1}: {str(e)}")
                            qa_result = None  # QA agent not available
                        
                        # Store results
                        timerange = f"{start_time:.2f}s - {min(start_time + interval_sec, duration):.2f}s"
                        results.append({
                            'segment': segment + 1,
                            'timerange': timerange,
                            'description': description,
                            'qa_result': qa_result,
                            'frames': frame_paths,
                            'frame_count': len(frames_bytes)
                        })
                        
                    except Exception as e:
                        st.warning(f"Failed to analyze segment {segment + 1}: {str(e)}")
                
                segment += 1
                start_time += interval_sec
            
            cap.release()
            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")
            
        except Exception as e:
            raise Exception(f"Video processing failed: {str(e)}")
        
        return results

    # describe_frames_ui removed - using EyerisAI.describe_frames instead

    def event_logs_page(self):
        """Event logs viewer"""
        st.header("📋 Event Logs")
        
        if not st.session_state.events_log:
            st.info("No events recorded yet.")
            return
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            event_types = list(set([e.get('type', 'unknown') for e in st.session_state.events_log]))
            selected_types = st.multiselect("Filter by Event Type:", event_types, default=event_types)
        
        with col2:
            date_filter = st.date_input("Filter by Date:", value=datetime.now().date())
        
        with col3:
            if st.button("🗑️ Clear All Logs"):
                st.session_state.events_log = []
                st.success("All logs cleared!")
                st.rerun()
        
        # Display filtered events
        filtered_events = []
        for event in st.session_state.events_log:
            if event.get('type') in selected_types:
                event_date = datetime.fromisoformat(event['timestamp']).date() if 'timestamp' in event else None
                if event_date == date_filter or date_filter is None:
                    filtered_events.append(event)
        
        if filtered_events:
            st.subheader(f"Showing {len(filtered_events)} events")
            
            for i, event in enumerate(reversed(filtered_events)):
                with st.expander(f"Event {len(filtered_events)-i}: {event.get('type', 'Unknown')} - {event.get('timestamp', 'Unknown time')}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.json(event)
                    
                    with col2:
                        if 'image_path' in event:
                            try:
                                img = Image.open(event['image_path'])
                                st.image(img, caption="Event Image", width=200)
                            except:
                                st.write("Image not available")
        
        else:
            st.info("No events match the selected filters.")

    def start_motion_detection_thread(self):
        """Start motion detection in a separate thread"""
        def run_detection():
            try:
                # This is a simplified version - in production you'd want proper thread management
                from EyerisAI import run_motion_detection
                run_motion_detection()
            except Exception as e:
                st.error(f"Motion detection error: {str(e)}")
        
        if not st.session_state.motion_running:
            return
            
        thread = threading.Thread(target=run_detection, daemon=True)
        thread.start()

    def test_camera_connection(self):
        """Test camera connection"""
        try:
            source = st.session_state.get('camera_source', 0)
            
            # Test connection
            if isinstance(source, str):  # IP camera
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            else:  # Local camera
                cap = cv2.VideoCapture(source)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    st.success("✅ Camera connection successful!")
                    
                    # Display test frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="Test Frame", width=400)
                    
                    # Show camera info
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    st.info(f"Camera Resolution: {int(width)}x{int(height)}, FPS: {fps:.1f}")
                else:
                    st.error("❌ Could not read frame from camera")
            else:
                st.error("❌ Could not connect to camera")
            
            cap.release()
            
        except Exception as e:
            st.error(f"❌ Camera test failed: {str(e)}")

    # save_configuration removed - config editing should be done directly in config.ini

# Run the application
if __name__ == "__main__":
    app = StreamlitEyerisAI()
    app.main()