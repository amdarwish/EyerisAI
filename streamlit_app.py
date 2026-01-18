import streamlit as st
import os
import json
import base64
from datetime import datetime
import pytz
from PIL import Image
import io
import configparser
from pathlib import Path
import tempfile
from typing import List, Optional
import cv2
import numpy as np

# Import your agent functions
try:
    from my_agent import run_motion_agent, run_qa_agent, CONFIG, load_config_local
except ImportError as e:
    st.error(f"Error importing agent functions: {e}")
    st.stop()

def update_config_with_overrides(override_config):
    """Update the global CONFIG with provided overrides"""
    global CONFIG
    if override_config:
        # Create a new config with overrides applied
        new_config = CONFIG.copy()
        
        # Apply AI configuration overrides
        if 'ai' in override_config:
            new_config['ai'].update(override_config['ai'])
        
        # Apply other section overrides as needed
        for section, values in override_config.items():
            if section != 'ai' and isinstance(values, dict):
                new_config.setdefault(section, {}).update(values)
        
        CONFIG = new_config
        print(f"[CONFIG] Updated with overrides: {override_config}")



def apply_config_to_agents(effective_config, base_config):
    """Apply effective config to agents if there are any differences"""
    if effective_config != base_config:
        # Extract only the differences for the override
        override_data = {}
        
        # Check AI section differences
        if effective_config.get('ai') != base_config.get('ai'):
            ai_overrides = {}
            for key in ['model', 'base_url', 'api_key']:
                if effective_config.get('ai', {}).get(key) != base_config.get('ai', {}).get(key):
                    ai_overrides[key] = effective_config.get('ai', {}).get(key)
            if ai_overrides:
                override_data['ai'] = ai_overrides
        
        if override_data:
            update_config_with_overrides(override_data)

# Set page configuration
st.set_page_config(
    page_title="EyerisAI Workflow Agent",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        color: #F24236;
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #F24236;
        padding-bottom: 0.5rem;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        margin-bottom: 1rem;
        border-radius: 0.375rem;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        margin-bottom: 1rem;
        border-radius: 0.375rem;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        margin-bottom: 1rem;
        border-radius: 0.375rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

def load_config():
    """Load and return configuration"""
    try:
        config = load_config_local()
        # Validate required configuration sections
        if not config.get('ai', {}).get('base_url'):
            st.warning("AI base_url not configured in config.ini")
        if not config.get('ai', {}).get('model'):
            st.warning("AI model not configured in config.ini")
        if config.get('email', {}).get('enabled') and not config.get('email', {}).get('smtp_server'):
            st.warning("Email enabled but SMTP server not configured")
        return config
    except FileNotFoundError:
        st.error("config.ini file not found. Please ensure it exists in the project directory.")
        return None
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None

def test_ai_connection(config):
    """Test AI endpoint connectivity"""
    import requests
    try:
        base_url = config.get('ai', {}).get('base_url')
        if not base_url:
            return False, "No base URL configured"
            
        # Simple health check to models endpoint
        health_url = f"{base_url.rstrip('/')}/v1/models"
        headers = {'Content-Type': 'application/json'}
        api_key = config.get('ai', {}).get('api_key')
        if api_key:
            headers['Authorization'] = f"Bearer {api_key}"
            
        response = requests.get(health_url, headers=headers, timeout=10)
        if response.status_code == 200:
            return True, "Connected successfully"
        else:
            return False, f"Error {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection failed: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def get_effective_config(base_config):
    """Get configuration with session state overrides applied"""
    if not base_config:
        return base_config
    
    # Start with base config
    effective_config = base_config.copy()
    
    # Apply any session state overrides
    if hasattr(st, 'session_state') and 'config_override' in st.session_state:
        overrides = st.session_state.config_override
        if overrides.get('model'):
            effective_config.setdefault('ai', {})['model'] = overrides['model']
        if overrides.get('base_url'):
            effective_config.setdefault('ai', {})['base_url'] = overrides['base_url']
    
    return effective_config

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary directory and return path"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def extract_video_frames(video_path: str, max_frames: int = 6, sample_rate: int = 30) -> List[bytes]:
    """Extract frames from video file and convert to bytes list"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error opening video file")
            return []
        
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Calculate sampling interval to get desired number of frames
        if total_frames <= max_frames:
            sample_interval = 1
        else:
            sample_interval = total_frames // max_frames
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_interval == 0:
                # Convert frame to bytes
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                frames.append(frame_bytes)
            
            frame_count += 1
        
        cap.release()
        st.success(f"Extracted {len(frames)} frames from video (Total frames: {total_frames}, FPS: {fps})")
        return frames
    
    except Exception as e:
        st.error(f"Error extracting video frames: {e}")
        return []

def convert_image_to_bytes(image_path: str) -> List[bytes]:
    """Convert image to bytes list (fallback for single images)"""
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return [image_bytes]
    except Exception as e:
        st.error(f"Error converting image: {e}")
        return []

def display_results(result: bool, agent_type: str, description: str):
    """Display agent results in a formatted way"""
    if result:
        st.markdown(
            f'<div class="status-success">✅ <strong>{agent_type} Agent Result:</strong> Alert triggered! Email would be sent.<br><strong>Description:</strong> {description}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="status-warning">⚠️ <strong>{agent_type} Agent Result:</strong> No alert triggered.<br><strong>Description:</strong> {description}</div>',
            unsafe_allow_html=True
        )

def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ EyerisAI Workflow Agent</h1>', unsafe_allow_html=True)
    
    # Load configuration
    base_config = load_config()
    if not base_config:
        st.error("Failed to load configuration. Please check your config.ini file.")
        return
    
    # Get effective config (with any overrides)
    config = get_effective_config(base_config)
    
    # Sidebar for configuration and settings
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Show if any overrides are active
        if hasattr(st, 'session_state') and st.session_state.get('config_override'):
            overrides = st.session_state.config_override
            active_overrides = [k for k, v in overrides.items() if v]
            if active_overrides:
                st.info(f"🔧 Active overrides: {', '.join(active_overrides)}")
        
        # Display current configuration
        st.markdown("### Current Settings")
        st.markdown(f"**Instance Name:** {config.get('instance_name', 'Not set')}")
        st.markdown(f"**AI Model:** {config.get('ai', {}).get('model', 'Not set')}")
        st.markdown(f"**Agent Model:** {config.get('ai', {}).get('agent_model', 'Same as AI Model')}")
        st.markdown(f"**Base URL:** {config.get('ai', {}).get('base_url', 'Not set')}")
        st.markdown(f"**Email Enabled:** {'✅' if config.get('email', {}).get('enabled') else '❌'}")
        
        # Show API key status (masked)
        api_key = config.get('ai', {}).get('api_key')
        if api_key:
            masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
            st.markdown(f"**API Key:** {masked_key} ✅")
        else:
            st.markdown("**API Key:** Not set ❌")
        
        # Email configuration section
        st.markdown("### 📧 Email Settings")
        if config.get('email', {}).get('enabled'):
            st.success("Email alerts are enabled")
            st.markdown(f"**SMTP Server:** {config.get('email', {}).get('smtp_server', 'Not set')}")
            st.markdown(f"**Port:** {config.get('email', {}).get('smtp_port', 'Not set')}")
            st.markdown(f"**From:** {config.get('email', {}).get('from_address', 'Not set')}")
            st.markdown(f"**To:** {config.get('email', {}).get('to_address', 'Not set')}")
            st.markdown(f"**TLS:** {'✅' if config.get('email', {}).get('use_tls') else '❌'}")
        else:
            st.warning("Email alerts are disabled")
        
        # Configuration file info
        st.markdown("### 📄 Configuration File")
        st.markdown(f"**Location:** `config.ini`")
        st.markdown(f"**Last loaded:** {datetime.now().strftime('%H:%M:%S')}")
        
        # Quick actions
        st.markdown("### 🔧 Quick Actions")
        if st.button("🔄 Reload Configuration"):
            # Clear any cached config and reload
            st.rerun()
            
        if st.button("🔌 Test AI Connection"):
            with st.spinner("Testing AI connection..."):
                connected, message = test_ai_connection(config)
                if connected:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
        
        # Configuration editor
        with st.expander("⚙️ Quick Config Editor"):
            st.markdown("**Note:** Changes here are temporary. Edit config.ini for permanent changes.")
            
            # Basic config overrides for session
            if 'config_override' not in st.session_state:
                st.session_state.config_override = {}
            
            temp_model = st.text_input(
                "AI Model Override:", 
                value=st.session_state.config_override.get('model', ''),
                placeholder=config.get('ai', {}).get('model', 'qwen3-vl:32b')
            )
            
            temp_base_url = st.text_input(
                "Base URL Override:", 
                value=st.session_state.config_override.get('base_url', ''),
                placeholder=config.get('ai', {}).get('base_url', 'http://localhost:11434')
            )
            
            if st.button("Apply Temporary Override"):
                st.session_state.config_override = {
                    'model': temp_model,
                    'base_url': temp_base_url
                }
                st.success("Temporary configuration applied!")
                st.rerun()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Motion Detection", "🔍 Quality Assurance", "📊 Batch Processing"])
    
    with tab1:
        st.markdown('<h2 class="section-header">Motion Detection Agent</h2>', unsafe_allow_html=True)
        st.markdown("Upload an image to test the motion detection agent for human/person detection.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # File upload (prioritize video)
            uploaded_file = st.file_uploader(
                "Choose a video or image file",
                type=['mp4', 'avi', 'mov', 'mkv', 'png', 'jpg', 'jpeg'],
                key="motion_upload",
                help="Upload video files for frame analysis or images for single-frame detection"
            )
            
            # Video processing settings
            if uploaded_file and uploaded_file.type.startswith('video/'):
                st.markdown("**Video Processing Settings**")
                col_a, col_b = st.columns(2)
                with col_a:
                    max_frames = st.slider("Max frames to extract", 1, 10, 6)
                with col_b:
                    sample_rate = st.slider("Frame sampling rate", 1, 60, 30)
            
            # Manual description input
            manual_description = st.text_area(
                "Or enter a description manually:",
                placeholder="Example: Person walking in the hallway",
                help="You can either upload an image or provide a text description"
            )
            
            # Test button
            if st.button("🔍 Run Motion Detection", type="primary"):
                timestamp = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
                
                if uploaded_file is not None:
                    # Process uploaded file
                    file_path = save_uploaded_file(uploaded_file)
                    
                    if uploaded_file.type.startswith('video/'):
                        # Process video file
                        frames_bytes = extract_video_frames(file_path, max_frames, sample_rate)
                        description = f"Video analysis: {uploaded_file.name} ({len(frames_bytes)} frames extracted)"
                    else:
                        # Process image file
                        frames_bytes = convert_image_to_bytes(file_path)
                        description = f"Image analysis: {uploaded_file.name}"
                    
                    if frames_bytes:
                        with st.spinner("Running motion detection agent..."):
                            try:
                                # Apply effective config to agents
                                apply_config_to_agents(config, base_config)
                                
                                result = run_motion_agent(
                                    description=description,
                                    frames_bytes=frames_bytes,
                                    image_path=file_path,
                                    timestamp=timestamp
                                )
                                display_results(result, "Motion Detection", description)
                            except Exception as e:
                                st.error(f"Error running motion detection agent: {e}")
                    else:
                        st.error("Failed to extract frames from the uploaded file.")
                
                elif manual_description.strip():
                    # Process manual description
                    with st.spinner("Running motion detection agent..."):
                        try:
                            # Apply effective config to agents
                            apply_config_to_agents(config, base_config)
                            
                            result = run_motion_agent(
                                description=manual_description,
                                frames_bytes=[],
                                image_path="",
                                timestamp=timestamp
                            )
                            display_results(result, "Motion Detection", manual_description)
                        except Exception as e:
                            st.error(f"Error running motion detection agent: {e}")
                else:
                    st.warning("Please upload an image or provide a description.")
        
        with col2:
            if uploaded_file is not None:
                st.markdown("### Preview")
                if uploaded_file.type.startswith('video/'):
                    st.video(uploaded_file, start_time=0)
                    st.caption(f"Video: {uploaded_file.name}")
                    
                    # Show extracted frames if video is processed
                    if st.button("Preview Extracted Frames"):
                        temp_path = save_uploaded_file(uploaded_file)
                        preview_frames = extract_video_frames(temp_path, max_frames=3)
                        if preview_frames:
                            st.markdown("**Sample Extracted Frames:**")
                            cols = st.columns(len(preview_frames))
                            for i, frame_bytes in enumerate(preview_frames):
                                with cols[i]:
                                    st.image(frame_bytes, caption=f"Frame {i+1}", use_column_width=True)
                else:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Image: {uploaded_file.name}", use_column_width=True)
            else:
                st.markdown("### Instructions")
                st.info("""
                **Motion Detection Agent** analyzes videos, images or descriptions to detect human presence.
                
                **Video Processing (Primary):**
                - Upload video files (MP4, AVI, MOV, MKV)
                - Automatically extracts frames for analysis
                - Adjustable frame sampling settings
                
                **Image Processing (Fallback):**
                - Upload image files (PNG, JPG, JPEG)
                - Single frame analysis
                
                **Manual Input:**
                - Provide text descriptions for testing
                
                The agent analyzes all frames/content to determine if humans are detected and triggers email alerts accordingly.
                """)
    
    with tab2:
        st.markdown('<h2 class="section-header">Quality Assurance Agent</h2>', unsafe_allow_html=True)
        st.markdown("Test the QA agent to analyze frame descriptions for quality issues.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Frame descriptions input
            frame_descriptions = st.text_area(
                "Frame Descriptions (JSON format):",
                value='{\n  "frame_1": "Normal production line operation",\n  "frame_2": "Products piling up high near conveyor",\n  "frame_3": "Potential issue detected - items can fall"\n}',
                height=200,
                help="Provide frame descriptions in JSON format where keys are frame numbers and values are descriptions"
            )
            
            # Test button
            if st.button("🔍 Run QA Analysis", type="primary"):
                timestamp = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
                
                try:
                    # Parse JSON
                    frame_data = json.loads(frame_descriptions)
                    description = json.dumps(frame_data)
                    
                    with st.spinner("Running QA agent..."):
                        try:
                            # Apply effective config to agents
                            apply_config_to_agents(config, base_config)
                            
                            result = run_qa_agent(
                                description=description,
                                frames_bytes=[],
                                image_path="",
                                timestamp=timestamp
                            )
                            display_results(result, "Quality Assurance", "Frame analysis completed")
                        except Exception as e:
                            st.error(f"Error running QA agent: {e}")
                
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON format: {e}")
        
        with col2:
            st.markdown("### Instructions")
            st.info("""
            **Quality Assurance Agent** analyzes frame descriptions to detect quality issues.
            
            - Provide frame descriptions in JSON format
            - The agent looks for keywords indicating problems
            - Keywords include: "issue", "problem", "piling up high", "can fall", etc.
            - If issues are detected, an alert would be triggered
            """)
            
            st.markdown("### Example Issues")
            st.warning("""
            The QA agent will detect these types of issues:
            - Products piling up dangerously
            - Items that can fall
            - Production line problems
            - Safety concerns
            - Quality defects
            """)
    
    with tab3:
        st.markdown('<h2 class="section-header">Batch Processing</h2>', unsafe_allow_html=True)
        st.markdown("Process multiple images or descriptions at once.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Multiple file upload
            uploaded_files = st.file_uploader(
                "Choose multiple video or image files",
                type=['mp4', 'avi', 'mov', 'mkv', 'png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="batch_upload",
                help="Upload videos for multi-frame analysis or images for single-frame detection"
            )
            
            # Batch descriptions
            batch_descriptions = st.text_area(
                "Or enter multiple descriptions (one per line):",
                placeholder="Person in hallway\nEmpty room\nCrowd gathering",
                height=150
            )
            
            # Agent selection
            agent_type = st.selectbox(
                "Select Agent Type:",
                ["Motion Detection", "Quality Assurance"]
            )
            
            # Batch process button
            if st.button("🚀 Run Batch Processing", type="primary"):
                timestamp = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
                results = []
                
                if uploaded_files:
                    # Process uploaded files
                    progress_bar = st.progress(0)
                    
                    # Apply effective config to agents once for batch processing
                    apply_config_to_agents(config, base_config)
                    
                    for i, file in enumerate(uploaded_files):
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        file_path = save_uploaded_file(file)
                        
                        if file.type.startswith('video/'):
                            frames_bytes = extract_video_frames(file_path, max_frames=6)
                            description = f"Video: {file.name} ({len(frames_bytes)} frames)"
                        else:
                            frames_bytes = convert_image_to_bytes(file_path)
                            description = f"Image: {file.name}"
                        
                        try:
                            if agent_type == "Motion Detection":
                                result = run_motion_agent(description, frames_bytes, file_path, timestamp)
                            else:
                                result = run_qa_agent(description, frames_bytes, file_path, timestamp)
                            
                            results.append({
                                "file": file.name,
                                "description": description,
                                "result": result,
                                "agent": agent_type
                            })
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {e}")
                
                elif batch_descriptions.strip():
                    # Process descriptions
                    descriptions = [desc.strip() for desc in batch_descriptions.split('\n') if desc.strip()]
                    progress_bar = st.progress(0)
                    
                    # Apply effective config to agents once for batch processing
                    apply_config_to_agents(config, base_config)
                    
                    for i, desc in enumerate(descriptions):
                        progress_bar.progress((i + 1) / len(descriptions))
                        
                        try:
                            if agent_type == "Motion Detection":
                                result = run_motion_agent(desc, [], "", timestamp)
                            else:
                                result = run_qa_agent(desc, [], "", timestamp)
                            
                            results.append({
                                "file": f"Description {i+1}",
                                "description": desc,
                                "result": result,
                                "agent": agent_type
                            })
                        except Exception as e:
                            st.error(f"Error processing description {i+1}: {e}")
                
                # Display results
                if results:
                    st.markdown("### Batch Results")
                    alerts_triggered = sum(1 for r in results if r["result"])
                    
                    # Summary metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Processed", len(results))
                    with col_b:
                        st.metric("Alerts Triggered", alerts_triggered)
                    with col_c:
                        st.metric("Success Rate", f"{(alerts_triggered/len(results)*100):.1f}%")
                    
                    # Detailed results
                    for result in results:
                        status = "🔥 Alert" if result["result"] else "✅ Normal"
                        st.markdown(f"**{result['file']}**: {status} - {result['description'][:100]}{'...' if len(result['description']) > 100 else ''}")
                
                else:
                    st.warning("Please upload files or provide descriptions.")
        
        with col2:
            if uploaded_files:
                st.markdown("### Preview")
                for file in uploaded_files[:3]:  # Show first 3 files
                    if file.type.startswith('video/'):
                        st.video(file, start_time=0)
                        st.caption(f"Video: {file.name}")
                    else:
                        image = Image.open(file)
                        st.image(image, caption=f"Image: {file.name}", use_column_width=True)
                if len(uploaded_files) > 3:
                    st.markdown(f"... and {len(uploaded_files) - 3} more files")
            else:
                st.markdown("### Batch Processing")
                st.info("""
                **Batch Processing** allows you to:
                
                - Upload multiple videos and images at once
                - Automatic frame extraction from videos
                - Process multiple descriptions simultaneously
                - Choose between Motion Detection and QA agents
                - Get summary statistics and detailed results
                - Efficient processing of large datasets
                - Mixed video/image processing support
                """)

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; padding: 1rem;">🤖 EyerisAI Workflow Agent - Powered by LangGraph & Streamlit</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
