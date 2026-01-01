
# EyerisAI.py   
# Standard library imports
import base64
import configparser
import json
import os
import smtplib
import time
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from my_agent import run_agent
# Third-party imports
import cv2
import numpy as np
import pyttsx3
import requests

def is_mostly_gray(frame, gray_threshold=0.5, tolerance=10):
    """
    Returns True if more than `gray_threshold` fraction of the frame's pixels are gray (R≈G≈B within `tolerance`).
    """
    if frame is None:
        return True
    arr = np.asarray(frame)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        diff_rg = np.abs(arr[:,:,0] - arr[:,:,1])
        diff_gb = np.abs(arr[:,:,1] - arr[:,:,2])
        gray_pixels = (diff_rg < tolerance) & (diff_gb < tolerance)
        gray_fraction = np.sum(gray_pixels) / (arr.shape[0] * arr.shape[1])
        return gray_fraction > gray_threshold
    return False

def load_config():
    """
    Load configuration from config.ini
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # Parse the contour color from string
    contour_color = tuple(map(int, config.get('Visualization', 'contour_color').split(',')))
    timestamp_color = tuple(map(int, config.get('Visualization', 'timestamp_color').split(',')))
    
    return {
        'save_directory': config.get('General', 'save_directory'),
        'log_file': config.get('General', 'log_file'),
        'ai_description': config.getboolean('General', 'ai_description'),
        'instance_name': config.get('General', 'instance_name', fallback='Motion Detector'),
        'ai': {
            'base_url': config.get('AI', 'base_url'),
            'model': config.get('AI', 'model'),
            'agent_model': config.get('AI', 'agent_model', fallback=None),
            'prompt': config.get('AI', 'prompt'),
            'api_key': config.get('AI', 'api_key', fallback=None),  # Optional for Ollama
            'max_tokens': config.getint('AI', 'max_tokens', fallback=300),
            # Optional motion prompt to describe motion across multiple frames
            'motion_prompt': config.get('AI', 'motion_prompt', fallback=config.get('AI', 'prompt'))
        },
        'camera': {
            # If use_ip is True, the IP camera URL in 'ip_url' will be used instead of the local device id
            'use_ip': config.getboolean('Camera', 'use_ip', fallback=False),
            'ip_url': config.get('Camera', 'ip_url', fallback=''),
            'device_id': config.getint('Camera', 'device_id', fallback=0),
            'width': config.getint('Camera', 'width', fallback=640),
            'height': config.getint('Camera', 'height', fallback=480),
            'auto_exposure': config.getfloat('Camera', 'auto_exposure', fallback=0.75),
        },
        'motion_detection': {
            'min_area': config.getint('MotionDetection', 'min_area'),
            'blur_size': (
                config.getint('MotionDetection', 'blur_size_x'),
                config.getint('MotionDetection', 'blur_size_y')
            ),
            'threshold': config.getint('MotionDetection', 'threshold'),
            'cooldown': config.getint('MotionDetection', 'cooldown'),
            # Number of frames to capture when motion is detected
            'n_frames': config.getint('MotionDetection', 'n_frames', fallback=5),
            # Seconds between captured frames
            'frames_interval': config.getfloat('MotionDetection', 'frames_interval', fallback=0.5),
         },
        'tts': {
            'enabled': config.getboolean('TTS', 'enabled'),
            'rate': config.getint('TTS', 'rate'),
            'volume': config.getfloat('TTS', 'volume')
        },
        'visualization': {
            'draw_contours': config.getboolean('Visualization', 'draw_contours'),
            'contour_color': contour_color,
            'contour_thickness': config.getint('Visualization', 'contour_thickness'),
            'draw_timestamp': config.getboolean('Visualization', 'draw_timestamp'),
            'timestamp_color': timestamp_color 
        },
        'email': {
            'enabled': config.getboolean('Email', 'enabled', fallback=False),
            'smtp_server': config.get('Email', 'smtp_server', fallback=''),
            'smtp_port': config.getint('Email', 'smtp_port', fallback=25),
            'smtp_username': config.get('Email', 'smtp_username', fallback=''),
            'smtp_password': config.get('Email', 'smtp_password', fallback=''),
            'from_address': config.get('Email', 'from_address', fallback=''),
            'to_address': config.get('Email', 'to_address', fallback=''),
            'use_tls': config.getboolean('Email', 'use_tls', fallback=True)
        }
    }

# Load configuration at startup
CONFIG = load_config()

def adjust_camera_settings(cap):
    """
    Adjust camera settings based on configuration
    """
    print("Adjusting camera settings...")
    # For IP cameras, setting hardware properties usually has no effect; skip those.
    if not CONFIG['camera'].get('use_ip', False):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, CONFIG['camera']['auto_exposure'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['camera']['height'])
        print(f"Camera resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    else:
        print("Using IP camera; skipping hardware property adjustments.")
    
    # Capture initial frames to allow camera to adjust exposure / prime stream
    ret_val, _ = cap.read()
    ret_val, _ = cap.read()

def tts(text):
    """
    Convert text to speech using configured TTS engine
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', CONFIG['tts']['rate'])
    engine.setProperty('volume', CONFIG['tts']['volume'])
    engine.say(text)
    engine.runAndWait()

def describe_image(image) -> str:
    """
    Describe image using configured AI service (Ollama or OpenAI-compatible API)
    """
    api_config = CONFIG['ai']
    # Convert image to base64
    image_b64 = base64.b64encode(image).decode('utf-8')
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_config['api_key']}"
    }
    
    payload = {
        'model': api_config['model'],
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': api_config['prompt']
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        'max_tokens': api_config.get('max_tokens', 300)
    }
    
    response = requests.post(
        f"{api_config['base_url']}/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"API request failed: {response.text}")
    
def describe_frames(images: list) -> str:
    """
    Describe motion across multiple images. `images` is a list of raw JPEG bytes.
    This function now sends only the first and last frame to the model and requests an English response.
    """
    api_config = CONFIG['ai']

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_config['api_key']}"
    }

    # Prepare a strong motion-focused prompt and require English output
    motion_prompt = api_config.get('motion_prompt') or api_config.get('prompt')
    motion_prompt = (
        "Say if a person/human is detected in any frame. Specify the frame and how many humans/persons are detected. Give short summary on what they are doing.\n"
        "If no person/human is visible in the frames then simply state that no human/person is seen. Don't try to infer that someone is there but not shown.\n"
        "Keep the answer as a single clear paragraph in English.\n\n"
        + motion_prompt
    )

    # Ensure we have at least one image
    if not images:
        return "No frames to describe."

    # Select only first and last frames
    first_img = images[0]
    last_img = images[-1] if len(images) > 1 else images[0]

    # Build content: prompt + labelled first/last images
    content_items = [
        {'type': 'text', 'text': motion_prompt},
        {'type': 'text', 'text': 'Frame 1 (first):'},
        {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64.b64encode(first_img).decode('utf-8')}"}},
        {'type': 'text', 'text': f'Frame {len(images)} (last):'},
        {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64.b64encode(last_img).decode('utf-8')}"}}
    ]

    payload = {
        'model': api_config['model'],
        'messages': [
            {
                'role': 'user',
                'content': content_items
            }
        ],
        'max_tokens': api_config.get('max_tokens', 600)
    }

    response = requests.post(
        f"{api_config['base_url']}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"API request failed: {response.status_code} {response.text}")
    
def detect_motion(frame1, frame2):
    """
    Detect motion between two frames and return (motion_detected, contours)
    """
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    gray1 = cv2.GaussianBlur(gray1, CONFIG['motion_detection']['blur_size'], 0)
    gray2 = cv2.GaussianBlur(gray2, CONFIG['motion_detection']['blur_size'], 0)
    
    # Calculate difference between frames
    frame_diff = cv2.absdiff(gray1, gray2)
    
    # Apply threshold
    thresh = cv2.threshold(frame_diff, CONFIG['motion_detection']['threshold'], 255, cv2.THRESH_BINARY)[1]
    
    # Dilate to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > CONFIG['motion_detection']['min_area']]
    
    return len(significant_contours) > 0, significant_contours

def draw_detection_info(frame, contours, timestamp):
    """
    Draw motion detection visualization on the frame
    """
    vis_config = CONFIG['visualization']
    annotated_frame = frame.copy()
    
    if vis_config['draw_contours']:
        # Draw all contours
        cv2.drawContours(
            annotated_frame, 
            contours, 
            -1,  # -1 means draw all contours
            vis_config['contour_color'],
            vis_config['contour_thickness']
        )
    
    if vis_config['draw_timestamp']:
        # Add timestamp to the image
        cv2.putText(
            annotated_frame,
            timestamp,
            (10, 30),  # Position (x, y) from top-left
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # Font scale
            vis_config['timestamp_color'],
            vis_config['contour_thickness'],
            cv2.LINE_AA
        )
    
    return annotated_frame

def log_event(image_path, description):
    """
    Log event details in JSONL format
    """
    log_file = CONFIG['save_directory'] + '/' + CONFIG['log_file']
    camera_info = {
        'resolution': f"{CONFIG['camera']['width']}x{CONFIG['camera']['height']}"
    }
    # Record camera identifier (device id or IP URL)
    if CONFIG['camera'].get('use_ip', False):
        camera_info['id'] = CONFIG['camera'].get('ip_url', '')
    else:
        camera_info['id'] = CONFIG['camera']['device_id']

    event = {
        'timestamp': datetime.now().isoformat(),
        'image_path': str(image_path),
        'description': description,
        'camera': camera_info,
        'motion_detection': {
            'min_area': CONFIG['motion_detection']['min_area'],
            'threshold': CONFIG['motion_detection']['threshold']
        },
        'model': CONFIG['ai']['model']
    }
    
    with open(log_file, 'a', encoding='utf-8') as f:
        json.dump(event, f, ensure_ascii=False)
        f.write('\n')
    
    return event

def ask_agent_should_alert(description: str) -> bool:
    """
    Use the configured AI to decide whether the event description indicates a person.
    Returns True if email should be sent (person detected), False otherwise.
    This function tries to parse a JSON response from the LLM; falls back to keyword checking.
    """
    api_config = CONFIG['ai']
    agent_model = api_config.get('AI', api_config.get('agent_model'))

    prompt = (
        "You are an automated security agent. Given the description below, decide whether this event involves a person or human activity and whether an email alert should be sent.\n\n"
        f"Description:\n{description}\n\n"
        "Answer in strict JSON with keys: {\n  \"alert\": true|false,\n  \"reason\": \"short explanation\"\n}\n"
        "Return only the JSON object and nothing else."
    )

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_config['api_key']}"
    }

    payload = {
        'model': agent_model,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'max_tokens': 200
    }

    try:
        resp = requests.post(f"{api_config['base_url']}/v1/chat/completions", headers=headers, json=payload, timeout=15)
        if resp.status_code != 200:
            print(f"Agent request failed: {resp.status_code} {resp.text}")
            # fallback to keyword check
            return ('person' in description.lower() or 'human' in description.lower())

        text = resp.json()['choices'][0]['message']['content']

        # Try to extract JSON from the model output
        try:
            # Sometimes model returns extra text; extract the first JSON object
            import re
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
            else:
                data = json.loads(text)

            return bool(data.get('alert'))
        except Exception:
            # Fallback simple heuristic
            return ('person' in text.lower() or 'human' in text.lower())

    except Exception as e:
        print(f"Agent call error: {e}")
        return ('person' in description.lower() or 'human' in description.lower())


def agent_decide_and_send(description: str, frames_bytes: list, image_path: str, timestamp: str):
    """
    Agent wrapper: delegate decision to the Agent in my_agent.run_agent. The agent
    itself will call the send_email_alert_tool if it decides to send an alert.
    This function invokes the agent and prints the final decision. Falls back to
    the simple LLM decisioner on any import/runtime error.
    """
    try:
        # run_agent returns True if an email was sent (or requested and sent), False otherwise
        sent = run_agent(description, frames_bytes, image_path, timestamp)
        if sent:
            print("Agent decided to send email and email was sent.")
        else:
            print("Agent decided NOT to send email: no human present.")
    except Exception as e:
        print(f"Agent invocation failed: {e}. Falling back to simple LLM decision.")
        # Fallback: use the original lightweight decision function
        try:
            should_send = ask_agent_should_alert(description)
            if should_send:
                # call the legacy email sender directly with up to 6 frames
                from my_agent import send_email_alert_tool
                send_email_alert_tool(image_path=str(image_path), description=description, timestamp=timestamp, frames_bytes=frames_bytes[:6])
                print("Fallback: Sent email based on simple LLM/heuristic decision.")
            else:
                print("Fallback: Decided not to send email.")
        except Exception as e2:
            print(f"Fallback also failed: {e2}")

def run_motion_detection():
    """
    Main function to run motion detection
    """
    # Ensure the save directory exists
    save_dir = Path(CONFIG['save_directory'])
    save_dir.mkdir(exist_ok=True)

    # Determine video source (IP URL or local device)
    cam_cfg = CONFIG['camera']
    if cam_cfg.get('use_ip', False):
        source = cam_cfg.get('ip_url')
    else:
        source = cam_cfg.get('device_id')

    # Access the video source
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    adjust_camera_settings(cap)

    if not cap.isOpened():
        # For IP streams it's often useful to retry; do a few attempts for robustness
        if cam_cfg.get('use_ip', False):
            print(f"Failed to open IP camera stream '{source}'. Retrying for 10 seconds...")
            start = time.time()
            while time.time() - start < 10:
                time.sleep(1)
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    print("Reconnected to IP camera stream.")
                    break
        if not cap.isOpened():
            raise IOError("Cannot open video source: {}".format(source))

    # Read two initial frames
    ret, frame1 = cap.read()
    if not ret or frame1 is None:
        print("Warning: failed to read initial frame from video source. Attempting up to 5 reconnects...")
        reconnect_attempts = 0
        max_reconnect = 5
        while reconnect_attempts < max_reconnect and (not (cap.isOpened() and ret)):
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(source)
            adjust_camera_settings(cap)
            ret, frame1 = cap.read()
            reconnect_attempts += 1

        if not ret or frame1 is None:
            print("Warning: unable to read initial frame after reconnect attempts. The loop will start and will try to recover on reads.")
            frame1 = None
    last_detection_time = 0

    print("Motion detection started. Press Ctrl+C to stop.")
    
    try:
        while True:
            # Read current frame
            ret, frame2 = cap.read()
            if not ret or frame2 is None:
                print("Warning: failed to read frame from video source.")
                # If using IP camera, try reconnecting up to 5 times then skip iteration
                if cam_cfg.get('use_ip', False):
                    reconnect_attempts = 0
                    max_reconnect = 5
                    reconnected = False
                    while reconnect_attempts < max_reconnect:
                        time.sleep(1)
                        cap.release()
                        cap = cv2.VideoCapture(source)
                        adjust_camera_settings(cap)
                        ret, frame2 = cap.read()
                        if ret and frame2 is not None:
                            reconnected = True
                            print("Reconnected to video source.")
                            break
                        reconnect_attempts += 1

                    if not reconnected:
                        print("Reconnect attempts failed; skipping this iteration.")
                        time.sleep(0.5)
                        continue
                else:
                    time.sleep(0.5)
                    continue

            # Ensure we have a previous frame to compare against
            if frame1 is None:
                frame1 = frame2.copy()
                # Not enough frames yet to detect motion
                time.sleep(0.1)
                continue

            # Skip motion detection if either frame is mostly gray
            if is_mostly_gray(frame1) or is_mostly_gray(frame2):
                # Optionally print or log this event
                # print("Skipped frame: mostly gray (likely corrupted or blank)")
                frame1 = frame2.copy()
                time.sleep(0.1)
                continue

            motion_detected, contours = detect_motion(frame1, frame2)
            
            if motion_detected:
                current_time = time.time()
                # Check if enough time has passed since last detection
                if current_time - last_detection_time > CONFIG['motion_detection']['cooldown']:
                    print("Motion detected!")
                    
                    # Create filename and path
                    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    filename = f"capture_{timestamp}.png"
                    image_path = save_dir / filename
                    last_detection_time = current_time

                    # Draw detection information on the frame
                    annotated_frame = draw_detection_info(frame2, contours, timestamp)

                    # Save the annotated image
                    # cv2.imwrite(str(image_path), annotated_frame)
                    # Capture a burst of frames (frame2 + N-1 more) at configured intervals
                    n_frames = int(CONFIG['motion_detection']['n_frames'])
                    frames_interval = float(CONFIG['motion_detection']['frames_interval'])

                    frames_bytes = []
                    # Include the previous frame (frame1) as the first attachment if available
                    if frame1 is not None:
                        _, jpg = cv2.imencode('.jpg', frame1)
                        frames_bytes.append(jpg.tobytes())

                    # Then include the current frame (frame2)
                    _, jpg = cv2.imencode('.jpg', frame2)
                    frames_bytes.append(jpg.tobytes())

                    # Poll for N-1 additional frames; prefer frames that differ from the last captured frame
                    last_captured_frame = frame2.copy()
                    # mean-difference threshold for accepting a new frame (tunable)
                    diff_threshold = 5.0

                    for i in range(max(0, n_frames - 1)):
                        start_poll = time.time()
                        captured = False
                        candidate = None

                        # Poll until we find a sufficiently different frame or we exceed frames_interval
                        while time.time() - start_poll < frames_interval:
                            ret_extra, candidate = cap.read()
                            if not ret_extra or candidate is None:
                                # Try a quick reconnect single attempt and continue polling
                                cap.release()
                                cap = cv2.VideoCapture(source)
                                adjust_camera_settings(cap)
                                ret_extra, candidate = cap.read()
                                if not ret_extra or candidate is None:
                                    time.sleep(0.05)
                                    continue

                            try:
                                gray_last = cv2.cvtColor(last_captured_frame, cv2.COLOR_BGR2GRAY)
                                gray_cand = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
                                mean_diff = float(np.mean(cv2.absdiff(gray_cand, gray_last)))
                            except Exception:
                                mean_diff = 255.0

                            if mean_diff > diff_threshold:
                                captured = True
                                break
                            # small sleep to avoid tight loop
                            time.sleep(0.05)

                        if not captured:
                            # fallback: if we read any candidate frame use it, else skip
                            if candidate is None:
                                print(f"Warning: failed to capture extra frame {i+1}; skipping")
                                continue
                            else:
                                frame_extra = candidate
                        else:
                            frame_extra = candidate

                        # Append encoded JPEG bytes and update last_captured_frame for next iteration
                        _, jpg = cv2.imencode('.jpg', frame_extra)
                        frames_bytes.append(jpg.tobytes())
                        last_captured_frame = frame_extra.copy()

                    if CONFIG['ai_description']:
                        try:
                            description = describe_frames(frames_bytes)
                        except Exception as e:
                            print(f"AI description failed: {e}")
                            description = "Motion detected"
                    else:
                        description = "Motion detected"

                    # Log the event
                    event = log_event(image_path, description)
                    
                    if CONFIG['ai_description']:
                        print(json.dumps(event, indent=2))
                    
                    if CONFIG['tts']['enabled']:
                        tts(description)
                    
                    # Use agent to decide whether to send email (agent will call send_email_alert tool)
                    agent_decide_and_send(description, frames_bytes, str(image_path), timestamp)
                    
            
            # Update frame1
            frame1 = frame2.copy()
            
            # Small delay to prevent high CPU usage
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping motion detection.")
    finally:
        cap.release()

if __name__ == "__main__":
    run_motion_detection()
