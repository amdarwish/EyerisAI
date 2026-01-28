# EyerisAI.py   
# Standard library imports
import base64
import configparser
import json
import os
import smtplib
import time
import threading
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from my_agent import run_motion_agent, run_qa_agent
# Third-party imports
import cv2
import numpy as np
import pyttsx3
import requests

# --- Item Counting Function ---
def count_items(image_path: str, item_name: str):
    """
    Count the number of specified items in an image using the configured AI model.
    """
    import cv2
    import base64
    import requests
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    # Encode as JPEG
    success, img_bytes = cv2.imencode('.jpg', image)
    if not success:
        print("Failed to encode image as JPEG.")
        return
    img_b64 = base64.b64encode(img_bytes.tobytes()).decode('utf-8')
    # Prepare prompt
    prompt = (
        f"You are an expert visual counter. Count how many '{item_name}' are present in the image. "
        f"Return ONLY a valid JSON object with a single key: \"count\" (integer). "
        f"If the item is not present, return {{\"count\": 0}}. Do not include any explanation. "
        f"Use double quotes for all keys and string values. Ensure all commas and brackets are correct so the output parses as valid JSON. Do not add any text before or after the JSON object."
    )
    api_config = CONFIG['ai']
    headers = {
        'Content-Type': 'application/json',
    }
    if api_config.get('api_key'):
        headers['Authorization'] = f"Bearer {api_config['api_key']}"
    payload = {
        'model': api_config['model'],
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ],
        'max_tokens': 100,
        # 'temperature': 0.0,
        # 'top_p': 1.0,
        'seed': 42
    }
    try:
        response = requests.post(f"{api_config['base_url']}/v1/chat/completions", headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            import re, json as pyjson
            m = re.search(r'\{[\s\S]*\}', content)
            if m:
                result = pyjson.loads(m.group(0))
                count = result.get('count', None)
                print(f"Counted {count} '{item_name}' in the image.")
            else:
                print("Could not parse count from model response:", content)
        else:
            print(f"API request failed: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Error during counting: {e}")



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

def load_config(config_path: str | os.PathLike | None = None):
    """
    Load configuration from config.ini
    """
    config = configparser.ConfigParser()
    if config_path is None:
        # Default to config.ini next to this file, so imports work regardless of CWD.
        config_path = Path(__file__).with_name("config.ini")
    config.read(str(config_path))
    
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
        'max_tokens': api_config.get('max_tokens', 300),
        # 'temperature': 0.0,
        # 'top_p': 1.0,
        'seed': 42
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
    
def describe_frames(images: list, prompt_text: str) -> str:
    """
    Describe frames using the provided prompt. `images` is a list of raw JPEG bytes.
    """
    api_config = CONFIG['ai']

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_config['api_key']}"
    }

    # Ensure we have at least one image
    if not images:
        return "No frames to describe."
    print(f"Describing {len(images)} frames with prompt: {prompt_text}")
    # Build content: prompt + all frames, each labeled
    content_items = [
        {'type': 'text', 'text': prompt_text}
    ]
    for idx, img in enumerate(images):
        content_items.append({'type': 'text', 'text': f'Frame {idx+1}:'})
        content_items.append({'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64.b64encode(img).decode('utf-8')}"}})

    payload = {
        'model': api_config['model'],
        'messages': [
            {
                'role': 'user',
                'content': content_items
            }
        ],
        'max_tokens': api_config.get('max_tokens', 10000),
        # 'temperature': 0.0,
        # 'top_p': 3.0,
        'seed': 42,
    }

    response = requests.post(
        f"{api_config['base_url']}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=360
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
        'max_tokens': 200,
        # 'temperature': 0.0,
        # 'top_p': 1.0,
        'seed': 42
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


def agent_decide_and_send(description: str, frames_bytes: list, image_path: str, timestamp: str, use_case: str = 'motion') -> bool:
    """
    Agent wrapper: delegate decision to the Agent in my_agent.run_motion_agent or run_qa_agent.
    The agent itself will call the send_email_alert_tool if it decides to send an alert.
    This function invokes the agent and prints the final decision. Falls back to the simple LLM decisioner on any import/runtime error.
    use_case: 'motion' or 'qa'
    """
    try:
        if use_case == 'motion':
            sent = run_motion_agent(description, frames_bytes, image_path, timestamp)
        else:
            sent = run_qa_agent(description, frames_bytes, image_path, timestamp)
        if sent:
            print("Agent decided to send email and email was sent.")
            return True
        else:
            print("Agent decided NOT to send email.")
            return False
        return bool(sent)
    except Exception as e:
        print(f"Agent invocation failed: {e}. Falling back to simple LLM decision.")
        try:
            should_send = ask_agent_should_alert(description)
            if should_send:
                from my_agent import send_email_alert_tool
                send_email_alert_tool(image_path=str(image_path), description=description, timestamp=timestamp, frames_bytes=frames_bytes[:6])
                print("Fallback: Sent email based on simple LLM/heuristic decision.")
                return True
            else:
                print("Fallback: Decided not to send email.")
                return False
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return False

def run_motion_detection(prompt: str = '', stop_event: threading.Event | None = None):
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
            if stop_event is not None and stop_event.is_set():
                print("Stop requested; exiting motion detection loop.")
                break
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
                        if stop_event is not None and stop_event.is_set():
                            print("Stop requested during reconnect; exiting.")
                            break
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

                    if stop_event is not None and stop_event.is_set():
                        break
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

                    # for i in range(max(0, n_frames - 1)):
                    #     start_poll = time.time()
                    #     captured = False
                    #     candidate = None

                    #     # Poll until we find a sufficiently different frame or we exceed frames_interval
                    #     while time.time() - start_poll < frames_interval:
                    #         ret_extra, candidate = cap.read()
                    #         if not ret_extra or candidate is None:
                    #             # Try a quick reconnect single attempt and continue polling
                    #             cap.release()
                    #             cap = cv2.VideoCapture(source)
                    #             adjust_camera_settings(cap)
                    #             ret_extra, candidate = cap.read()
                    #             if not ret_extra or candidate is None:
                    #                 time.sleep(0.05)
                    #                 continue

                    #         try:
                    #             gray_last = cv2.cvtColor(last_captured_frame, cv2.COLOR_BGR2GRAY)
                    #             gray_cand = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
                    #             mean_diff = float(np.mean(cv2.absdiff(gray_cand, gray_last)))
                    #         except Exception:
                    #             mean_diff = 255.0

                    #         if mean_diff > diff_threshold:
                    #             captured = True
                    #             break
                    #         # small sleep to avoid tight loop
                    #         time.sleep(0.05)

                    #     if not captured:
                    #         # fallback: if we read any candidate frame use it, else skip
                    #         if candidate is None:
                    #             print(f"Warning: failed to capture extra frame {i+1}; skipping")
                    #             continue
                    #         else:
                    #             frame_extra = candidate
                    #     else:
                    #         frame_extra = candidate

                    #     # Append encoded JPEG bytes and update last_captured_frame for next iteration
                    #     _, jpg = cv2.imencode('.jpg', frame_extra)
                    #     frames_bytes.append(jpg.tobytes())
                    #     last_captured_frame = frame_extra.copy()

                    if CONFIG['ai_description']:
                        try:
                            # Prepare motion prompt
                            api_config = CONFIG['ai']
                            motion_prompt = ""#prompt
                            motion_prompt = (
                                "Say if a person/human is detected in any frame. Specify the frame and how many humans/persons are detected. Give short summary on what they are doing.\n"
                                "If no person/human is visible in the frames then simply state that no human/person is seen. Don't try to infer that someone is there but not shown.\n"
                                "Keep the answer as a single clear paragraph in English.\n\n"
                                + motion_prompt
                            )
                            description = describe_frames(frames_bytes, motion_prompt)
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
                    
                    # Flush camera buffer to ensure fresh frames - read and discard several frames
                    print("Flushing camera buffer to start fresh...")
                    flush_count = 10  # Number of frames to flush from buffer
                    cap.release()
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                    adjust_camera_settings(cap)
                    for i in range(flush_count):
                        if stop_event is not None and stop_event.is_set():
                            print("Stop requested during buffer flush; exiting.")
                            break
                        ret_flush, _ = cap.read()
                        if not ret_flush:
                            # If flush fails, try reconnecting
                            cap.release()
                            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                            adjust_camera_settings(cap)
                            break
                        time.sleep(0.02)  # Small delay between buffer flushes
                    if stop_event is not None and stop_event.is_set():
                        break
                    
                    # Now capture completely fresh frames for next motion detection cycle
                    ret_fresh1, frame1 = cap.read()
                    if not ret_fresh1 or frame1 is None:
                        print("Warning: failed to read fresh frame1 after agent processing")
                        frame1 = frame2.copy()  # fallback to previous frame


            # Small delay to prevent high CPU usage
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping motion detection.")
    finally:
        cap.release()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EyerisAI: Motion Detection, Item Counting, and Video Analysis")
    parser.add_argument('--use-case', choices=['motion', 'count', 'video'], default='motion', help='Which use case to run: motion, count, or video')
    parser.add_argument('--image', type=str, help='Path to image file (required for count)')
    parser.add_argument('--item', type=str, help='Name of the item to count (required for count)')
    parser.add_argument('--video', type=str, help='Path to local video file (required for video mode)')
    parser.add_argument('--video-use-case', type=str, default=None, choices=['trash', 'bottles'], help='Predefined video use case: trash or bottles (optional for video mode)')
    parser.add_argument('--n-frames', type=int, default=3, help='Number of frames to extract (video mode, default: 3)')
    parser.add_argument('--interval', type=float, default=0.0, help='Interval in seconds to sample frames over (video mode, default: 10.0)')
    parser.add_argument('--prompt', type=str, default='', help='Custom prompt for the use case')
    args = parser.parse_args()

    if args.use_case == 'motion':
        run_motion_detection(prompt=args.prompt)
    elif args.use_case == 'count':
        if not args.image or not args.item:
            print("For counting, you must provide --image and --item arguments.")
        else:
            count_items(args.image, args.item)
    elif args.use_case == 'video':
        if not args.video:
            print("For video mode, you must provide --video argument with the path to the video file.")
        else:
            from my_agent import run_qa_agent
            
            # Define predefined prompts for video use cases
            def get_video_prompt(video_use_case, custom_prompt):
                if video_use_case == 'trash':
                    # return ("The frames provided are part of a video showing a machine in a lab that uses solid thin pins then dispenses them in the trash bin located in the bottom of the images."
                    #        "The trash bin is covered with a plastic shield. Your role is to detect for each frame if the pins are piling up to an extent that can cause an issue or not. "
                    #        "You should report in JSON foramt with keys being the frame number and values being 'Pins are piling up high, they can fall out of the bin, which is an issue' or 'Pins are at normal level, not piling up high'")
                    return ("""
                            # Analyze the provided video frames to determine if the black pins in the clear plastic trash bin (positioned at the bottom center of the image) are piling up. Use the following criteria:
                            # The pins are piling up if there is a clear pile.
                            # The other case: The pins aren't piling up if ther is no obvious pile for the pins.
                            # You should report in JSON foramt with keys being the frame number and values being 'Pins are piling up, which is an issue' or 'Pins are at normal level, not piling up'
                            You should report in JSON foramt with keys being the frame number and values being 'Pins are piling up, which is an issue'
                            """)
                elif video_use_case == 'bottles':
                    # return ("The frames provided are part of a video showing filled bottles moving on a conveyer belt in a production line. "
                    #        "Bottles being produced are passed from the left to the right of the image. In the middle of the image there is a metallic vertical tube with horizontal separator, where according to the flow, bottles are expected to be on the left side of the horizontal separator. "
                    #        "From the angle where the camera is placed, bottles shouldn't be infront of the horizontal separator. In some cases, one or few bottles could be pushed to the right of the horizontal separator in their flow, which according to the camera angle is infront of the horizontal separator. "
                    #        "You will need to report if you see any bottle seen to the right of the horizontal separator in the flow, which according to the camera angle will be seen is infront of the horizontal separator. "
                    #        "If no bottles are spotted right of the horizontal separator then state that all bottles are normally located left of the horizontal separator. "
                    #        "You should report in JSON format with keys being the frame number and values being 'bottle or more are pushed to the wrong position, which is an issue' or 'the bottles are in the normal position.'. "
                    #        "Keep the answer strictly in JSON format and nothing else.")
                    return ("""
                            Analyze this frame from a bottle production line. The critical reference is the fixed metallic arm with horizontal separator (the metal bar) in the center.
                            Normal case: All bottles are positioned behind the horizontal separator arm (i.e., moving away from the arm toward the right of the frame).
                            Anomaly case: Any bottle is positioned in front of the horizontal separator arm (i.e., to the left of the arm's metal bar in the frame).
                            You should report in JSON format with keys being the frame number and values being 'bottle or more are pushed to the wrong position, which is an issue (anomaly case)' or 'the bottles are in the normal position (normal case).'
                            Keep the answer strictly in JSON format and nothing else.
                            """)
                else:
                    return custom_prompt
            
            def extract_and_describe_video_segments(video_path, n_frames, interval_sec, prompt=''):
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Failed to open video file: {video_path}")
                    return
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                if fps <= 0 or total_frames <= 0:
                    print("Invalid video file or unable to read FPS/frames.")
                    cap.release()
                    return
                # If interval is 0.0 (default), use the entire video duration
                if interval_sec == 0.0:
                    interval_sec = duration
                    print(f"Using entire video duration as interval: {duration:.2f} seconds")
                segment = 0
                start_time = 0.0
                while start_time < duration:
                    if n_frames > 1:
                        frame_indices = [int((start_time + i * interval_sec / (n_frames - 1)) * fps) for i in range(n_frames)]
                        # Ensure the last frame index doesn't exceed total_frames - 1
                        frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]
                    else:
                        frame_indices = [int(start_time * fps)]
                    frames_bytes = []
                    for frame_num, idx in enumerate(frame_indices):
                        if idx >= total_frames:
                            continue
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            print(f"Failed to read frame at index {idx}")
                            continue
                        success, jpg = cv2.imencode('.jpg', frame)
                        if not success:
                            print(f"Failed to encode frame at index {idx}")
                            continue
                        frames_bytes.append(jpg.tobytes())
                        # Save the frame as a JPEG file
                        save_dir = Path(CONFIG['save_directory'])
                        save_dir.mkdir(exist_ok=True)
                        frame_filename = save_dir / f"video_segment{segment+1}_frame{frame_num+1}_idx{idx}.jpg"
                        cv2.imwrite(str(frame_filename), frame)
                        # # Save the frame as a JPEG file
                        # save_dir = Path(CONFIG['save_directory'])
                        # save_dir.mkdir(exist_ok=True)
                        # frame_filename = save_dir / f"video_segment{segment+1}_frame{frame_num+1}_idx{idx}.jpg"
                        # cv2.imwrite(str(frame_filename), frame)
                    print(f"Extracted {len(frames_bytes)} frames for segment {segment+1} (from {start_time:.2f}s to {min(start_time+interval_sec, duration):.2f}s)")
                    if frames_bytes:
                        try:
                            # Prepare video prompt
                            # video_prompt = (
                            #     """
                                # The frame(s) provided is/are part of a video showing filled bottles moving on a conveyer belt in a production line.
                                # Bottles being produced are passed from the left to the right of the image. In the middle of the image there is a metallic arm separator, where according to the flow are bottles are expected to be located on its left side, which is the upper part of the image. 
                                # In some cases, one or few bottles could be pushed to the right of the separator which will be in the bottom of the image. You will need to report if you see any bottle in the right of the separator.
                                # If no bottles are spotted right of the separator (bottom of the image) then state that all bottles are normally located left of the separator (top of the image).
                                # You should report in JSON format with keys being the frame number and values being "bottle or more are pushed to the wrong position, which is an issue" or "the bottles are in the normal position.".
                                # Keep the answer strictly in JSON format and nothing else.
                            #     """
                            # )
                            # lab_prompt = (
                            #     """
                            #     The frames provided are part of a video showing a machine in a lab that uses solid thin pins then dispenses them in the trash bin located in the bottom of the images.
                            #     The trash bin is covered with a plastic shield. Your role is to detect for each frame if the pins are piling up to an extent that can cause an issue or not.
                            #     You should report in JSON foramt with keys being the frame number and values being 'Pins are piling up high, they can fall out of the bin, which is an issue' or 'Pins are still at normal level'
                            #     """
                            # )
                            description = describe_frames(frames_bytes, prompt)
                            print(f"\nSegment {segment+1} (from {start_time:.2f}s to {min(start_time+interval_sec, duration):.2f}s):")
                            print(description)
                            # Call QA agent for each segment
                            run_qa_agent(description, frames_bytes, video_path, f"segment_{segment+1}")
                        except Exception as e:
                            print(f"Failed to describe frames for segment {segment+1}: {e}")
                    else:
                        print(f"No frames extracted for segment {segment+1}.")
                    segment += 1
                    start_time += interval_sec
                cap.release()
            print("Current time:", datetime.now().isoformat())
            final_prompt = get_video_prompt(args.video_use_case, args.prompt)
            extract_and_describe_video_segments(args.video, args.n_frames, args.interval, prompt=final_prompt)
            print("Current time:", datetime.now().isoformat())
