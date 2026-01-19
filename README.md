![EyerisAI-Mac-M4-Mini](images/EyerisAI-banner-image2.png)

# EyerisAI 🧿
AI powered camera and event detection system with web interface

## What is EyerisAI?

EyerisAI (pronounced "IrisAI", 👁️ + 🤖) is a project to create an AI-powered camera and event detection system that uses a computer, webcam, computer vision, and a multi-modal LLM (💻 + 📷 + 👁️ + 🤖) to "watch" for specific events in real time. When it detects an event, EyerisAI uses generative AI to analyze the scene, log the findings, and respond, either by speaking aloud (TTS) or sending an email alert.

This system now includes both a command-line interface and a modern **web-based frontend** with three specialized use cases:
- **Security**: Motion detection for restricted area access
- **Trash Monitoring**: Smart bin status monitoring and overflow detection
- **Production Line**: Conveyor belt anomaly detection for quality control

The system features a React frontend for easy interaction and a FastAPI backend for real-time AI analysis.

## 🏗️ Project Structure

```
EyerisAI/
├── backend/                 # Backend Python API
│   ├── api_server.py       # Main FastAPI server
│   ├── EyerisAI.py         # Core vision processing
│   ├── my_agent.py         # AI agent logic
│   ├── config.ini          # Configuration file
│   ├── requirements-core.txt # Python dependencies
│   └── __init__.py         # Python package marker
├── Frontend/               # React frontend
│   ├── src/                # React source code
│   ├── package.json        # Node.js dependencies
│   └── .env                # Frontend environment
├── launcher.py             # Python launcher script
├── dev-backend.sh          # Backend development script
└── README.md              # This file
```

## 🚀 Quick Start

### Option 1: Full System (Recommended)
```bash
python launcher.py
```
This starts the backend API server with comprehensive logging and monitoring.

### Option 2: Development Mode
```bash
# Backend only (with auto-reload)
./dev-backend.sh

# Frontend only (separate terminal)
cd Frontend
npm install
npm run dev
```

## 🌐 Access Points

- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Frontend UI**: http://localhost:5173 (when running frontend)
- **System Status**: http://localhost:8000/status

## 📋 Specialized Use Cases

### 1. **Security (Motion Detection)**
- **Mode**: Live camera monitoring
- **Purpose**: Detects unauthorized access to restricted areas
- **Features**: Real-time motion detection with email alerts
- **Endpoint**: `POST /motion/start`, `POST /motion/stop`

### 2. **Trash Bin Monitoring**
- **Mode**: Video upload analysis
- **Purpose**: Monitors trash bin status and overflow conditions
- **Features**: Upload videos for smart analysis
- **Endpoint**: `POST /video/analyze` (use_case: trash)

### 3. **Production Line Monitoring**
- **Mode**: Video upload analysis  
- **Purpose**: Detects bottle positioning anomalies on conveyor belt
- **Features**: Quality control for manufacturing processes
- **Endpoint**: `POST /video/analyze` (use_case: bottles)

## ⚙️ Configuration

Backend configuration is stored in `backend/config.ini`:

- **AI Settings**: Model, API endpoints, prompts for each use case
- **Camera Settings**: Device ID, IP camera, resolution
- **Email Alerts**: SMTP configuration for notifications
- **Motion Detection**: Sensitivity, thresholds, cooldowns

## Potential Use Cases

There are many potential applications of an AI enabled camera system. You're imagination is the limit. Here are some potential focus areas and their use cases:

**Healthcare Monitoring**
 - Patient fall detection in medical facilities
 - Elder care movement monitoring
 - Behavioral health observation
 - Emergency response triggering

**Security Applications**
- Intelligent home surveillance
- Business premises monitoring
- Weapon detection
- Suspicious activity alerts

**Family Care**
- Smart baby monitoring
- Child safety oversight
- Pet behavior tracking
- Elderly care assistance

**Professional Monitoring**
- Workplace safety compliance
- Construction site supervision
- Equipment monitoring
- Quality control observation

**Research & Analysis**
- Wildlife observation
- Experimental observation and logging


## Examples

### Hospital Fall Detection

Here's an example of the detecting a patient falling from a hospital bed *(to simulate this event, I used a virtual webcam to feed a youtube video of a commercial for a fall mat to EyerisAI)*. I used a prompt to tell the AI that it was a fall detection system and should report and incidents that it detected:

![fall-detection-image](images/fall-detection_large.gif)

The resulting email alert that was sent in response to the AI determining that a fall event had occurred:

![fall-detection-email-alert](images/fall-dectection-alert.png)

### CCTV Monitoring

In this example, I captured myself entering my office (wearing a headlamp to obscure my face). I provided a prompt to the AI telling it that it was a security camera and it was to watch for and alert when it detected any activity while I was out of town: 

![home-security-image](images/home-security_large.gif)

The email alert that it send me in response to detecting a potential intruder:

![home-secuirty-alert](images/home-secuirty-alert.png)

## Hardware and Setup used for testing and development

For the development and testing of this system I'm using the following setup:

- Apple M4 Mac Mini, 16GB RAM (base model)
- Logitech C920 HD Pro Webcam
- Ollama 0.5.11
- Moondream2 local LLM
- LiteLLM Proxy (**\*optional\***, not need if only using local models via Ollama)

The above setup will run complexly locally. I also tested integrating with OpenAI models (gpt4o) via Azure AI, and Anthropic (Claude 3.5 Sonnet) via Amazon AWS Bedrock. Both the OpenAI and Anthropic models were accessed via LiteLLM proxy running locally. See my article on how to setup LiteLLM if interested: [Centralizing Multiple AI Services with LiteLLM Proxy](https://robert-mcdermott.medium.com/centralizing-multi-vendor-llm-services-with-litellm-9874563f3062) 

If you are going to run a private LLM like Moondream locally, a GPU is recommended, but not required. Moondream is only 1.8B parameters and requires only 4.5GB to run, so it will technically run without a GPU, but it will take it a minute (guess) to process a detection using just the CPU.  If you are using an OpenAI compatible API located remotely, then a GPU is not used, so even a very low powered system (IOT device?) can be used as an EyerisAI camera node and function well.

## Install

Clone this repository to the system you want to run EyerisAI on:

```bash
git clone https://github.com/robert-mcdermott/EyerisAI.git
cd EyerisAI
```

### Backend Dependencies

If you are using ***pip*** 

```bash
cd backend
pip install -r requirements-core.txt
```

Alternatively, if you are cool 😎, and are using ***uv***:

```bash
uv sync
```

### Frontend Dependencies (Optional)

For the web interface:

```bash
cd Frontend
npm install
```

## Configure

Edit the ***backend/config.ini*** file to suit your needs. Here are some of the most likely variables you'll want to change:

Update the name of this camera, where it's located, what it's purpose it or something to identify it:

```ini
[General]
instance_name = Give this Camera a name 
```

Adjust the sensitivity of the motion detection. The **min_area** is the area size in pixels that needs to change to trigger. Smaller is more sensitive, larger is less. The **threshold** is the intensity of pixel value shifts required to trigger detection, smaller is more sensitive, larger is less. **cooldown** is the time to wait before another motion detection even can be triggered to prevent too many events.

```ini 
[MotionDetection]
min_area = 700
threshold = 50
cooldown = 3
```
If you have multiple cameras registered on your system, you may need to change the **device_index** to select the correct camera (this may require some trial and error to find the desired camera):

```ini
[Camera]
device_id = 0
```

If you want the system to speak aloud the AI's response (description of what it sees), change the **enabled** variable int the TTS section to *true*:

```ini
[TTS]
enabled = false
```

In the *AI* section, you select that LLM model that you want to use. If you want use a local model, make sure that ollama is install and that you have pulled the model that use want to use. "Moondream" is a good local model to start with. EyerisAI can work with any OpenAI compatible API (like LiteLLM Proxy) and multi-modal LLM. The *prompt* variable is important, that's were you provide the AI with instructions what you want the AI to examine in the image, or how to act, respond. 

```ini
[AI]
# Ollama model and endpoint
model = moondream
base_url = http://localhost:11434
# Any OpenAI compatible API endpoint:
#model = gpt4o
#base_url = https://api.openai.com/v1
api_key = sk-12345
prompt = You are a CCTV camera that has detected motion. The areas where motion was detected are indicated with magenta contours. Examine the image and report what activity you see, especially any humans visible.
```

There are other things that can be configured or adjusted, such as sending emails, that aren't covered here but should be self explanatory.

## Ollama

If you will be running the AI inference locally on the system with the camera and running EyerisAI, you'll need to first have Ollama running with the models that you'll be using downloaed. I'm not going to cover this in detail here but, I provide detailed instructions in my article ["LLM Zero-to-Hero with Ollama"](https://blog.cubed.run/llm-zero-to-hero-with-ollama-913e50d6b7f0) 

## Running 

### Method 1: Using the Launcher (Recommended)

The Python launcher provides the easiest way to start the system:

```bash
python launcher.py
```

This will:
- Check all configuration files
- Start the FastAPI backend server
- Provide comprehensive logging
- Handle port cleanup automatically

### Method 2: Development Mode

For development with auto-reload:

```bash
./dev-backend.sh
```

### Method 3: Manual Startup

To run the original command-line interface:

```bash
python backend/EyerisAI.py
```

Or for the cool 😎 ***uv*** folks:

```bash
uv run backend/EyerisAI.py
```

### Frontend (Optional)

To run the web interface (in a separate terminal):

```bash
cd Frontend
npm run dev
```

Then visit http://localhost:5173 to use the web interface.

## 🔧 API Endpoints

- `GET /status` - System status and configuration
- `POST /motion/detect` - Analyze single image for motion
- `POST /video/analyze` - Analyze video for trash/production issues  
- `POST /count/items` - Count specific items in images
- `GET /logs` - Retrieve system logs

## 🐛 Troubleshooting

### Common Issues
1. **Port 8000 already in use**: The launcher automatically kills existing processes
2. **Module import errors**: Ensure you're running from the root directory
3. **Camera not available**: Normal for development, check backend/config.ini for camera settings
4. **Dependencies missing**: Run `pip install -r backend/requirements-core.txt`

### Development Tips
- Use `python launcher.py` for the most reliable startup experience
- Check logs for detailed error messages
- API documentation available at http://localhost:8000/docs

## Output

The images where motion was detected are PNG images named after their date-time stamp and located in the ***captures** directory in the project folder.

The detection log that captures details about the even, such as the date/time, model that was used and the AI generated description of the image event are in JSONL format and are available in the ***'captures/motion_events.jsonl'*** file.

Example Log entry:

```json
{
  "timestamp": "2025-02-17T13:51:15.599307",
  "image_path": "captures/capture_2025-02-17_13-51-08.png",
  "description": "⚠️ SECURITY ALERT ⚠️\n\nTime: 13:51:08\nDate: 02/17/2025\nLocation: Living Room\n\nPOTENTIAL INTRUDER DETECTED\n- Subject wearing dark clothing\n- Using headlamp with bright LED light\n- Light appears to be deliberately obscuring facial features\n- Subject's movement detected in living room area\n\n",
  "camera": {
    "id": 0,
    "resolution": "1920x1080"
  },
  "motion_detection": {
    "min_area": 700,
    "threshold": 50
  },
  "model": "moondream"
}
```

