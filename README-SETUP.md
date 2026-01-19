# EyerisAI - AI-Powered Surveillance and Monitoring System

EyerisAI is a comprehensive AI surveillance system that supports three main use cases:

1. **Restricted Area Access** (Motion Detection) - Real-time monitoring for unauthorized access
2. **Trash Bin Status** (Video Analysis) - Monitor trash bin fill levels and overflow conditions  
3. **Production Line Anomaly** (Video Analysis) - Detect production line issues and bottle positioning problems

## Architecture

- **Frontend**: React + TypeScript + Vite + Tailwind CSS
- **Backend**: FastAPI + Python + OpenCV + LangGraph
- **AI Integration**: Supports OpenAI-compatible APIs and Ollama

## Project Structure

```
EyerisAI/
├── api_server.py              # FastAPI backend server
├── EyerisAI.py               # Core motion detection and analysis logic
├── my_agent.py               # LangGraph agents for decision making
├── config.ini                # Configuration file
├── requirements-api.txt       # Python dependencies for API
├── start.sh                  # Startup script
├── .env.example              # Environment variables template
└── Frontend/                 # React frontend
    ├── src/
    │   ├── App.tsx           # Main application component
    │   └── components/       # React components
    ├── package.json          # Frontend dependencies
    └── .env                  # Frontend environment variables
```

## Quick Start

### Prerequisites

- Python 3.8+ with pip
- Node.js 16+ with npm
- A compatible camera (USB or IP camera)

### Installation & Setup

1. **Clone and navigate to the project:**
   ```bash
   cd EyerisAI
   ```

2. **Configure the system:**
   ```bash
   # Copy and edit environment file
   cp .env.example .env
   
   # Edit config.ini for camera, AI, and email settings
   # Key settings to configure:
   # - AI API endpoint and model
   # - Camera settings (USB device ID or IP camera URL)
   # - Email SMTP settings (optional)
   ```

3. **Run the startup script:**
   ```bash
   ./start.sh
   ```

   This script will:
   - Create a Python virtual environment
   - Install all dependencies (backend and frontend)
   - Start the FastAPI backend server
   - Start the React development server

### Manual Setup (Alternative)

If you prefer manual setup:

1. **Backend Setup:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements-api.txt
   python api_server.py
   ```

2. **Frontend Setup (in a new terminal):**
   ```bash
   cd Frontend
   npm install
   npm run dev
   ```

## Access Points

Once running, access the system at:

- **Frontend UI**: http://localhost:5173
- **Backend API**: http://localhost:8000  
- **API Documentation**: http://localhost:8000/docs
- **API Status**: http://localhost:8000/status

## Usage

### 1. Restricted Area Access (Live Motion Detection)

- Select "Security" use case in the UI
- Switch to "Live" mode
- Click "Start Detection"
- The system will monitor camera feed for motion/people
- Alerts are sent via email when unauthorized access is detected

**Equivalent CLI Command:**
```bash
python EyerisAI.py --use-case motion
```

### 2. Trash Bin Status Analysis

- Select "Trash" use case in the UI
- Upload a video file of the trash bin
- Click "Start Analysis"
- System analyzes trash levels and overflow conditions

**Equivalent CLI Command:**
```bash
python EyerisAI.py --use-case video --video path/to/video.mp4 --video-use-case trash --n-frames 3 --interval 5.0
```

### 3. Production Line Anomaly Detection

- Select "Production" use case in the UI  
- Upload a video file of the production line
- Click "Start Analysis"
- System detects bottle positioning issues and production anomalies

**Equivalent CLI Command:**
```bash
python EyerisAI.py --use-case video --video path/to/video.mp4 --video-use-case bottles --n-frames 4 --interval 2.0
```

## API Endpoints

The FastAPI backend provides these REST endpoints:

### System
- `GET /status` - System status and configuration
- `GET /logs` - Recent log entries

### Motion Detection  
- `POST /motion/start` - Start live motion detection
- `POST /motion/stop` - Stop motion detection
- `POST /motion/detect` - Analyze single image for motion

### Video Analysis
- `POST /video/analyze` - Analyze video for trash/production issues
  - Form data: `file` (video), `use_case` (trash/bottles), `n_frames`, `interval`

### Item Counting
- `POST /count/items` - Count specific items in an image
  - Form data: `file` (image), `item_name` (string)

## Configuration

### config.ini

Key configuration sections:

```ini
[General]
save_directory = ./captures
ai_description = true
instance_name = EyerisAI

[AI]
base_url = http://localhost:11434
model = llama3.2-vision:11b
api_key = your_api_key

[Camera] 
use_ip = false
device_id = 0
# For IP camera: use_ip = true, ip_url = rtsp://...

[Email]
enabled = true
smtp_server = smtp.gmail.com
smtp_port = 587
# ... other email settings
```

### Environment Variables

**Backend (.env):**
```env
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:5173
```

**Frontend (.env):**
```env
VITE_API_URL=http://localhost:8000
```

## Development

### Backend Development
```bash
# Run with auto-reload
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

# Run tests (if available)
pytest tests/
```

### Frontend Development
```bash
cd Frontend
npm run dev        # Development server
npm run build      # Production build  
npm run lint       # Linting
```

### Adding New Use Cases

1. Update `UseCaseSelector.tsx` with new use case definition
2. Add corresponding logic in `api_server.py`
3. Update `App.tsx` to handle the new use case
4. Add agent logic in `my_agent.py` if needed

## Troubleshooting

### Common Issues

1. **Camera not detected**: Check camera connection and permissions
2. **API connection failed**: Verify API server is running on port 8000
3. **AI model errors**: Ensure AI service (Ollama/OpenAI) is configured correctly
4. **Email alerts not working**: Check SMTP settings in config.ini

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python api_server.py
```

### Logs Location

- Backend logs: Console output and `./captures/events.jsonl`
- Frontend logs: Browser developer console

## License

See LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Verify configuration settings
4. Check API documentation at `/docs` endpoint