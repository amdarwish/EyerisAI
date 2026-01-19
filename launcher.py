#!/usr/bin/env python3
"""
EyerisAI Launcher Script
A Python-based launcher that starts the backend API server with comprehensive logging.
"""

import os
import sys
import time
import logging
import signal
import subprocess
from datetime import datetime
from pathlib import Path

# Setup logging
def setup_logging():
    """Configure logging to both file and console"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eyerisai_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def check_dependencies(logger):
    """Check if required dependencies are available"""
    logger.info("🔍 Checking dependencies...")
    
    # Check Python version
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        logger.error("❌ Python 3.8 or higher is required")
        return False
    
    # Check required modules
    critical_modules = [
        'fastapi',
        'uvicorn'
    ]
    
    optional_modules = [
        'opencv-python',
        'numpy',
        'requests'
    ]
    
    missing_critical = []
    missing_optional = []
    
    for module in critical_modules:
        try:
            __import__(module.replace('-', '_'))
            logger.info(f"✅ {module} - Available")
        except ImportError:
            logger.error(f"❌ {module} - Not found")
            missing_critical.append(module)
    
    for module in optional_modules:
        try:
            if module == 'opencv-python':
                import cv2
                logger.info(f"✅ {module} - Available")
            else:
                __import__(module.replace('-', '_'))
                logger.info(f"✅ {module} - Available")
        except ImportError:
            logger.warning(f"⚠️ {module} - Not found")
            missing_optional.append(module)
    
    if missing_critical:
        logger.error(f"Missing critical modules: {missing_critical}")
        logger.error("Install them with: pip install " + " ".join(missing_critical))
        return False
    
    if missing_optional:
        logger.warning(f"Missing optional modules: {missing_optional}")
        logger.info("Some features may be limited. Install them with: pip install " + " ".join(missing_optional))
    
    return True

def check_config_files(logger):
    """Check if configuration files exist"""
    logger.info("🔍 Checking configuration files...")
    
    required_files = [
        'backend/config.ini',
        'backend/EyerisAI.py',
        'backend/my_agent.py',
        'backend/api_server.py'
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            logger.info(f"✅ {file} - Found")
        else:
            logger.error(f"❌ {file} - Missing")
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    return True

def check_and_kill_port(port, logger):
    """Check if port is in use and kill the process"""
    try:
        result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                logger.info(f"🔄 Killing existing process on port {port}: PID {pid}")
                subprocess.run(['kill', pid], check=False)
                time.sleep(1)
    except Exception as e:
        logger.warning(f"Could not check/kill port {port}: {e}")

def start_backend_server(logger):
    """Start the FastAPI backend server"""
    logger.info("🚀 Starting EyerisAI Backend Server...")
    
    # Check and clean up port 8000
    check_and_kill_port(8000, logger)
    
    try:
        # Try to start with uvicorn if available
        try:
            import uvicorn
            logger.info("Starting with uvicorn...")
            
            # Change to backend directory for proper imports
            os.chdir("backend")
            
            # Start uvicorn server
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            return process
            
        except ImportError:
            logger.warning("uvicorn not found, trying direct execution...")
            
            # Try direct execution of main API server
            try:
                os.chdir("backend")
                process = subprocess.Popen(
                    [sys.executable, "api_server.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                return process
            except Exception:
                logger.warning("Main API server failed, trying simplified server...")
                
                # Fallback to simplified server
                process = subprocess.Popen(
                    [sys.executable, "api_server_simple.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                return process
    
    except Exception as e:
        logger.error(f"❌ Failed to start backend server: {e}")
        return None

def monitor_process(process, logger, name):
    """Monitor a subprocess and log its output"""
    logger.info(f"📊 Monitoring {name} process (PID: {process.pid})")
    
    try:
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Log the output from the subprocess
                logger.info(f"[{name}] {output.strip()}")
    
    except Exception as e:
        logger.error(f"Error monitoring {name}: {e}")
    
    # Check if process ended
    return_code = process.poll()
    if return_code is not None:
        logger.warning(f"{name} process ended with return code: {return_code}")
    
    return return_code

def test_api_server(logger):
    """Test if the API server is responding"""
    logger.info("🧪 Testing API server...")
    
    import time
    time.sleep(2)  # Give server time to start
    
    try:
        import requests
        response = requests.get("http://localhost:8000/status", timeout=10)
        if response.status_code == 200:
            logger.info("✅ API server is responding")
            data = response.json()
            logger.info(f"Server status: {data}")
            return True
        else:
            logger.warning(f"⚠️ API server returned status code: {response.status_code}")
            return False
    except ImportError:
        logger.warning("requests module not available, cannot test API")
        return True
    except Exception as e:
        logger.error(f"❌ API server test failed: {e}")
        return False

def show_info(logger):
    """Show information about the running services"""
    logger.info("\n" + "="*60)
    logger.info("🎉 EyerisAI Backend is running!")
    logger.info("="*60)
    logger.info("📍 Backend API: http://localhost:8000")
    logger.info("📍 API Documentation: http://localhost:8000/docs")
    logger.info("📍 System Status: http://localhost:8000/status")
    logger.info("📍 Logs: Check logs/ directory for detailed logs")
    logger.info("="*60)
    logger.info("Use Ctrl+C to stop the server")
    logger.info("="*60 + "\n")

def main():
    """Main launcher function"""
    logger = setup_logging()
    
    logger.info("🚀 EyerisAI Launcher Starting...")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python executable: {sys.executable}")
    
    # Check dependencies
    if not check_dependencies(logger):
        logger.error("❌ Dependency check failed")
        sys.exit(1)
    
    # Check configuration files
    if not check_config_files(logger):
        logger.error("❌ Configuration check failed")
        sys.exit(1)
    
    # Start backend server
    backend_process = start_backend_server(logger)
    
    if backend_process is None:
        logger.error("❌ Failed to start backend server")
        sys.exit(1)
    
    # Test API server
    if test_api_server(logger):
        show_info(logger)
    else:
        logger.warning("⚠️ API server may not be working properly")
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\n🛑 Received shutdown signal...")
        if backend_process and backend_process.poll() is None:
            logger.info("Terminating backend server...")
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing backend server...")
                backend_process.kill()
        logger.info("✅ Shutdown complete")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Monitor the backend process
        return_code = monitor_process(backend_process, logger, "Backend")
        if return_code != 0:
            logger.error(f"❌ Backend process exited with code: {return_code}")
            sys.exit(return_code)
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Keyboard interrupt received")
        signal_handler(signal.SIGINT, None)
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if backend_process:
            backend_process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()