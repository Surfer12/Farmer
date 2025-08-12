#!/usr/bin/env python3
"""
Python Launcher for HR and IT Assistant Services
Runs both FastAPI services concurrently using uvicorn
"""

import asyncio
import subprocess
import sys
import os
import signal
import time
from typing import List, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceLauncher:
    """Manages concurrent FastAPI service processes"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.services_dir = Path(__file__).parent.parent / "assistants"
        
    def start_service(self, service_name: str, port: int) -> Optional[subprocess.Popen]:
        """Start a single FastAPI service"""
        service_path = self.services_dir / f"{service_name}.py"
        
        if not service_path.exists():
            logger.error(f"Service file not found: {service_path}")
            return None
        
        try:
            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                f"services.assistants.{service_name}:app",
                "--host", "0.0.0.0",
                "--port", str(port),
                "--reload",
                "--log-level", "info"
            ]
            
            logger.info(f"Starting {service_name} on port {port}...")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent.parent.parent  # Workspace root
            )
            
            # Give the service time to start
            time.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info(f"✓ {service_name} started successfully on port {port}")
                return process
            else:
                logger.error(f"✗ {service_name} failed to start")
                return None
                
        except Exception as e:
            logger.error(f"Error starting {service_name}: {e}")
            return None
    
    def stop_all_services(self):
        """Stop all running services"""
        logger.info("Stopping all services...")
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        logger.info("All services stopped")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_all_services()
        sys.exit(0)
    
    async def monitor_services(self):
        """Monitor service health and restart if needed"""
        while True:
            for i, process in enumerate(self.processes):
                if process.poll() is not None:
                    service_name = "hr_assistant" if i == 0 else "it_assistant"
                    port = 8001 if i == 0 else 8002
                    logger.warning(f"{service_name} crashed, restarting...")
                    
                    new_process = self.start_service(service_name, port)
                    if new_process:
                        self.processes[i] = new_process
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    def run(self):
        """Main launcher function"""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("=" * 60)
        logger.info("FastAPI Assistant Services Launcher")
        logger.info("=" * 60)
        
        # Start HR Assistant
        hr_process = self.start_service("hr_assistant", 8001)
        if hr_process:
            self.processes.append(hr_process)
        
        # Start IT Assistant
        it_process = self.start_service("it_assistant", 8002)
        if it_process:
            self.processes.append(it_process)
        
        if not self.processes:
            logger.error("No services could be started")
            sys.exit(1)
        
        logger.info("-" * 60)
        logger.info("Services Running:")
        logger.info("  HR Assistant: http://localhost:8001")
        logger.info("  IT Assistant: http://localhost:8002")
        logger.info("-" * 60)
        logger.info("API Documentation:")
        logger.info("  HR Assistant: http://localhost:8001/docs")
        logger.info("  IT Assistant: http://localhost:8002/docs")
        logger.info("-" * 60)
        logger.info("Press Ctrl+C to stop all services")
        logger.info("=" * 60)
        
        try:
            # Run the monitoring loop
            asyncio.run(self.monitor_services())
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all_services()

def main():
    """Entry point"""
    launcher = ServiceLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
