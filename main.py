import sys
import logging
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from main_window import WebcamClearVSTPro

def setup_logging():
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.DEBUG, # Keep app logs at DEBUG
        format=log_format,
        handlers=[
            # mode='w' overwrites the log file on every app launch
            logging.FileHandler("app_debug.log", mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout) # Also prints to console
        ]
    )
    
    # Silence Numba's extremely verbose internal compiler logs
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numba.core").setLevel(logging.WARNING)

if __name__ == "__main__":
    setup_logging()
    logging.info("Starting Webcam Recorder Application...")
    
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    window = WebcamClearVSTPro()
    window.show()
    sys.exit(app.exec())