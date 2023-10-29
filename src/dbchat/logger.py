import os
from datetime import datetime
from pathlib import Path
import subprocess
import logging

class GitLogger(logging.Logger):

    def __init__(self, log_dir, filename=None):
        self.log_dir = log_dir
        if filename is None:
            filename = f"log{datetime.now().strftime('%Y-%m-%d')}.log"
        self.filename = filename
        self._fpath = Path(self.log_dir, self.filename)
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        self.setup_file_handler()

    def setup_file_handler(self):
        self._fpath.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(self._fpath)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)


    def get_git_user_name(self):
        try:
            result = subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "Unknown"

    def log(self, message, level=logging.INFO):
        git_user_name = self.get_git_user_name()
        formatted_message = f"{git_user_name} - {message}"
        self._logger.log(level, formatted_message)
        