import time
import subprocess
from pathlib import Path
import pyautogui


class ChatGPTImageUploader:
    def __init__(
        self,
        images_folder: str,
        chatgpt_url: str = "https://chatgpt.com/",
        chrome_window_title: str = "ChatGPT - Google Chrome",
        action_delay: float = 1.0,
        alt_tab_count: int = 2,
        post_upload_wait: float = 3.0,
        response_wait: float = 4.0,
        prompt_text: str = (
            "assume to be a commentator and give a short 3-4 word commentary "
            "for the gameplay screenshot and make sure to be energetic and make a donwloadable "
            "yaml file with only the commentary output and image name."
        ),
    ):
        self.images_folder = str(Path(images_folder).expanduser().resolve())
        self.chatgpt_url = chatgpt_url
        self.chrome_window_title = chrome_window_title
        self.action_delay = action_delay
        self.alt_tab_count = alt_tab_count
        self.post_upload_wait = post_upload_wait
        self.response_wait = response_wait
        self.prompt_text = prompt_text

        pyautogui.PAUSE = self.action_delay

    def _focus_chrome_window(self):
        ids = subprocess.check_output(
            ["xdotool", "search", "--onlyvisible", "--name", self.chrome_window_title]
        ).decode().strip().splitlines()
        if not ids:
            raise RuntimeError(f"No window matches: {self.chrome_window_title}")
        subprocess.check_call(["xdotool", "windowactivate", "--sync", ids[0]])
        time.sleep(0.6)

    @staticmethod
    def _focus_page_body():
        pyautogui.hotkey("ctrl", "l")
        time.sleep(0.2)
        pyautogui.press("enter")
        time.sleep(0.4)

    def run(self):
        folder_path = Path(self.images_folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_folder}")

        subprocess.Popen(["google-chrome", "--new-window", self.chatgpt_url])
        time.sleep(10)

        self._focus_chrome_window()
        self._focus_page_body()

        pyautogui.hotkey("ctrl", "u")
        time.sleep(1.5)

        for _ in range(self.alt_tab_count):
            pyautogui.hotkey("alt", "tab")
            time.sleep(0.2)

        time.sleep(1)

        pyautogui.write(self.images_folder, interval=0.01)
        pyautogui.press("enter")
        pyautogui.hotkey("ctrl", "a")

        # Navigate dialog controls; keep as-is to match your environment
        for _ in range(5):
            pyautogui.press("tab")

        pyautogui.press("enter")

        time.sleep(self.post_upload_wait)

        pyautogui.write(self.prompt_text, interval=0.01)
        pyautogui.press("enter")

        time.sleep(self.response_wait)
