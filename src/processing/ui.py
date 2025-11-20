import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
from processing.processing_thread import run_entire_pipeline
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

IMG_HEIGHT = 450
IMG_WIDTH = 900

class HighlightUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gameplay Highlight Generator")
        self.root.geometry("1000x850")
        self._continue_flag = tk.BooleanVar(value=False)

        self.video_path = None

        # LOAD BUTTON
        self.btn_load = tk.Button(root, text="Load Video", command=self.load_video, width=20)
        self.btn_load.pack(pady=10)

        # SELECTED FILE LABEL
        self.lbl_video = tk.Label(root, text="No video selected.")
        self.lbl_video.pack()

        # GENERATE BUTTON (hidden initially)
        self.btn_generate = tk.Button(root, text="Generate Highlights", command=self.start_pipeline, width=20)
        self.btn_generate.pack(pady=10)
        self.btn_generate.pack_forget()

        # STATUS BOX
        self.status_box = tk.Text(root, height=12, width=60, state=tk.DISABLED)
        self.status_box.pack(pady=10)

        # IMAGE PREVIEW AREA
        self.image_label = tk.Label(root, bg="black", width=IMG_WIDTH, height=IMG_HEIGHT)
        self.image_label.pack(pady=10)

        # PROGRESS BAR (Windows-like green bar)
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            "Green.Horizontal.TProgressbar",
            troughcolor="#e0e0e0",
            background="#4caf50",   # Green fill
            thickness=20
        )

        self.progress_bar = ttk.Progressbar(
            root,
            style="Green.Horizontal.TProgressbar",
            orient="horizontal",
            mode="determinate",
            length=400
        )
        self.progress_bar.pack(pady=10)
        self.update_image(np.full((IMG_HEIGHT, IMG_WIDTH, 3), 255, dtype=np.uint8))

    def set_progress(self, value: int):
        self.progress_bar["value"] = value
        self.root.update_idletasks()

    def update_image(self, frame):
        """Update the center image box with a new OpenCV frame."""
        if frame is None:
            return

        # Convert OpenCV BGR → RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL image
        pil_img = Image.fromarray(img_rgb)

        # Resize to fit 400x225 box while keeping aspect ratio
        pil_img = pil_img.resize((IMG_WIDTH, IMG_HEIGHT))

        # Convert to Tkinter image
        self.tk_image = ImageTk.PhotoImage(image=pil_img)

        # Update label
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image  # prevent GC


    def wait_for_user_to_continue(self):
        top = tk.Toplevel(self.root)
        top.title("Continue")
        top.geometry("800x120")
        top.grab_set()

        label = tk.Label(top, text="Press Continue when you have copied the gameplay_commentary.yaml to the saved folder.")
        label.pack(pady=10)

        def do_continue():
            self._continue_flag.set(True)
            top.destroy()

        btn_continue = tk.Button(top, text="Continue", command=do_continue)
        btn_continue.pack(pady=10)

        self.root.wait_window(top)
    
    # ---------------------------------------------------------
    def load_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if filename:
            self.video_path = filename
            self.lbl_video.config(text=f"Selected: {filename}")
            self.btn_generate.pack()

    # ---------------------------------------------------------
    def start_pipeline(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please load a video first.")
            return

        self.log("Starting processing…\n")
        self.btn_generate.config(state=tk.DISABLED)

        thread = threading.Thread(target=self.run_pipeline_thread)
        thread.daemon = True
        thread.start()

    # ---------------------------------------------------------
    def run_pipeline_thread(self):
        try:
            output_file = run_entire_pipeline(self.video_path, self.log,self.set_progress,self,self.update_image)
            messagebox.showinfo("Finished", f"Processing complete!\nOutput saved at:\n{output_file}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.btn_generate.config(state=tk.NORMAL)

    def log(self, text):
        self.status_box.config(state=tk.NORMAL)
        self.status_box.insert(tk.END, text + "\n")
        self.status_box.config(state=tk.DISABLED)
        self.status_box.see(tk.END)


