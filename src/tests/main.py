from processing.ui import HighlightUI
import threading
import sys
import tkinter as tk


if __name__ == "__main__":
    root = tk.Tk()
    app = HighlightUI(root)
    root.mainloop()