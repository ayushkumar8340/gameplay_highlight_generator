# file: hype_line_llava15.py
import os, re, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

os.environ["TRANSFORMERS_NO_TF"] = "1"

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
IMAGE_PATH = "/mnt/data/test.png"

PROMPT = """You are an expert esports commentator for mobile FPS gameplay.
Write exactly ONE hype line (<= 90 chars), grounded in the HUD: kills (red),
health bar (bottom), and medals (top). Mention streaks/medals concisely when visible.
One sentence, no emojis/hashtags. Avoid inventing details.
If HP <25% per hints, note the danger.
"""

def clean(text):
    t = text.strip().splitlines()[0]
    t = re.sub(r"[#\uFE0F\u200D]+", "", t)
    t = re.sub(r"[\U0001F300-\U0001FAFF]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    m = re.search(r"([^.?!]*[.?!])", t)
    return (m.group(1) if m else t)[:90]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

proc = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
).to(device)

img = Image.open(IMAGE_PATH).convert("RGB")

msgs = [
    {"role":"system","content":[{"type":"text","text":PROMPT.strip()}]},
    {"role":"user","content":[
        {"type":"text","text":"Generate ONE hype line from this HUD."},
        {"type":"image","image":img},
    ]},
]

tmpl = proc.apply_chat_template(msgs, add_generation_prompt=True)
inputs = proc(text=tmpl, images=img, return_tensors="pt").to(device)

out = model.generate(**inputs, max_new_tokens=60, temperature=0.0, do_sample=False)
gen = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print(clean(gen))
