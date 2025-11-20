from pathlib import Path
import yaml
from TTS.api import TTS


class TTSGenerator:
    def __init__(self, yaml_path: str, model_name: str = "tts_models/en/ljspeech/vits--neon",
                 output_dir: str = "generated_voice"):
        self.yaml_path = Path(yaml_path)
        self.model_name = model_name
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True)

        print(f"Loading TTS model: {self.model_name} (first time may be slow)...")
        self.tts_engine = TTS(self.model_name)

    def load_commentary(self):
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def save_audio(self, text: str, index: int):
        out_file = self.output_dir / f"line_{index}.wav"

        self.tts_engine.tts_to_file(
            text=text,
            file_path=str(out_file)
        )

        print(f"Saved: {out_file}")

    def generate(self):
        data = self.load_commentary()

        for i, entry in enumerate(data):
            text = entry.get("commentary", "")

            if not text:
                continue

            print(f"[{i}] {text}")
            self.save_audio(text, i)
