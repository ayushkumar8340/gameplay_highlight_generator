from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1].parent
YAML_PATH = REPO_ROOT / "saved" / "gameplay_commentary.yaml"

from processing.caption_gen import TTSGenerator


def main():

    yaml_path = YAML_PATH

    tts = TTSGenerator(yaml_path)
    tts.generate()


if __name__ == "__main__":
    main()
