from pathlib import Path
from processing.editing_module import HighlightVideoCreator  


if __name__ == "__main__":
    
    timestamps = ["0:30", "1:10", "1:40"]

    video_path = "/home/heavya/github_akp/gameplay_highlight_generator/vids/1.MP4"
    audio_dir = "/home/heavya/github_akp/gameplay_highlight_generator/generated_voice" 

    output_path = Path("/home/heavya/github_akp/gameplay_highlight_generator/highlights_output.mp4")

    creator = HighlightVideoCreator(
        video_path=video_path,
        timestamps=timestamps,
        audio_dir=audio_dir,
        buffer_seconds=5,
        output_path=output_path,
    )

    creator.create()
