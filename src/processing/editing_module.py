from pathlib import Path

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout
from moviepy.video.fx.resize import resize

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip


class HighlightVideoCreator:


    def __init__(
        self,
        video_path: str | Path,
        timestamps: list[str],
        audio_dir: str | Path,
        buffer_l_seconds: int = 4,
        buffer_r_seconds: int = 1,
        output_path: str | Path = "final_highlight.mp4",
    ):
        self.video_path = Path(video_path)
        self.timestamps = timestamps
        self.audio_dir = Path(audio_dir)
        self.buffer_l_seconds = buffer_l_seconds
        self.buffer_r_seconds = buffer_r_seconds
        self.output_path = Path(output_path)

    @staticmethod
    def _to_seconds(t: str) -> int:
        parts = parts = t.split(":")
        parts = [float(p) for p in parts]
        if len(parts) == 2:
            m, s = parts
            return m * 60 + s
        elif len(parts) == 3:
            h, m, s = parts
            return h * 3600 + m * 60 + s
        elif len(parts) == 4:
            _, h, m, s = parts
            return h * 3600 + m * 60 + s
        else:
            raise ValueError("Timestamp format should be M:SS or H:MM:SS or D:H:MM:SS")

    def _collect_commentary_files(self) -> list[Path | None]:
        files: list[Path | None] = []

        for i in range(len(self.timestamps)):
            
            matches = sorted(self.audio_dir.glob(f"line_{i}.*"))
            if matches:
                files.append(matches[0])
            else:
                print(f"[WARN] No commentary file found for index {i} (expected line_{i}.*)")
                files.append(None)

        return files

    def create(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Input video not found: {self.video_path}")

        commentary_files = self._collect_commentary_files()

        print(f"[INFO] Loading video: {self.video_path}")
        video = VideoFileClip(str(self.video_path))
        duration = video.duration

        base_h, base_w = video.size

        subclips = []

        try:
            for i, ts in enumerate(self.timestamps):
                center = self._to_seconds(ts)
                start = max(center - self.buffer_l_seconds, 0)
                end = min(center + self.buffer_r_seconds, duration)

                print(f"[INFO] Creating subclip {i}: {start:.2f}s -> {end:.2f}s (ts={ts})")
                clip = video.subclip(start, end)

                clip = resize(clip, (base_w, base_h))

                comment_path = commentary_files[i]
                if comment_path is not None and comment_path.exists():
                    commentary = AudioFileClip(str(comment_path))

                    comment_start = self.buffer_l_seconds - 1
                    comment_start = max(0, min(comment_start, clip.duration))

                    max_comment_dur = max(0, clip.duration - comment_start)
                    commentary = commentary.subclip(0, min(commentary.duration, max_comment_dur))

                    commentary = commentary.set_start(comment_start)

                    if clip.audio is not None:
                        new_audio = CompositeAudioClip([clip.audio, commentary])
                    else:
                        new_audio = commentary

                    clip = clip.set_audio(new_audio)
                else:
                    print(f"[INFO] No commentary audio for clip {i}, keeping original audio.")

                clip = fadein(clip, 0.5)
                clip = fadeout(clip, 0.5)

                subclips.append(clip)

            if not subclips:
                raise RuntimeError("No subclips were created. Check timestamps and video duration.")

            print("[INFO] Concatenating subclips...")
            final = concatenate_videoclips(subclips, method="chain")

            print(f"[INFO] Writing output video to {self.output_path}")
            final.write_videofile(
                str(self.output_path),
                codec="libx264",
                audio_codec="aac",
                fps=video.fps,
            )

            final.close()

        finally:
            video.close()
            for c in subclips:
                c.close()

        print("[INFO] Highlight video creation complete.")
