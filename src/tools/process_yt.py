import os

import stable_whisper
import torchaudio
import torchvision.io
import webvtt
import yt_dlp
from loguru import logger

# Configure logging
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
logger.add(f"{log_dir}/youtube_processing.log", rotation="100 MB")


class YtProcessor:
    def __init__(self):
        self._transcribe_model = stable_whisper.load_model('turbo')

    def transcribe(self, audio_path: str) -> tuple:
        result = self._transcribe_model.transcribe(audio_path)
        language = result.language
        trans_vtt_file = audio_path.replace('.wav', f'_{language}.vtt')
        result.to_srt_vtt(trans_vtt_file, vtt=True, word_level=False)
        return language, trans_vtt_file





def process_youtube(video_url: str, output_dir: str, lang: str) -> list:
    """Process a YouTube video: segment audio/video based on all .vtt subtitles or generate VTT with stable-whisper."""
    video_id = video_url.split("v=")[1]
    out_tmpl = f'{output_dir}/media/v33/%(id)s.%(ext)s'

    # Target languages
    target_langs = ["ar", "en", "zh", "hi", "ml", "fr", "es"]
    logger.info(f"Processing video: {video_url} (primary lang: {lang})")

    # Download video/audio and manual subtitles
    ydl_opts = {
        'keepvideo': True,
        'geo_bypass': True,
        'source_address': '0.0.0.0',
        'download_archive': f'{output_dir}/ytd_dwl.log',
        "outtmpl": out_tmpl,
        'merge_output_format': 'mp4',
        "format": "bestvideo+bestaudio/best",
        'audio_format': 'wav',
        'audio_quality': 0,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        "writesubs": True,
        "subtitleslangs": target_langs,
        "skip_download_automatic_subtitles": True,
        "sleep_interval": 1,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            title = info.get("title", "Untitled")
            logger.info(f"Downloaded video: {title}")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return []

    # Collect subtitles with webvtt-py
    subtitles = {}
    subtitle_file = f"{output_dir}/temp.{lang}.vtt"
    segments = []

    # Check for downloaded subtitles
    any_subtitles = False
    for sub_lang in target_langs:
        sub_file = f"{output_dir}/temp.{sub_lang}.vtt"
        if os.path.exists(sub_file):
            any_subtitles = True
            try:
                vtt = webvtt.read(sub_file)
                if sub_lang == lang:
                    # Primary language subtitles
                    for idx, caption in enumerate(vtt.captions):
                        segments.append({
                            "start_sec": caption.start_in_seconds,
                            "end_sec": caption.end_in_seconds,
                            "text": caption.text,
                            "lang": sub_lang
                        })
                    logger.info(f"Found {len(segments)} segments for primary lang {sub_lang}")
                else:
                    # Store translations
                    subtitles[sub_lang] = {idx: caption.text for idx, caption in enumerate(vtt.captions)}
                    logger.info(f"Found subtitles for {sub_lang}")
            except Exception as e:
                logger.error(f"Failed to parse VTT for {sub_lang}: {e}")
            os.remove(sub_file)

    # If no subtitles, generate VTT with stable-whisper
    if not segments:
        logger.info("No manual subtitles found, generating VTT with stable-whisper")
        try:
            model = stable_whisper.load_model('base')
            result = model.transcribe(temp_video, language=lang)
            vtt_file = f"{output_dir}/temp.{lang}.vtt"
            result.to_srt_vtt(vtt_file)  # Direct save as VTT
            vtt = webvtt.read(vtt_file)
            for caption in vtt.captions:
                segments.append({
                    "start_sec": caption.start_in_seconds,
                    "end_sec": caption.end_in_seconds,
                    "text": caption.text,
                    "lang": lang
                })
            logger.info(f"Generated {len(segments)} segments with stable-whisper for {lang}")
            os.remove(vtt_file)
        except Exception as e:
            logger.error(f"Stable-whisper transcription failed: {e}, using title with 10s chunk")
            segments = [{
                "start_sec": 0,
                "end_sec": 10,
                "text": title,
                "lang": lang
            }]

    # Process each segment
    output_segments = []
    for idx, segment in enumerate(segments):
        audio_path = f"{output_dir}/{video_id}_audio_{idx}.wav"
        image_path = f"{output_dir}/{video_id}_thumb_{idx}.jpg"
        video_path = f"{output_dir}/{video_id}_clip_{idx}.mp4"

        # Audio segmentation
        try:
            waveform, sample_rate = torchaudio.load(temp_video)
            start_sample = int(segment["start_sec"] * sample_rate)
            end_sample = int(segment["end_sec"] * sample_rate)
            waveform_segment = waveform[:, start_sample:end_sample]
            torchaudio.save(audio_path, waveform_segment, sample_rate)
            logger.info(f"Audio segmented: {audio_path} ({segment['start_sec']}-{segment['end_sec']}s)")
        except Exception as e:
            logger.error(f"Audio segmentation failed for segment {idx}: {e}")
            continue

        # Video segmentation
        try:
            video, _, info = torchvision.io.read_video(temp_video, start_pts=segment["start_sec"],
                                                       end_pts=segment["end_sec"], pts_unit="sec")
            fps = info["video_fps"]
            torchvision.io.write_video(video_path, video, fps)
            logger.info(f"Video clipped: {video_path} ({segment['start_sec']}-{segment['end_sec']}s)")
        except Exception as e:
            logger.error(f"Video clipping failed for segment {idx}: {e}")
            continue

        # Thumbnail
        try:
            os.system(f"yt-dlp --get-thumbnail {video_url} --skip-download -o {image_path}")
            if os.path.exists(image_path):
                logger.info(f"Thumbnail saved: {image_path}")
            else:
                logger.warning(f"Thumbnail extraction failed for segment {idx}")
        except Exception as e:
            logger.error(f"Thumbnail extraction failed for segment {idx}: {e}")

        # Compile segment data
        translations = {k: v.get(idx, "") for k, v in subtitles.items() if k != segment["lang"]}
        output_segments.append({
            "text": segment["text"],
            "translations": translations,
            "audio_path": audio_path,
            "image_path": image_path,
            "video_path": video_path
        })

    # Cleanup
    os.remove(temp_video)

    return output_segments
