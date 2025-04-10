import os

import torchaudio
import torchvision.io
import yt_dlp
from loguru import logger
import stable_whisper


def search_youtube(query: str, lang: str = 'en', max_results: int = 30) -> list:
    """
    Search YouTube for multiple videos based on a query and language, returning a list of URLs.

    Args:
        query (str): The search query.
        lang (str): The preferred language for the search results (default is English).
        max_results (int): The maximum number of results to return (default is 30).

    Returns:
        list: A list of URLs of the search results.
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "language": lang,
    }

    # Use the ytsearch special query to search for videos
    query = f"ytsearch{max_results}:{query}"
    urls = []
    try:
        # Use yt_dlp to extract the video URLs from the search results
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(query, download=False)
            if "entries" in result and result["entries"]:
                urls = [f"https://www.youtube.com/watch?v={entry['id']}" for entry in result["entries"]]
        return urls
    except Exception as e:
        logger.info(f"Error in search_youtube: {e}")
        return urls


def process_youtube(video_url: str, output_dir: str, lang: str) -> dict:
    """Process a YouTube video: extract subtitles or transcribe, align audio/video chunks, save thumbnail."""
    video_id = video_url.split("v=")[1]
    audio_path = f"{output_dir}/{video_id}_audio.wav"
    image_path = f"{output_dir}/{video_id}_thumb.jpg"
    video_path = f"{output_dir}/{video_id}_clip.mp4"
    temp_video = f"{output_dir}/temp.mp4"
    out_tmpl = f'{output_dir}/wav/v33/%(id)s.%(ext)s'

    # Target languages
    target_langs = ["ar", "en", "zh", "hi", "ml", "fr", "es"]

    # Configure logging
    logger.add(f"{output_dir}/youtube_processing.log", rotation="1 MB")
    logger.info(f"Processing video: {video_url} (primary lang: {lang})")

    # Download video and manual subtitles
    ydl_opts = {
        'netrc': True,
        'force_ipv6': True,
        "format": "bestvideo+bestaudio/best",
        'ignoreerrors': True,
        'geo_bypass': True,
        'audio_format': 'wav',
        'audio_quality': 0,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'cookiefile': 'cookies.txt',
        "outtmpl": out_tmpl,
        "writesubs": True,
        "subtitleslangs": target_langs,
        "skip_download_automatic_subtitles": True,
        "sleep_interval_requests": 1,
        "sleep_interval": 1,
        "sleep_interval_subtitles": 1,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            title = info.get("title", "Untitled")
            logger.info(f"Downloaded video: {title}")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return {}

    # Collect subtitles
    subtitles = {}
    for sub_lang in target_langs:
        subtitle_file = f"{output_dir}/temp.{sub_lang}.vtt"
        if os.path.exists(subtitle_file):
            with open(subtitle_file, "r", encoding="utf-8") as f:
                text = " ".join([line.strip() for line in f.readlines() if "-->" not in line and line.strip()][:10])
            subtitles[sub_lang] = text
            logger.info(f"Found subtitles for {sub_lang}")
            os.remove(subtitle_file)

    # Primary text
    if lang in subtitles:
        text = subtitles[lang]
        logger.info(f"Using subtitles for primary lang {lang}")
    elif subtitles:
        text = list(subtitles.values())[0]
        logger.info(f"Using first available subtitle: {list(subtitles.keys())[0]}")
    else:
        logger.info("No manual subtitles, transcribing with Whisper Turbo")
        try:
            model = stable_whisper.load_model('turbo')
            result = model.transcribe(temp_video, language=lang)
            text = " ".join([seg.text for seg in result[:10]])
            logger.info(f"Transcription successful: {text[:50]}...")
        except Exception as e:
            logger.warning(f"Transcription failed: {e}, falling back to title")
            text = title

    # Translation details
    translations = {k: v for k, v in subtitles.items() if k != lang} if lang in subtitles else subtitles

    # Audio segmentation with torchaudio
    audio_duration = 10  # seconds
    try:
        waveform, sample_rate = torchaudio.load(temp_video)
        segment_length = audio_duration * sample_rate
        waveform_segment = waveform[:, :segment_length]
        torchaudio.save(audio_path, waveform_segment, sample_rate)
        logger.info(f"Audio segmented: {audio_path}")
    except Exception as e:
        logger.error(f"Audio segmentation failed: {e}")
        return {}

    # Video segmentation with torchvision (aligned to audio)
    try:
        video, _, info = torchvision.io.read_video(temp_video, pts_unit="sec")
        fps = info["video_fps"]
        num_frames = int(audio_duration * fps)  # Align with audio duration
        video_segment = video[:num_frames]  # First 10s
        torchvision.io.write_video(video_path, video_segment, fps)
        logger.info(f"Video clipped: {video_path}")
    except Exception as e:
        logger.error(f"Video clipping failed: {e}")
        return {}

    # Thumbnail
    try:
        os.system(f"yt-dlp --get-thumbnail {video_url} --skip-download -o {image_path}")
        if os.path.exists(image_path):
            logger.info(f"Thumbnail saved: {image_path}")
        else:
            logger.warning("Thumbnail extraction failed")
    except Exception as e:
        logger.error(f"Thumbnail extraction failed: {e}")

    # Cleanup
    os.remove(temp_video)

    return {
        "text": text,
        "translations": translations,
        "audio_path": audio_path,
        "image_path": image_path,
        "video_path": video_path
    }
