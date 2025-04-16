import os
import uuid
from typing import List

import stable_whisper
import torch
import torchaudio
import torchvision.io
import webvtt
import yt_dlp
from loguru import logger
from pydantic import BaseModel

# Configure logging
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
logger.add(f"{log_dir}/youtube_processing.log", rotation="100 MB")


class _DataElement(BaseModel):
    id: str
    video_url: str
    video_id: str
    thumbnail: str
    title: str
    description: str
    video_path: str
    audio_path: str
    subtitle_paths: List[str]
    segments: List[dict]


class _YtPostProcessor:
    def __init__(self):
        self._transcribe_model = stable_whisper.load_model('turbo')
        self._target_langs = ["ar", "en", "zh", "hi", "ml", "fr", "es", "ur",
                              "ja", "ko", "de", "it", "pt", "tr", "ru", "kk"]

    def transcribe(self, audio_path: str) -> tuple:
        result = self._transcribe_model.transcribe(audio_path)
        language = result.language
        trans_vtt_file = audio_path.replace('.wav', f'.{language}.vtt')
        result.to_srt_vtt(trans_vtt_file, vtt=True, word_level=False)
        return language, trans_vtt_file

    def segment_from_vtt(self, vtt_filepath: str, audio_filepath: str, video_filepath: str, lang: str) -> list:
        """
        Segment audio and video based on timestamps in a .vtt file.

        Args:
            vtt_filepath (str): Path to the .vtt subtitle file.
            audio_filepath (str): Path to the source audio file.
            video_filepath (str): Path to the source video file.
            lang (str): Language of the transcription.

        Returns:
            list: List of segment dicts with text, audio_path, and video_path.
        """
        segments = []

        # Read the .vtt file and extract the captions
        try:
            vtt = webvtt.read(vtt_filepath)
            waveform, sample_rate = torchaudio.load(audio_filepath)
            logger.info(f"Loaded VTT file: {vtt_filepath} with {len(vtt.captions)} captions")
        except Exception as e:
            logger.error(f"Failed to read VTT file {vtt_filepath}: {e}")
            return []

        # Iterate over the captions and segment the audio and video
        for idx, caption in enumerate(vtt.captions):
            start_sec = caption.start_in_seconds
            end_sec = caption.end_in_seconds
            text = caption.text

            # Generate indexed output paths for the audio and video segments
            audio_segment_path = f"{audio_filepath}_audio_{idx}.wav"
            video_segment_path = f"{audio_filepath}_clip_{idx}.mp4"

            # Segment the audio
            try:
                self.__segment_audio(waveform, sample_rate, audio_segment_path, start_sec, end_sec)
                logger.info(f"Segmented audio: {audio_segment_path} ({start_sec}-{end_sec}s)")
            except Exception as e:
                logger.error(f"Audio segmentation failed for segment {idx}: {e}")
                continue

            # Segment the video
            try:
                self.__segment_video(video_filepath, video_segment_path, start_sec, end_sec)
                logger.info(f"Segmented video: {video_segment_path} ({start_sec}-{end_sec}s)")
            except Exception as e:
                logger.error(f"Video segmentation failed for segment {idx}: {e}")
                continue

            segments.append({
                "index": idx,
                "text": text,
                "audio_path": audio_segment_path,
                "video_path": video_segment_path,
                "lang": lang
            })

        return segments

    @staticmethod
    def __segment_audio(waveform: torch.Tensor, sample_rate: int, output_filepath: str, start_sec: float,
                        end_sec: float):
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        waveform_segment = waveform[:, start_sample:end_sample]
        torchaudio.save(output_filepath, waveform_segment, sample_rate)

    @staticmethod
    def __segment_video(input_filepath: str, output_filepath: str, start_sec: float, end_sec: float):
        """
        Segments a video file from start_sec to end_sec and writes the segment to output_filepath.

        Args:
            input_filepath (str): Path to the input video file.
            output_filepath (str): Path to save the segmented video.
            start_sec (float): Start time in seconds for the segment.
            end_sec (float): End time in seconds for the segment.
        """
        # Read the video segment from the input file
        video, _, info = torchvision.io.read_video(input_filepath, start_pts=start_sec, end_pts=end_sec, pts_unit="sec")

        # Extract frames per second (fps) from video info
        fps = info["video_fps"]

        # Write the segmented video to the output file
        torchvision.io.write_video(output_filepath, video, fps)


class YtProcessor:
    def __init__(self, output_dir: str = "resources", version: str = "v1"):
        self.yt_post_processor = _YtPostProcessor()
        self.target_langs = ["ar", "en", "zh", "hi", "ml", "fr", "es", "ur",
                             "ja", "ko", "de", "it", "pt", "tr", "ru", "kk"]
        self.version = version
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def __video_id(video_url: str) -> str:
        """
        Extract the YouTube video ID from a URL.

        Args:
            video_url (str): The YouTube video URL.

        Returns:
            str: The YouTube video ID.
        """
        return video_url.split("v=")[1]

    def __dwl_options(self):
        """
        Generate download options for yt_dlp.

        Returns:
            dict: A dictionary of download options for yt_dlp.
        """
        # Template for output filenames
        out_tmpl = f'{self.output_dir}/media/{self.version}/%(id)s.%(ext)s'

        # yt_dlp configuration options
        return {
            'keepvideo': True,  # Do not delete the video file after extracting audio
            'geo_bypass': True,  # Bypass geographic restrictions
            # 'source_address': '0.0.0.0',  # Bind to this IP address
            'download_archive': f'{self.output_dir}/{self.version}/ytd_dwl.log',  # Log to track downloaded videos
            "outtmpl": out_tmpl,  # Filename template for output files
            'merge_output_format': 'mp4',  # Output format for merged files
            "format": "bestvideo+bestaudio/best",  # Best quality video and audio
            'audio_format': 'wav',  # Output audio format
            'audio_quality': 0,  # Best quality audio
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',  # Extract audio using FFmpeg
                'preferredcodec': 'wav',
                'preferredquality': '192',  # Preferred audio quality
            }],
            "writesubtitles": True,  # Download subtitles
            "subtitleslangs": self.target_langs,  # Languages for subtitles
            "writeautomaticsub": False,  # Do not write automatic subtitles
            "writethumbnail": True,  # Download thumbnail images
            "writedescription": True,  # Download video description
            "writeinfojson": True,  # Download video metadata in JSON format
            "clean_infojson": True,  # Clean up the JSON metadata
            "getcomments": True,  # Download video comments
            "sleep_interval": 1,  # Sleep interval between downloads
        }

    def process_youtube(self, video_url: str) -> _DataElement:

        video_id = self.__video_id(video_url)
        media_path = f"{self.output_dir}/media/{self.version}/{video_id}"
        audio_fp = f"{media_path}.wav"
        video_fp = f"{media_path}.mp4"
        image_fp = f"{media_path}.jpg"
        desc_fp = f"{media_path}.description"
        subtitle_fp_template = f"{media_path}" + ".{}.vtt"

        # Configure logging
        logger.info(f"Processing video: {video_url}")

        # Download video and manual subtitles
        ydl_opts = self.__dwl_options()
        title = "Untitled"

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                title = info.get("title", "Untitled")
                logger.info(f"Downloaded video: {title}")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return []

        # Collect subtitles and segments
        segments_dict = {}
        subtitle_files = []

        # Check for downloaded subtitles
        any_subtitles = False
        for sub_lang in self.target_langs:
            sub_file = subtitle_fp_template.format(sub_lang)
            if os.path.exists(sub_file):
                any_subtitles = True
                segments = self.yt_post_processor.segment_from_vtt(vtt_filepath=sub_file,
                                                                   audio_filepath=audio_fp,
                                                                   video_filepath=video_fp,
                                                                   lang=sub_lang)
                segments_dict[sub_lang] = segments
                subtitle_files.append(sub_file)

                logger.info(f"Processed {len(segments)} segments for lang: {sub_lang}")

        # If no subtitles, generate VTT with stable-whisper
        if not any_subtitles:
            logger.info("No manual subtitles found, generating VTT with stable-whisper")
            try:
                language, trans_vtt_file = self.yt_post_processor.transcribe(audio_path=audio_fp)
                segments = self.yt_post_processor.segment_from_vtt(vtt_filepath=trans_vtt_file,
                                                                   audio_filepath=audio_fp,
                                                                   video_filepath=video_fp,
                                                                   lang=language)

                subtitle_files.append(trans_vtt_file)

                segments_dict[language] = segments

                logger.info(f"Processed transcribed {len(segments)} segments for lang: {language}")
            except Exception as e:
                logger.error(f"Stable-whisper transcription failed: {e}")

        logger.info(f"Processed {len(segments_dict)} language segments, languages: {segments_dict.keys()}.")

        return _DataElement(
            id=str(uuid.uuid4()),
            video_url=video_url,
            video_id=video_id,
            thumbnail=image_fp,
            title=title,
            description=desc_fp,
            video_path=video_fp,
            audio_path=audio_fp,
            subtitle_paths=subtitle_files,
            segments=segments_dict
        )
