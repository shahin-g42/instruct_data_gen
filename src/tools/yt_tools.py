# import os
#
# import torchaudio
# import torchvision.io
# import yt_dlp
# from loguru import logger
# import stable_whisper
#
#
# def search_youtube(query: str, lang: str = 'en', max_results: int = 30) -> list:
#     """
#     Search YouTube for multiple videos based on a query and language, returning a list of URLs.
#
#     Args:
#         query (str): The search query.
#         lang (str): The preferred language for the search results (default is English).
#         max_results (int): The maximum number of results to return (default is 30).
#
#     Returns:
#         list: A list of URLs of the search results.
#     """
#     ydl_opts = {
#         "quiet": True,
#         "no_warnings": True,
#         "extract_flat": False,
#         "language": lang,
#     }
#
#     # Use the ytsearch special query to search for videos
#     query = f"ytsearch{max_results}:{query}"
#     urls = []
#     try:
#         # Use yt_dlp to extract the video URLs from the search results
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             result = ydl.extract_info(query, download=False)
#             if "entries" in result and result["entries"]:
#                 urls = [f"https://www.youtube.com/watch?v={entry['id']}" for entry in result["entries"]]
#         return urls
#     except Exception as e:
#         logger.info(f"Error in search_youtube: {e}")
#         return urls
#
#
# def download_youtube(video_url: str, output_dir: str):
#     out_tmpl = f'{output_dir}/media/v33/%(id)s.%(ext)s'
#     target_langs = ["ar", "en", "zh", "hi", "ml", "fr", "es", "ur", "ja", "ko", "de", "it", "pt", "tr", "ru", "kk"]
#
#     ydl_opts = {
#         'keepvideo': True,
#         'geo_bypass': True,
#         'source_address': '0.0.0.0',
#         'download_archive': f'{output_dir}/ytd_dwl.log',
#         "outtmpl": out_tmpl,
#         'merge_output_format': 'mp4',
#         "format": "bestvideo+bestaudio/best",
#         'audio_format': 'wav',
#         'audio_quality': 0,
#         'postprocessors': [{
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': 'wav',
#             'preferredquality': '192',
#         }],
#         "writesubtitles": True,
#         "subtitleslangs": target_langs,
#         "writeautomaticsub": False,
#         "writethumbnail": True,
#         "write_all_thumbnails": True,
#         "writedescription": True,
#         "writeinfojson": True,
#         "clean_infojson": True,
#         "getcomments": True,
#         "sleep_interval": 1,
#     }
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(video_url, download=True)
#         # logger.info(info)
#
#
# def process_youtube(video_url: str, output_dir: str, lang: str) -> dict:
#     """Process a YouTube video: extract subtitles or transcribe, align audio/video chunks, save thumbnail."""
#     video_id = video_url.split("v=")[1]
#     audio_path = f"{output_dir}/media/v33/{video_id}.wav"
#     image_path = f"{output_dir}/media/v33/{video_id}_thumb.jpg"
#     video_path = f"{output_dir}/media/v33/{video_id}_clip.mp4"
#     out_tmpl = f'{output_dir}/media/v33/%(id)s.%(ext)s'
#
#     # Target languages
#     target_langs = ["ar", "en", "zh", "hi", "ml", "fr", "es"]
#
#     # Configure logging
#     logger.add(f"{output_dir}/youtube_processing.log", rotation="1 MB")
#     logger.info(f"Processing video: {video_url} (primary lang: {lang})")
#
#     # Download video and manual subtitles
#     ydl_opts = {
#         'keepvideo': True,
#         'geo_bypass': True,
#         'source_address': '0.0.0.0',
#         'download_archive': f'{output_dir}/ytd_dwl.log',
#         "outtmpl": out_tmpl,
#         'merge_output_format': 'mp4',
#         "format": "bestvideo+bestaudio/best",
#         'audio_format': 'wav',
#         'audio_quality': 0,
#         'postprocessors': [{
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': 'wav',
#             'preferredquality': '192',
#         }],
#         "writesubs": True,
#         "subtitleslangs": target_langs,
#         "skip_download_automatic_subtitles": True,
#         "sleep_interval": 1,
#     }
#
#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             info = ydl.extract_info(video_url, download=True)
#             title = info.get("title", "Untitled")
#             logger.info(f"Downloaded video: {title}")
#     except Exception as e:
#         logger.error(f"Download failed: {e}")
#         return {}
#
#     # Collect subtitles
#     subtitles = {}
#     for sub_lang in target_langs:
#         subtitle_file = f"{output_dir}/temp.{sub_lang}.vtt"
#         if os.path.exists(subtitle_file):
#             with open(subtitle_file, "r", encoding="utf-8") as f:
#                 text = " ".join([line.strip() for line in f.readlines() if "-->" not in line and line.strip()][:10])
#             subtitles[sub_lang] = text
#             logger.info(f"Found subtitles for {sub_lang}")
#             os.remove(subtitle_file)
#
#     # Primary text
#     if lang in subtitles:
#         text = subtitles[lang]
#         logger.info(f"Using subtitles for primary lang {lang}")
#     elif subtitles:
#         text = list(subtitles.values())[0]
#         logger.info(f"Using first available subtitle: {list(subtitles.keys())[0]}")
#     else:
#         logger.info("No manual subtitles, transcribing with Whisper...")
#         try:
#             model = stable_whisper.load_model('turbo')
#             result = model.transcribe(audio_path, language=lang)
#             text = " ".join([seg.text for seg in result[:10]])
#             logger.info(f"Transcription successful: {text[:50]}...")
#         except Exception as e:
#             logger.warning(f"Transcription failed: {e}, falling back to title")
#             text = title
#
#     # Translation details
#     translations = {k: v for k, v in subtitles.items() if k != lang} if lang in subtitles else subtitles
#
#     # Audio segmentation with torchaudio
#     audio_duration = 10  # seconds
#     try:
#         waveform, sample_rate = torchaudio.load(audio_path)
#         segment_length = audio_duration * sample_rate
#         waveform_segment = waveform[:, :segment_length]
#         torchaudio.save(audio_path, waveform_segment, sample_rate)
#         logger.info(f"Audio segmented: {audio_path}")
#     except Exception as e:
#         logger.error(f"Audio segmentation failed: {e}")
#         return {}
#
#     # Video segmentation with torchvision (aligned to audio)
#     try:
#         video, _, info = torchvision.io.read_video(temp_video, pts_unit="sec")
#         fps = info["video_fps"]
#         num_frames = int(audio_duration * fps)  # Align with audio duration
#         video_segment = video[:num_frames]  # First 10s
#         torchvision.io.write_video(video_path, video_segment, fps)
#         logger.info(f"Video clipped: {video_path}")
#     except Exception as e:
#         logger.error(f"Video clipping failed: {e}")
#         return {}
#
#     # Thumbnail
#     try:
#         os.system(f"yt-dlp --get-thumbnail {video_url} --skip-download -o {image_path}")
#         if os.path.exists(image_path):
#             logger.info(f"Thumbnail saved: {image_path}")
#         else:
#             logger.warning("Thumbnail extraction failed")
#     except Exception as e:
#         logger.error(f"Thumbnail extraction failed: {e}")
#
#     # Cleanup
#     os.remove(temp_video)
#
#     return {
#         "text": text,
#         "translations": translations,
#         "audio_path": audio_path,
#         "image_path": image_path,
#         "video_path": video_path
#     }


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
from torchcodec.decoders import VideoDecoder

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

    def segment_from_vtt(self, video_id: str,
                         vtt_filepath: str,
                         audio_filepath: str,
                         video_filepath: str,
                         lang: str) -> list:
        """
        Segment audio and video based on timestamps in a .vtt file.

        Args:
            video_id (str): Identifier for the video being processed.
            vtt_filepath (str): Path to the .vtt subtitle file.
            audio_filepath (str): Path to the source audio file.
            video_filepath (str): Path to the source video file.
            lang (str): Language of the transcription.

        Returns:
            list: List of segment dicts with index, text, audio_path, video_path, and lang.
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
            audio_segment_path = f"{video_id}_audio_{idx}.wav"
            video_segment_path = f"{video_id}_clip_{idx}.mp4"

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
                raise e

            # Append segment details to the list
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
        video, audio, info = torchvision.io.read_video(input_filepath,
                                                       start_pts=start_sec,
                                                       end_pts=end_sec,
                                                       pts_unit="sec")

        # Extract frames per second (fps) from video info
        vid_fps = info["video_fps"]
        aud_sr = info["audio_fps"]

        # Write the segmented video to the output file
        torchvision.io.write_video(filename=output_filepath,
                                   video_array=video,
                                   fps=vid_fps,
                                   video_codec="h264",
                                   audio_array=audio,
                                   audio_fps=aud_sr,
                                   audio_codec="aac")
        # torchvision.io.write_video(output_filepath, video, fps, video_codec="h264")


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

    def __download_options(self):
        """
        Generate download options for yt_dlp.

        Returns:
            dict: A dictionary of download options for yt_dlp.
        """
        # Template for output filenames
        out_tmpl = f'{self.output_dir}/media/{self.version}/%(id)s.%(ext)s'
        _log_dir = os.path.join(self.output_dir, f"yt_dlp/log/{self.version}")
        os.makedirs(_log_dir, exist_ok=True)

        # yt_dlp configuration options
        return {
            'keepvideo': True,  # Do not delete the video file after extracting audio
            'geo_bypass': True,  # Bypass geographic restrictions
            # 'source_address': '0.0.0.0',  # Bind to this IP address
            'cookiefile': 'cookies.txt',
            'download_archive': f'{_log_dir}/ytd_dwl.log',  # Log to track downloaded videos
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
            "getcomments": False,  # Download video comments
            "sleep_interval": 1,  # Sleep interval between downloads
            "sleep_interval_requests": 2,
            "sleep_interval_subtitles": 2,
        }

    @staticmethod
    def __search_options(lang: str = "ar"):
        return {
            'geo_bypass': True,
            'cookiefile': 'cookies.txt',
            "no_warnings": True,
            "extract_flat": False,
            "language": lang,
            "sleep_interval": 1,  # Sleep interval between downloads
            "sleep_interval_requests": 2,
            "sleep_interval_subtitles": 2,
        }

    def search_youtube(self, query: str, lang: str = 'en', max_results: int = 30) -> list:
        """
        Search YouTube for multiple videos based on a query and language, returning a list of URLs.

        Args:
            query (str): The search query.
            lang (str): The preferred language for the search results (default is English).
            max_results (int): The maximum number of results to return (default is 30).

        Returns:
            list: A list of URLs of the search results.
        """
        ydl_opts = self.__search_options(lang)

        # Use the ytsearch special query to search for videos
        query = f"ytsearch{max_results}:{query}"
        urls = []
        try:
            logger.info(f"Search for videos, query: {query}, lang: {lang}, max_results: {max_results}")
            # Use yt_dlp to extract the video URLs from the search results
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(query, download=False)
                if "entries" in result and result["entries"]:
                    urls = [f"https://www.youtube.com/watch?v={entry['id']}" for entry in result["entries"]]
        except Exception as e:
            logger.info(f"Error in search_youtube: {e}")

        logger.info(f"Found {len(urls)} videos")
        return urls

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
        ydl_opts = self.__download_options()
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
                segments = self.yt_post_processor.segment_from_vtt(
                    video_id=video_id,
                    vtt_filepath=sub_file,
                    audio_filepath=audio_fp,
                    video_filepath=video_fp,
                    lang=sub_lang
                )

                segments_dict[sub_lang] = segments
                subtitle_files.append(sub_file)

                logger.info(f"Processed {len(segments)} segments for lang: {sub_lang}")

        # If no subtitles, generate VTT with stable-whisper
        if not any_subtitles:
            logger.info("No manual subtitles found, generating VTT with stable-whisper")
            try:
                language, trans_vtt_file = self.yt_post_processor.transcribe(audio_path=audio_fp)
                segments = self.yt_post_processor.segment_from_vtt(video_id=video_id,
                                                                   vtt_filepath=trans_vtt_file,
                                                                   audio_filepath=audio_fp,
                                                                   video_filepath=video_fp,
                                                                   lang=language)

                subtitle_files.append(trans_vtt_file)

                segments_dict[language] = segments

                logger.info(f"Processed transcribed {len(segments)} segments for lang: {language}")
            except Exception as e:
                logger.error(f"Stable-whisper transcription failed: {e}")

        logger.info(f"Processed {len(segments_dict)} language segments, languages: {segments_dict.keys()}.")

        # Get description
        with open(desc_fp, "r", encoding="utf-8") as f:
            description = f.read()
            f.flush()
            f.close()
            logger.debug(f"Description {description}.")

        return _DataElement(
            id=str(uuid.uuid4()),
            video_url=video_url,
            video_id=video_id,
            thumbnail=image_fp,
            title=title,
            description=description,
            video_path=video_fp,
            audio_path=audio_fp,
            subtitle_paths=subtitle_files,
            segments=segments_dict
        )
