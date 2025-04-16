FROM huggingface/transformers-pytorch-gpu
RUN apt-get update -y
RUN python3 -m pip install -U torch torchaudio stable-ts webvtt-py yt-dlp loguru crawl4ai
RUN crawl4ai-setup
RUN crawl4ai-doctor