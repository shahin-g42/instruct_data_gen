from yt_tools import search_youtube, process_youtube

urls = search_youtube("machupicchu", max_results=5)

res = process_youtube(urls[0], "res", "en")
