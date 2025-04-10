from src import search_youtube, process_youtube

urls = search_youtube("home", max_results=5)

for url in urls:
    res = process_youtube(url, "res", "en")
