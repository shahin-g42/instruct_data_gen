from src.tools import search_youtube, process_youtube, download_youtube

# urls = search_youtube("charlie chaplin", max_results=15)

# for url in urls:
#     res = download_youtube(url, "res")

url = "https://www.youtube.com/watch?v=apbSsILLh28"
res = download_youtube(url, "res")