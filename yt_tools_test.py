from src.tools import YtProcessor

ytp = YtProcessor(output_dir="res_test")

urls = ytp.search_youtube("dublin", max_results=15)

for url in urls:
    res = ytp.process_youtube(url)
    print(res)
