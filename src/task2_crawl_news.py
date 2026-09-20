"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""
import asyncio
from datetime import datetime
import json
from pathlib import Path
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://tramdoc.vn/tin-tuc/angela-duckworth-no-luc-thoi-chua-du-moi-truong-cung-quyet-dinh-thanh-cong-nNOn3W.html",
    "https://tramdoc.vn/tin-tuc/hoi-ky-cua-hy-vong-khi-mot-co-gai-phai-cuu-lay-chinh-minh-de-buoc-toi-vi-sao-nDaANW.html",
    "https://tramdoc.vn/tin-tuc/thuong-thu-mot-trong-nhung-bo-kinh-dien-quan-trong-bac-nhat-cua-nho-hoc-nlKMQW.html",
    "https://tramdoc.vn/tin-tuc/la-thu-tai-sinh-khi-mot-nguoi-chiu-ngoi-xuong-lang-nghe-noi-dau-cua-nguoi-khac-nw8MLW.html",
    "https://tramdoc.vn/tin-tuc/di-tim-hien-tai-khoa-hoc-ve-khoanh-khac-chung-ta-dang-song-nr8M9W.html",
]


async def crawl_article(crawler: AsyncWebCrawler, url: str) -> dict:
    run_config = CrawlerRunConfig(
        word_count_threshold=10, 
        exclude_external_links=True, 
    )

    result = await crawler.arun(url=url, config=run_config)

    title = result.metadata.get("title") or result.metadata.get(
        "og:title", "Unknown"
    )

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": result.markdown,
    }


async def crawl_all() -> None:

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    browser_config = BrowserConfig(
        browser_type="chromium",
        chrome_channel="chrome",
        headless=False,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for index, url in enumerate(ARTICLE_URLS, 1):
            try:
                print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
                article = await crawl_article(crawler, url)

                output = DATA_DIR / f"article_{index:02d}.json"
                output.write_text(
                    json.dumps(article, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f" Saved: {output}")
            except Exception as error:
                print(f" Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())