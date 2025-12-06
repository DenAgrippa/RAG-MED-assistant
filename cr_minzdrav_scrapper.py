# from playwright.sync_api import sync_playwright
# from bs4 import BeautifulSoup
# import sys

# def get_page_content(link):
#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=True)
#         page = browser.new_page()
#         page.set_viewport_size({"width": 476, "height": 1003})
#         page.goto(link, wait_until="networkidle")
# #        page.wait_for_selector("#app > div > div > main > div > div > div > div.d-flex.flex-column.mt-4.mb-4 > div > div > div.v-card-text", timeout=10000)
#         page_content = page.content()
#         browser.close()
#     return page_content

# def save_page_file(page_content, file_name = "page"):
#     with open(f"{file_name}.html", "w", encoding="utf-8") as f:
#         f.write(page_content)

# def extract_text(page_content):
#     page_content_soup = BeautifulSoup(page_content, "html.parser")
# #    page_content_soup.img.decompose()
#     return page_content_soup.get_text()

# def save_text_file(page_text, file_name = "text"):
#     with open(f"{file_name}.txt", "w", encoding="utf-8") as f:
#         f.write(page_text)

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         link = sys.argv[1]
#         page_content = get_page_content(link)

# page_content = get_page_content("https://cr.minzdrav.gov.ru/view-cr/946_1")
# # save_page_file(page_content)
# page_text = extract_text(page_content)
# save_text_file(page_text)


from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import sys

def get_page_content(link):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1920, "height": 1080})
        page.goto(link, wait_until="networkidle")
        page.wait_for_timeout(250)
        page_content = page.content()
        browser.close()
    return page_content

def save_page_file(page_content, file_name="page"):
    with open(f"{file_name}.html", "w", encoding="utf-8") as f:
        f.write(page_content)

def extract_html_content(page_content):
    """Извлекает HTML-контент для дальнейшего парсинга"""
    return page_content

def save_html_file(page_html, file_name="page"):
    """Сохраняет HTML-контент в файл"""
    with open(f"{file_name}.html", "w", encoding="utf-8") as f:
        f.write(page_html)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        link = sys.argv[1]
        page_content = get_page_content(link)
        # Сохраняем HTML
        save_html_file(page_content, "page")
        print(f"✅ HTML сохранён в page.html")
    else:
        # Для теста
        link = "https://cr.minzdrav.gov.ru/view-cr/946_1"
        page_content = get_page_content(link)
        save_html_file(page_content, "page")
        print(f"✅ HTML сохранён в page.html")