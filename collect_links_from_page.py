from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

def get_page(p, url):
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(url, wait_until="networkidle")
    return page

def get_links(page, list):
    pass

def display_next_page(page):
#    page.locator("i.mdi-chevron-right").click()
    page.get_by_label('Следующая страница').click()
    page.wait_for_timeout(1000)

def display_all_items(page):
    pass



url = 'https://cr.minzdrav.gov.ru/clin-rec/'
all_links = []
last_page = False

with sync_playwright() as p:
    page = get_page(p, url)
    get_links(page, all_links)
    display_next_page(page)
    last_page = page.locator('button[aria-label="Следующая страница"][aria-disabled="true"]').is_visible()
    page.screenshot(path='screen.png')
