from playwright.sync_api import sync_playwright
import pandas as pd
import sqlite3 as sqlite

def get_page(p, url):
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(url, wait_until="networkidle")
    return page

def download_report(page, folder_path):
    with page.expect_download() as d:
        page.get_by_text("Получить отчет").click()
        download = d.value
        suggested_filename = download.suggested_filename
        save_path = f"{folder_path}/{suggested_filename}"
        download.save_as(save_path)
        return save_path

def get_df_with_links(report_path):
    df = pd.read_excel(report_path)
    df['HTML_link'] = "https://cr.minzdrav.gov.ru/view-cr/" + df['ID'].astype(str)
    return df

def write_to_excel(df, path):
    df.to_excel(f'{path}/Список КР со ссылками.xlsx', index=False)

def write_to_sqlite(df, db_path = "db/db.sqlite"):
    connector = sqlite.connect(db_path)
    df.to_sql(name='cr', con=connector, if_exists='replace', index=False)



url = 'https://cr.minzdrav.gov.ru/clin-rec/'

with sync_playwright() as p:
    page = get_page(p, url)
    report_path = download_report(page, "downloads")
    df = get_df_with_links(report_path)
#    write_to_excel(df, 'downloads')
    write_to_sqlite(df)