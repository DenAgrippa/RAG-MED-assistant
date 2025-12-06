import pandas as pd
import sqlite3 as sqlite
import cr_minzdrav_scrapper
import parse_cr
import json

def get_json(link):
    page_content = cr_minzdrav_scrapper.get_page_content(link)
    json_content = parse_cr.kr_to_json(page_content)
    return json.dumps(json_content, ensure_ascii=False)


db_path = "db/db.sqlite"

with sqlite.connect(db_path) as connector:
    query = "SELECT * FROM cr"
    df = pd.read_sql_query(query, connector)
    df['JSON_content'] = df['HTML_link'].apply(get_json)
    df.to_sql(name='cr', con=connector, if_exists='replace', index=False)