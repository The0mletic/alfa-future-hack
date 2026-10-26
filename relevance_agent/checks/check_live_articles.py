"""Ручная сетевая проверка, не входит в unit-тесты. Запуск из корня как модуль."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import httpx
from relevance_agent.articles import ArticleTools, extract_article
from relevance_agent.contracts import NewsItem

SOURCES = [
 ('cbr', 'https://www.cbr.ru/press/event/?id=32910'),
 ('kommersant', 'https://www.kommersant.ru/doc/9049591'),
 ('cnews', 'https://biz.cnews.ru/news/line/2026-10-23_mts_uspeshno_zakryla_knigu'),
 ('interfax', 'https://www.interfax-russia.ru/main/cb-suzil-prognoz-rosta-vvp-rf-v-2026g-do-0-5-1-0-ponizil-v-2027g-do-1-0-2-0'),
]

def check(source):
    name,url = source
    start=time.monotonic()
    n=NewsItem(id=name,title='Проверка источника',summary='',link=url,published='',source=name)
    with ArticleTools([n]) as tools:
        result=tools.fetch_article(name)
        if name in ('kommersant', 'interfax'):
            try:
                _, raw, _ = tools._download(url)
                (Path(__file__).parent / f'{name}.html').write_bytes(raw)
            except Exception:
                pass
    record={'source':name,'url':url,'elapsed_seconds':round(time.monotonic()-start,2),**result.model_dump()}
    # Контроль транспортного соединения обычным HTTPX по имени, без изменения рабочего инструмента.
    if result.status != 'ok':
        try:
            with httpx.Client(trust_env=False,timeout=10,follow_redirects=False) as client:
                r=client.get(url,headers={'Accept-Encoding':'identity'})
                record['control_by_hostname']={'status':r.status_code,'bytes':len(r.content)}
        except httpx.HTTPError as exc:
            record['control_by_hostname']={'error':type(exc).__name__}
    return record

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(check,SOURCES))
    path=Path(__file__).parent/'live_articles.json'
    path.write_text(json.dumps({'checked_at':datetime.now(timezone.utc).isoformat(),'results':results},ensure_ascii=False,indent=2)+'\n')
    for row in results:
        print(json.dumps({k:v for k,v in row.items() if k!='text'},ensure_ascii=False))
        if row['text']:print('TEXT START:',row['text'][:180],'\nTEXT END:',row['text'][-180:])
