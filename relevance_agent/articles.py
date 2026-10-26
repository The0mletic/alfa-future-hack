"""Получение текста публичной HTML-статьи по ID входной новости. Без LLM."""
import argparse
import ipaddress
import re
import socket
import sys
import time
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup

from .contracts import ArticleResult, NewsItem, RelevanceInput, RunOptions
from .loaders import InputDataError, _validate, read_json


class ArticleError(ValueError):
    pass


def public_target(url: str) -> tuple[str, str, str]:
    """Проверяет адрес и закрепляет соединение за проверенным IP (включая редиректы)."""
    try:
        parts = urlsplit(url)
        port = parts.port or (443 if parts.scheme == 'https' else 80)
        host = (parts.hostname or '').encode('idna').decode('ascii')
        if parts.scheme not in ('http', 'https') or not host or parts.username is not None or parts.password is not None:
            raise ArticleError('Недопустимый URL: требуется публичный HTTP(S) без учётных данных.')
        if port not in (80, 443) or any(ord(c) < 33 for c in url):
            raise ArticleError('Недопустимый порт или символы в URL.')
        records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        addresses = list(dict.fromkeys(row[4][0] for row in records))
        if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
            raise ArticleError('Локальные, частные и служебные сетевые адреса запрещены.')
        address = addresses[0]
        ip_host = f'[{address}]' if ':' in address else address
        authority = f'[{host}]' if ':' in host else host
        default_port = 443 if parts.scheme == 'https' else 80
        if port != default_port:
            authority += f':{port}'
        pinned = urlunsplit((parts.scheme, f'{ip_host}:{port}', parts.path or '/', parts.query, ''))
        return pinned, authority, host
    except (ValueError, UnicodeError, OSError) as exc:
        if isinstance(exc, ArticleError):
            raise
        raise ArticleError('Не удалось проверить адрес или разрешить имя источника.') from exc


def extract_article(html: bytes, charset: str | None = None) -> str:
    """Консервативное извлечение HTML-текста; без JS, OCR и обхода подписки."""
    soup = BeautifulSoup(html, 'html.parser', from_encoding=charset)
    title = soup.title.get_text(' ', strip=True).lower() if soup.title else ''
    if any(marker in title for marker in ('access denied', 'just a moment', 'captcha', 'доступ ограничен', 'проверка браузера')):
        raise ArticleError('Источник вернул страницу ограничения доступа или проверки браузера.')
    for node in soup.select('script, style, noscript, nav, footer, aside, form, button, iframe, svg, [hidden], [aria-hidden="true"]'):
        node.decompose()
    for node in soup.select('[class], [id]'):
        if node.attrs is None:
            continue
        labels = ' '.join(node.get('class', [])) + ' ' + node.get('id', '')
        if re.search(r'(^|[\s_-])(cookie|subscribe|subscription|paywall|advert|advertisement|social|related|comments)([\s_-]|$)', labels, re.I):
            node.decompose()

    def paragraphs(node):
        # Исключаем вложенное дублирование, оставляем заголовки, абзацы и списки.
        tags = node.find_all(['h1','h2','h3','p','li','blockquote'])
        lines = []
        for tag in tags:
            if tag.find(['p','li','blockquote']):
                continue
            text = re.sub(r'\s+', ' ', tag.get_text(' ', strip=True)).strip()
            linked = sum(len(a.get_text(' ', strip=True)) for a in tag.find_all('a'))
            if text and linked <= len(text) * .6 and (not lines or text != lines[-1]):
                lines.append(text)
        return '\n\n'.join(lines)

    # Вёрстка Коммерсанта: основные абзацы — div, а не p; main содержит
    # курсы валют и виджеты. Выбираем только абзацы внутри тела статьи.
    blocks = soup.select('.article_text_wrapper .doc-text-block__paragraph')
    if blocks:
        text = '\n\n'.join(re.sub(r'\s+', ' ', node.get_text(' ', strip=True)).strip() for node in blocks)
        if len(text) >= 120:
            return text

    # articleBody приоритетнее article: в контейнере могут быть связанные материалы.
    for selector in ('[itemprop="articleBody"]', 'article', 'main', '[role="main"]'):
        candidates = soup.select(selector)
        texts = [paragraphs(node) or node.get_text(' ', strip=True) for node in candidates]
        if texts and len(max(texts, key=len)) >= 120:
            return max(texts, key=len).strip()
    # Для старых сайтов без семантической разметки выбираем блок с абзацами,
    # не весь body: меню и список ссылок не выдаём за статью.
    candidates = {id(p.parent): p.parent for p in soup.find_all('p') if p.parent.name in ('div','section')}
    texts = [paragraphs(node) for node in candidates.values()]
    text = max(texts, key=len, default='')
    if len(text) < 120:
        raise ArticleError('Основной текст статьи не найден или слишком короткий; возможно, требуется JavaScript или доступ по подписке.')
    return text


class ArticleTools:
    """Один экземпляр на запуск; кэш и бюджет используют настройки RelevanceInput."""
    def __init__(self, news: list[NewsItem], options: RunOptions | None = None, *,
                 client: httpx.Client | None = None, max_bytes: int = 2_000_000,
                 timeout_seconds: float = 10, deadline_seconds: float = 25, max_redirects: int = 5):
        if max_bytes < 1 or timeout_seconds <= 0 or deadline_seconds <= 0 or max_redirects < 0:
            raise ValueError('Некорректные ограничения загрузки')
        self.news = {n.id: n.model_copy(deep=True) for n in news}
        if len(self.news) != len(news):
            raise InputDataError('Повторяющийся ID новости')
        self.options = (options or RunOptions()).model_copy(deep=True)
        self.max_bytes = max_bytes
        self.timeout_seconds = timeout_seconds
        self.deadline_seconds = deadline_seconds
        self.max_redirects = max_redirects
        self._owned_client = client is None
        self.client = client or httpx.Client(trust_env=False, follow_redirects=False)
        self._cache = {}
        self.fetch_count = 0

    def close(self):
        if self._owned_client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _download(self, url: str) -> tuple[str, bytes, str | None]:
        current = url
        visited = set()
        deadline = time.monotonic() + self.deadline_seconds
        for hop in range(self.max_redirects + 1):
            if current in visited:
                raise ArticleError('Цикл перенаправлений.')
            visited.add(current)
            pinned, authority, sni_host = public_target(current)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ArticleError('Превышено время загрузки статьи.')
            # Подключаемся именно к проверенному IP, сохраняя Host, TLS SNI и проверку сертификата.
            with self.client.stream('GET', pinned,
                    headers={'Host':authority, 'User-Agent':'AlfaHackRelevance/1.0',
                             'Accept':'text/html, application/xhtml+xml', 'Accept-Encoding':'identity'},
                    extensions={'sni_hostname':sni_host}, follow_redirects=False,
                    timeout=min(self.timeout_seconds, remaining)) as response:
                if response.status_code in (301,302,303,307,308):
                    location = response.headers.get('location')
                    if not location or hop == self.max_redirects:
                        raise ArticleError('Некорректное перенаправление или превышен их лимит.')
                    current = urljoin(current, location)
                    continue
                if response.status_code != 200:
                    raise ArticleError(f'Источник вернул HTTP {response.status_code}.')
                content_type = response.headers.get('content-type', '')
                if content_type.split(';')[0].strip().lower() not in ('text/html','application/xhtml+xml'):
                    raise ArticleError('Источник вернул не HTML-статью.')
                if response.headers.get('content-encoding','identity').lower() not in ('','identity'):
                    raise ArticleError('Источник не поддержал загрузку без сжатия; ответ не обработан.')
                length = response.headers.get('content-length')
                if length and (not length.isdigit() or int(length) > self.max_bytes):
                    raise ArticleError('Недопустимый размер страницы.')
                content = bytearray()
                for chunk in response.iter_raw(chunk_size=16384):
                    if time.monotonic() > deadline:
                        raise ArticleError('Превышено время загрузки статьи.')
                    if len(content) + len(chunk) > self.max_bytes:
                        raise ArticleError('Страница превышает ограничение размера.')
                    content.extend(chunk)
                match = re.search(r'charset\s*=\s*["\']?([\w-]+)', content_type, re.I)
                charset = match.group(1) if match else None
                return current, bytes(content), charset
        raise ArticleError('Превышен лимит перенаправлений.')

    def fetch_article(self, news_id: str) -> ArticleResult:
        if news_id not in self.news:
            raise InputDataError('news_id отсутствует во входном списке')
        news = self.news[news_id]
        if news.full_text.strip():
            # Upstream parser does not report final redirect URL or extraction method.
            limit = self.options.max_article_chars
            return ArticleResult(news_id=news_id, status='ok', requested_url=news.link,
                text_source='upstream_parser', text=news.full_text[:limit],
                truncated=len(news.full_text) > limit)
        if news.link in self._cache:
            result = self._cache[news.link].model_copy(deep=True)
            result.news_id = news_id
            return result
        if self.fetch_count >= self.options.max_article_fetches:
            return ArticleResult(news_id=news_id, status='unavailable', requested_url=news.link,
                                 error='Исчерпан лимит получения статей для этого запуска.')
        self.fetch_count += 1
        try:
            final_url, html, charset = self._download(news.link)
            text = extract_article(html, charset)
            limit = self.options.max_article_chars
            result = ArticleResult(news_id=news_id, status='ok', requested_url=news.link,
                final_url=final_url, text=text[:limit], truncated=len(text) > limit)
        except (ArticleError, httpx.HTTPError, OSError, UnicodeError, LookupError) as exc:
            message = str(exc) if isinstance(exc, ArticleError) else f'Ошибка загрузки: {type(exc).__name__}.'
            result = ArticleResult(news_id=news_id, status='unavailable', requested_url=news.link, error=message)
        self._cache[news.link] = result
        return result.model_copy(deep=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Получить статью для новости из собранного RelevanceInput')
    parser.add_argument('--input', required=True)
    parser.add_argument('--news-id', required=True)
    args = parser.parse_args(argv)
    try:
        request = _validate(RelevanceInput, read_json(args.input), args.input)
        with ArticleTools(request.news, request.options) as tools:
            result = tools.fetch_article(args.news_id)
        print(result.model_dump_json(indent=2))
        return 0 if result.status == 'ok' else 1
    except (InputDataError, ValueError) as exc:
        print(f'Ошибка: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
