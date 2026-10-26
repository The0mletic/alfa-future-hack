"""Получение текста статьи по ссылке, если парсер отдал только анонс."""
import ssl
from html.parser import HTMLParser

import certifi
import httpx

MAX_CHARS = 15000
MIN_CHARS = 400
USER_AGENT = "Mozilla/5.0 (compatible; AlfaNewsAnalysis/1.0)"


class _ParagraphParser(HTMLParser):
    """Собирает текст абзацев: <p> и блоки <div class="…paragraph…"> (так верстают, например, Коммерсантъ).

    Скрипты, стили и навигация пропускаются.
    """

    SKIP = {"script", "style", "nav", "header", "footer", "aside", "form"}

    def __init__(self):
        super().__init__()
        self.paragraphs: list[str] = []
        self._buf: list[str] = []
        self._para_tag: str | None = None
        self._depth = 0
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif self._para_tag is None:
            css = dict(attrs).get("class") or ""
            if tag == "p" or (tag == "div" and "paragraph" in css):
                self._para_tag, self._depth = tag, 1
        elif tag == self._para_tag:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == self._para_tag:
            self._depth -= 1
            if self._depth == 0:
                text = " ".join("".join(self._buf).split())
                if len(text) > 40:
                    self.paragraphs.append(text)
                self._buf, self._para_tag = [], None

    def handle_data(self, data):
        if self._para_tag and not self._skip:
            self._buf.append(data)


def fetch_article(url: str) -> tuple[str | None, str | None]:
    """Возвращает (текст, ошибка). Ошибка не роняет анализ — он пойдёт по анонсу."""
    if not url or not url.startswith(("http://", "https://")):
        return None, "нет ссылки на статью"
    try:
        with httpx.Client(timeout=15, follow_redirects=True, headers={"User-Agent": USER_AGENT},
                          verify=ssl.create_default_context(cafile=certifi.where())) as client:
            resp = client.get(url)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        return None, f"статья недоступна: {exc}"
    parser = _ParagraphParser()
    parser.feed(resp.text)
    text = "\n\n".join(parser.paragraphs)
    if len(text) < MIN_CHARS:
        return None, "не удалось извлечь текст статьи (возможно, нужен JavaScript или подписка)"
    return text[:MAX_CHARS], None
