# THIRD_PARTY_NOTICES

Сторонние компоненты, модели, данные, сервисы и смоделированные интеграции, использованные в решении.

## Языковые модели

Используются только модели из списка кейса, через OpenAI-совместимый прокси организаторов
`https://changellenge.up.railway.app/v1` (ключ `LLM_API_KEY` в `.env`, в репозиторий не входит).

| Модель | Где используется |
|---|---|
| google/gemma-4-26b-a4b-it | аналитик новости (news_analysis) |
| deepseek/deepseek-v4.1-flash | аналитик влияния (news_analysis), агент релевантности (relevance_agent), чат-агент; запасная для аналитика новости |
| z-ai/glm-5.3-flash | критик (news_analysis) |
| qwen/qwen3.6-35b-a3b | запасная модель аналитика влияния и критика |

Модели эмбеддингов не используются: поиск по аналитике банка — лексический (BM25).

## Данные

| Данные | Источник | Как используются |
|---|---|---|
| `applications/csv/*.csv` — обезличенные компании, связи с физлицами, платежи | материалы кейса (Альфа-Банк) | профили компаний, контрагенты, база `db.sqlite3` |
| `applications/analytics/*.pdf` — аналитика Центра макроэкономических исследований Альфа-Банка | материалы кейса | распознаны в базу знаний `applications/knowledge_base` (RAG, позиция банка) |
| `applications/filtered_data/*` — профили компаний | получены из данных кейса скриптами `scripts/` | вход агентов |
| Новости | RSS-ленты из `feeds.txt` и статьи по ссылкам из них (Банк России, РБК, Интерфакс, Ведомости, Коммерсантъ, Металлинфо, vc.ru, CNews, Bloomberg, ФРС США, ЕЦБ, BBC, The Guardian) | загружаются во время работы; тексты статей принадлежат издателям и хранятся в базе только для демонстрации |
| `applications/news_samples/*` | тестовые новости: заголовки и ссылки из RSS, пересказ статей командой, синтетический пример по материалам кейса | разработка и проверка агентов |

## Дизайн

Интерфейс повторяет визуальный стиль интернет-банка «Альфа-Бизнес» (дизайн-система Core DS Альфа-Банка).
Шрифт Alfa Interface Sans подключается с CDN Альфа-Банка `alfabank.servicecdn.ru`. Логотип и фирменные элементы
принадлежат АО «Альфа-Банк» и используются в рамках кейса хакатона.

## Смоделированные интеграции

- Интернет-банк «Альфа-Бизнес»: точка входа и экраны сервиса воспроизведены в отдельном Django-приложении, реальной интеграции нет.
- Авторизация: вход по обезличенному ID компании из датасета (`U_…`) без пароля — демо-доступ вместо входа в интернет-банк.
- Блок «Рынки» на главной: статичные демо-данные (`base/mock_data.py`), не котировки в реальном времени.

## Python-пакеты

Версии зафиксированы в `requirements.lock`. Лицензии — из метаданных пакетов.

| Пакет | Версия | Лицензия |
|---|---|---|
| aiohappyeyeballs | 2.7.1 | Python Software Foundation License |
| aiohttp | 3.14.5 | Apache-2.0 AND MIT |
| aiosignal | 1.4.0 | Apache Software License |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.15.1 | MIT |
| asgiref | 3.12.1 | BSD License |
| attrs | 26.1.0 | MIT |
| babel | 2.18.0 | BSD License |
| beautifulsoup4 | 4.15.0 | MIT License |
| certifi | 2026.10.23 | Mozilla Public License 2.0 (MPL 2.0) |
| charset-normalizer | 3.5.2 | MIT |
| click | 8.5.0 | BSD-3-Clause |
| cloudpickle | 3.1.2 | BSD License |
| courlan | 1.4.0 | Apache-2.0 |
| cssselect | 1.6.0 | BSD-3-Clause |
| dateparser | 1.4.3 | BSD-3-Clause |
| defusedxml | 0.7.1 | Python Software Foundation License |
| distro | 1.9.0 | Apache Software License |
| Django | 5.2.18 | BSD-3-Clause |
| feedfinder2 | 0.0.4 | MIT License |
| feedparser | 6.0.14 | BSD-2-Clause |
| feedparser-sgmllib | 2.1.0 | PSF-2.0 |
| filelock | 4.2.4 | MIT |
| frozenlist | 1.8.0 | Apache-2.0 |
| h11 | 0.16.0 | MIT License |
| htmldate | 1.11.0 | Apache-2.0 |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpcore2 | 2.16.0 | BSD-3-Clause |
| httpx | 0.28.1 | BSD License |
| httpx2 | 2.16.0 | BSD-3-Clause |
| idna | 3.20 | BSD-3-Clause |
| jieba3k | 0.35.1 | MIT |
| jiter | 0.18.0 | MIT |
| joblib | 1.6.0 | BSD-3-Clause |
| jsonpatch | 1.37 | BSD-3-Clause |
| jsonpointer | 3.5 | BSD License |
| jusText | 3.0.2 | BSD License |
| langchain | 1.4.7 | MIT License |
| langchain-core | 1.7.0 | MIT License |
| langchain-openai | 1.7.2 | MIT License |
| langchain-protocol | 0.0.20 | MIT License |
| langgraph | 1.2.17 | MIT |
| langgraph-checkpoint | 4.3.0 | MIT |
| langgraph-prebuilt | 1.1.1 | MIT |
| langgraph-sdk | 0.4.9 | MIT |
| langsmith | 0.14.12 | MIT License |
| lxml | 6.1.3 | BSD-3-Clause |
| lxml_html_clean | 0.4.5 | BSD-3-Clause |
| multidict | 7.2.1 | Apache-2.0 |
| newspaper3k | 0.2.8 | MIT |
| nltk | 3.10.4 | Apache Software License |
| openai | 3.37.0 | Apache-2.0 |
| orjson | 3.13.0 | MPL-2.0 AND (Apache-2.0 OR MIT) |
| ormsgpack | 1.13.0 | Apache-2.0 OR MIT |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pillow | 13.0.0 | MIT-CMU |
| propcache | 0.5.4 | Apache-2.0 |
| pydantic | 2.14.1 | MIT |
| pydantic_core | 2.50.1 | MIT |
| python-dateutil | 3.0.0 | BSD License, Apache Software License |
| python-dotenv | 1.2.4 | BSD-3-Clause |
| pytz | 2026.5 | MIT License |
| PyYAML | 6.0.3 | MIT License |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python |
| requests | 2.34.2 | Apache Software License |
| requests-file | 3.0.1 | Apache Software License |
| requests-toolbelt | 1.0.0 | Apache Software License |
| six | 1.17.0 | MIT License |
| sniffio | 1.3.1 | MIT License, Apache Software License |
| soupsieve | 3.0.4 | MIT |
| sqlparse | 0.6.0 | BSD License |
| tenacity | 9.2.1 | Apache-2.0 |
| tiktoken | 0.14.0 | MIT License |
| tinysegmenter | 0.3 | BSD License |
| tld | 0.13.2 | MPL-1.1 OR GPL-2.0-only OR LGPL-2.1-or-later |
| tldextract | 5.4.0 | BSD-3-Clause |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| trafilatura | 2.3.1 | Apache-2.0 |
| truststore | 0.10.5 | MIT |
| typing-inspection | 0.4.4 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| tzdata | 2026.5 | Apache-2.0 |
| tzlocal | 5.4.5 | MIT |
| urllib3 | 2.8.0 | MIT |
| uuid_utils | 0.18.0 | BSD-3-Clause |
| websockets | 17.3 | BSD-3-Clause |
| xxhash | 4.0.2 | BSD-2-Clause |
| yarl | 1.25.1 | Apache-2.0 |
| zstandard | 0.25.0 | BSD-3-Clause |
