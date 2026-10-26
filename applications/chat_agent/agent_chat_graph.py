"""Чат-агент «AI-аналитик» на LangGraph.

Что тут в целом происходит:
есть несколько шагов (collect, check, agent, tools, review), и каждый отвечает за определённую функцию
(берёт оригинал новости, проверяет разбор агента Аркаши по источникам и аналитике Альфы, проверяет,
подходят ли выводы Аркаши именно этой компании, сам решает, какие данные взять + самопроверка ответа).

Каждый шаг пишется в agent_log.jsonl

Структура:
есть чат в терминале и на сайте (окно «AI-аналитик»), человек пишет какой-то вопрос вроде:
"что делать? верно ли, что..." и т.д.
агент проходит шаги по графу, и появляется ответ.

Расход: каждый запрос к нейросети пишется в журнал с числом токенов;

Поиск по другим новостям (search_news) идёт по базе news.md (новости из RSS: ЦБ, Ведомости,
Коммерсантъ и др.). Файл лежит рядом с агентом; другой путь можно задать через NEWS_FILE.

"""
import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from base.mock_data import COMPANY, NEWS  # noqa: E402  пока берём демо-данные проекта


def _load_env(path: Path):
    """Читает общий .env команды в корне проекта (LLM_API_KEY и др.).
    Файл в .gitignore, на GitHub не попадает. Переменные, уже заданные в Терминале, не перезаписываются."""
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env(ROOT / '.env')

PROMPT_ANSWER = (HERE / 'prompt.md').read_text(encoding='utf-8')
PROMPT_CHECK = (HERE / 'prompt_check.md').read_text(encoding='utf-8')
PROMPT_REVIEW = (HERE / 'prompt_review.md').read_text(encoding='utf-8')
LOG_FILE = HERE / 'agent_log.jsonl'

MAX_FIXES = 1         # сколько раз самопроверка может вернуть ответ на доработку
MAX_TOOL_ROUNDS = 2   # сколько раз подряд агент может ходить за данными
CHECK_CACHE = {}      # проверка разбора одной новости делается один раз - экономим лимит

# Бюджет времени на один ответ. Агент сам решает, что успевает:
# не хватает времени - не ходит за данными, пропускает самопроверку, отдаёт то, что успел.
TIME_BUDGET = float(os.getenv('CHAT_TIMEOUT_SECONDS') or 60) - 2   # 2 с запаса на сайт и сеть
MIN_FOR_REVIEW = 7    # если осталось меньше - пропускает самопроверку
MIN_FOR_FIX = 12      # если осталось меньше - не отправляет ответ на доработку
CALL_SECONDS = 10.0   # сколько в среднем отвечает модель; уточняется на каждом запросе
FALLBACK = ('Не успел подготовить ответ за отведённое время. '
            'Попробуйте задать вопрос короче или уточнить, что именно важно.')


# журнал:

def log(step: str, **data):
    """Пишет шаг агента в журнал: что сделал и что получил."""
    record = {'time': datetime.now().isoformat(timespec='seconds'), 'step': step, **data}
    with LOG_FILE.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


#данные:

def _find_news(news_id: int) -> dict:
    for n in NEWS:
        if n['id'] == news_id:
            return n
    raise ValueError(f'Новость {news_id} не найдена')


def _stems(text: str) -> set:
    """Грубый поиск по словам: первые 5 букв, чтобы «ставка» и «ставки» совпадали."""
    return {w[:5] for w in re.findall(r'[а-яёa-z0-9]+', text.lower()) if len(w) >= 4}


def get_news_original(news_id: int) -> str:
    """Оригинальный текст новости (первоисточник)."""
    n = _find_news(news_id)
    url = next((s['url'] for s in n['sources'] if s.get('url')), 'нет')
    return (f"Источник: {n['source']}, {n['published']}\nСсылка: {url}\n"
            f"Заголовок: {n['title']}\n\n" + '\n\n'.join(n['body']))


def get_analysis(news_id: int) -> str:
    """Разбор новости от агента Аркаши (то, что пользователь видит в карточке)."""
    n = _find_news(news_id)
    lines = ['Почему важно: ' + n['why'], 'Что произошло:']
    lines += [f'- {f}' for f in n['facts']]
    lines.append('Риски:')
    lines += [f"- {r['title']}: {r['text']}" for r in n['risks']]
    lines.append('Возможности:')
    lines += [f"- {o['title']}: {o['text']}" for o in n['opportunities']]
    if n['bank_view']:
        lines.append('Мнение банка: ' + n['bank_view']['text'])
    return '\n'.join(lines)


# Нейросеть видит инструкции (названия и описания) и сама решает, какие вызвать.

@tool
def get_company_profile() -> str:
    """Профиль компании пользователя: отрасль, регион, масштаб, продукты банка, ВЭД, контрагенты."""
    c = COMPANY
    result = (f"{c['name']}. Отрасль: {c['industry']} (ОКВЭД {c['okved']}). Регион: {c['region']} "
              f"Выручка {c['revenue']}, {c['employees']} сотрудников. ВЭД: {c['foreign_trade']}. "
              f"Продукты банка: {', '.join(c['products'])}.")
    log('tool', name='get_company_profile')
    return result


@tool
def search_analytics(query: str) -> str:
    """Поиск по аналитике Альфа-Банка. query - короткий запрос по теме, например «ключевая ставка»."""
    words = _stems(query)
    hits = []
    for n in NEWS:
        bv = n['bank_view']
        if bv and words & _stems(f"{bv['doc']} {bv['text']} {n['title']}"):
            hits.append(f"Обзор «{bv['doc']}» ({bv['author']}, {bv['date']}): {bv['text']}")
    log('tool', name='search_analytics', query=query, found=len(hits))
    return '\n\n'.join(hits[:3]) or 'По этому запросу аналитики не найдено.'


NEWS_FILE = Path(os.getenv('NEWS_FILE') or HERE / 'news.md')
_NEWS_DB = None   # база новостей читается один раз, при первом поиске


def _load_news_db() -> list:
    """Читает news.md: заголовок, источник, дата, текст и ссылка каждой новости."""
    if not NEWS_FILE.exists():
        return []
    items = []
    for block in NEWS_FILE.read_text(encoding='utf-8').split('\n## ')[1:]:
        lines = block.strip().split('\n')
        title = re.sub(r'^\d+\.\s*', '', lines[0]).strip()
        meta = next((l.strip('_') for l in lines[1:] if l.startswith('_источник:')), '')
        source = re.search(r'источник:\s*([^·]+)', meta)
        date = re.search(r'дата:\s*([\d-]+)', meta)
        url = re.search(r'\[ссылка\]\((.+?)\)', block)
        text = ' '.join(l for l in lines[1:] if not l.startswith(('_источник:', '[ссылка]'))).strip()
        items.append({
            'title': title, 'text': text, 'url': url.group(1) if url else 'нет',
            'source': source.group(1).strip().strip('"') if source else 'неизвестен',
            'date': date.group(1) if date else 'неизвестна',
            'title_stems': _stems(title), 'text_stems': _stems(text),
        })
    return items


@tool
def search_news(query: str) -> str:
    """Поиск по базе новостей (ЦБ, Ведомости, Коммерсантъ и др.). query - короткий запрос:
    тема, отрасль или название компании, например «ключевая ставка» или «грузоперевозки»."""
    global _NEWS_DB
    if _NEWS_DB is None:
        _NEWS_DB = _load_news_db()
    words = _stems(query)

    if _NEWS_DB:
        scored = []
        for n in _NEWS_DB:
            score = 2 * len(words & n['title_stems']) + len(words & n['text_stems'])
            if score:
                scored.append((score, n['date'], n))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)   # сначала самые подходящие и свежие
        hits = [f"{n['title']} ({n['source']}, {n['date']}). {n['text'][:400]}\nСсылка: {n['url']}"
                for _, _, n in scored[:3]]
        base = 'news.md'
    else:   # базы нет - ищем по демо-новостям проекта
        hits = [f"{n['title']}. {n['summary']}" for n in NEWS if words & _stems(f"{n['title']} {n['summary']}")][:3]
        base = 'demo'

    log('tool', name='search_news', query=query, base=base, found=len(hits))
    return '\n\n'.join(hits) or 'По этому запросу новостей не найдено. Попробуй другой запрос.'


TOOLS = [get_company_profile, search_analytics, search_news]


#нейронка:

class DemoModel(BaseChatModel):
    """Заглушка вместо нейросети, пока нет ключа. Не думает, а проходит шаги графа."""

    @property
    def _llm_type(self) -> str:
        return 'demo'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = messages[0].content
        if system == PROMPT_CHECK:
            msg = AIMessage(content='{"демо": "здесь будет результат проверки разбора"}')
        elif system == PROMPT_REVIEW:
            msg = AIMessage(content='{"ok": true, "problems": [], "fix": ""}')
        else:
            # Были ли вызовы инструментов после последнего вопроса?
            last_q = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
            used_tools = any(isinstance(m, ToolMessage) for m in messages[last_q:])
            if not used_tools:
                msg = AIMessage(content='', tool_calls=[
                    {'name': 'get_company_profile', 'args': {}, 'id': uuid.uuid4().hex},
                    {'name': 'search_analytics', 'args': {'query': 'ключевая ставка'}, 'id': uuid.uuid4().hex},
                ])
            else:
                msg = AIMessage(content='[Демо-режим: здесь будет ответ нейросети. '
                                        'Агент собрал данные и прошёл самопроверку.]')
        # Примерный расход токенов (≈ 4 символа на токен), чтобы в демо было видно, как считается
        tokens_in = sum(len(str(m.content)) for m in messages) // 4
        tokens_out = len(str(msg.content)) // 4 + 20 * len(msg.tool_calls)
        msg.usage_metadata = {'input_tokens': tokens_in, 'output_tokens': tokens_out,
                              'total_tokens': tokens_in + tokens_out}
        return ChatResult(generations=[ChatGeneration(message=msg)])


def make_llm(model: str, timeout: float = 60):
    """Настоящая нейросеть, если есть ключ и модель, иначе демо-модель.
    timeout - сколько секунд ждать ответа модели, потом запрос обрывается."""
    api_key = os.getenv('LLM_API_KEY')
    if not api_key or not model:
        return DemoModel()
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(api_key=api_key, base_url=os.getenv('LLM_BASE_URL') or None,
                      model=model, temperature=0.2, timeout=max(timeout, 1), max_retries=0)


# Разные модели для разных шагов, чтобы экономить общий лимит команды:
MODEL_ANSWER = os.getenv('LLM_MODEL', '')
MODEL_CHECK = os.getenv('LLM_MODEL_CHECK') or MODEL_ANSWER
LLM = make_llm(MODEL_ANSWER)


# Расход токенов
# Каждый запрос к нейросети пишется в журнал (step='llm') с числом токенов.
# Если задать цены (₽ за 1000 токенов), журнал считает и стоимость

PRICE_IN = float(os.getenv('PRICE_IN_RUB_PER_1K') or 0)
PRICE_OUT = float(os.getenv('PRICE_OUT_RUB_PER_1K') or 0)
USAGE = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}   # счётчик на текущий вопрос


def call_llm(model: str, messages: list, step: str, news_id: int, timeout: float, tools=None):
    """Запрос к нейросети (не дольше timeout секунд) + запись в журнал, сколько токенов он потратил."""
    llm = make_llm(model, timeout)
    if tools:
        llm = llm.bind_tools(tools)
    global CALL_SECONDS
    started = time.monotonic()
    response = llm.invoke(messages)
    if step == 'agent':   # запоминаем скорость модели, чтобы планировать время
        CALL_SECONDS = 0.5 * CALL_SECONDS + 0.5 * (time.monotonic() - started)
    usage = getattr(response, 'usage_metadata', None) or {}
    tokens_in, tokens_out = usage.get('input_tokens', 0), usage.get('output_tokens', 0)
    USAGE['calls'] += 1
    USAGE['input_tokens'] += tokens_in
    USAGE['output_tokens'] += tokens_out
    log('llm', news_id=news_id, for_step=step, model=model or 'demo', seconds=round(time.monotonic() - started, 1),
        input_tokens=tokens_in, output_tokens=tokens_out, cost_rub=_cost(tokens_in, tokens_out))
    return response


def _cost(tokens_in: int, tokens_out: int) -> float:
    return round(tokens_in / 1000 * PRICE_IN + tokens_out / 1000 * PRICE_OUT, 4)


def _parse_json(text: str) -> dict:
    """Достаёт JSON из ответа нейросети, даже если вокруг есть лишний текст."""
    match = re.search(r'\{.*\}', text, re.S)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}


#граф:
#Всё, что агент знает на текущем шаге. Сохраняется между вопросами (история чата).

class State(TypedDict, total=False):
    news_id: int
    question: str
    messages: Annotated[list, add_messages]   # история: вопросы, вызовы инструментов, ответы
    original: str
    analysis: str
    profile: str
    check: str
    check_ready: bool   # проверка разбора уже готова и передана агенту
    original_sent: bool # новость и профиль уже отправлены агенту (в истории чата)
    deadline: float     # к какому моменту (time.monotonic) нужно отдать ответ
    final: bool         # время вышло - отдаём ответ без самопроверки
    answer: str
    verdict: str        # 'ok' или 'fix' - решение самопроверки
    fixes: int
    tool_rounds: int


#шаги графа:

def collect(state: State) -> dict:
    """Шаг 1. Берёт новость, разбор и профиль. На повторных вопросах уже всё есть."""
    if state.get('original'):
        return {}
    news_id = state['news_id']
    log('collect', news_id=news_id)
    return {
        'original': get_news_original(news_id),
        'analysis': get_analysis(news_id),
        'profile': get_company_profile.invoke({}),
    }


CHECK_RUNNING = set()   # новости, по которым проверка разбора идёт прямо сейчас


def _run_check(news_id: int, original: str, analysis: str, profile: str):
    """Проверка разбора (в фоне). Результат сохраняется и используется в следующих ответах."""
    try:
        title = _find_news(news_id)['title']
        analytics = search_analytics.invoke({'query': title})
        context = (f"Оригинал новости:\n{original}\n\n"
                   f"Разбор:\n{analysis}\n\n"
                   f"Профиль компании:\n{profile}\n\n"
                   f"Аналитика Альфа-Банка:\n{analytics}")
        result = call_llm(MODEL_CHECK, [SystemMessage(content=PROMPT_CHECK), HumanMessage(content=context)],
                          'check', news_id, timeout=90).content
        CHECK_CACHE[news_id] = result
        log('check', news_id=news_id, result=result)
    except Exception as e:   # проверка не удалась - чат всё равно работает
        log('check_failed', news_id=news_id, error=str(e)[:300])
    finally:
        CHECK_RUNNING.discard(news_id)


def check(state: State) -> dict:
    """Шаг 2. Проверка разбора + сообщение с вопросом для агента.
    Проверка идёт в фоне, чтобы не тормозить ответ: если она уже готова - агент её учитывает,
    если нет - сам сверяет разбор с оригиналом, а результат проверки получит в следующих вопросах."""
    news_id, question = state['news_id'], state['question']

    if news_id not in CHECK_CACHE and news_id not in CHECK_RUNNING:
        CHECK_RUNNING.add(news_id)
        thread = threading.Thread(target=_run_check, daemon=True,
                                  args=(news_id, state['original'], state['analysis'], state['profile']))
        thread.start()
        thread.join(timeout=1)   # быстрая модель (или демо) успеет, медленная - доделает в фоне

    ready = news_id in CHECK_CACHE
    check_text = CHECK_CACHE.get(news_id, 'Проверка разбора ещё не готова: сверяй разбор с оригиналом сам.')

    if state.get('original_sent'):          # повторный вопрос по той же новости
        parts = []
        if ready and not state.get('check_ready'):
            parts.append(f'Результат проверки разбора (пользователь это не должен видеть):\n{check_text}')
        parts.append(f'Вопрос пользователя:\n{question}')
        return {'check': check_text, 'check_ready': ready, 'messages': [HumanMessage(content='\n\n'.join(parts))]}

    first = (f"Оригинал новости:\n{state['original']}\n\n"
             f"Разбор (пользователь видел его в карточке):\n{state['analysis']}\n\n"
             f"Результат проверки разбора (пользователь это не должен видеть):\n{check_text}\n\n"
             f"Профиль компании:\n{state['profile']}\n\n"
             f"Вопрос пользователя:\n{question}")
    return {'check': check_text, 'check_ready': ready, 'original_sent': True,
            'messages': [HumanMessage(content=first)]}


def _left(state: State) -> float:
    """Сколько секунд осталось до дедлайна ответа."""
    return state['deadline'] - time.monotonic()


def _last_draft(state: State) -> str:
    """Последний готовый текст ответа (если агент уже что-то написал)."""
    for m in reversed(state['messages']):
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            return m.content
    return ''


def agent(state: State) -> dict:
    """Шаг 3. Нейросеть решает: вызвать инструменты или уже ответить.
    Если времени мало - инструменты не даём, сразу просим ответ."""
    left = _left(state)
    if left < 3:
        log('timeout', news_id=state['news_id'], at='agent', left=round(left, 1))
        return {'messages': [AIMessage(content=_last_draft(state) or FALLBACK)], 'final': True}

    # За данными идём, только если успеем потом ещё и написать ответ (два запроса к модели)
    use_tools = state.get('tool_rounds', 0) < MAX_TOOL_ROUNDS and left >= 2 * CALL_SECONDS + 2
    try:
        response = call_llm(MODEL_ANSWER, [SystemMessage(content=PROMPT_ANSWER)] + state['messages'],
                            'agent', state['news_id'], timeout=left - 1, tools=TOOLS if use_tools else None)
    except Exception as e:   # модель не успела или сеть упала
        log('timeout', news_id=state['news_id'], at='agent', error=str(e)[:200])
        return {'messages': [AIMessage(content=_last_draft(state) or FALLBACK)], 'final': True}

    if response.tool_calls:
        log('plan', news_id=state['news_id'],
            tools=[{'name': c['name'], 'args': c['args']} for c in response.tool_calls])
    else:
        log('draft', news_id=state['news_id'], answer=response.content[:500])
    return {'messages': [response]}


_tool_node = ToolNode(TOOLS)


def tools(state: State) -> dict:
    """Шаг 4. Выполняет инструменты, которые выбрал агент."""
    result = _tool_node.invoke(state)
    return {**result, 'tool_rounds': state.get('tool_rounds', 0) + 1}


def review(state: State) -> dict:
    """Шаг 5. Самопроверка ответа. Если есть проблемы - возвращает агенту с замечаниями."""
    draft = state['messages'][-1].content
    left = _left(state)
    if state.get('final') or left < MIN_FOR_REVIEW:
        log('review_skipped', news_id=state['news_id'], left=round(left, 1))
        return {'answer': draft, 'verdict': 'ok'}

    collected = '\n\n'.join(m.content for m in state['messages'] if isinstance(m, ToolMessage))
    context = (f"Оригинал новости:\n{state['original']}\n\n"
               f"Результат проверки разбора:\n{state['check']}\n\n"
               f"Данные, которые собрал агент:\n{collected or 'нет'}\n\n"
               f"Вопрос пользователя:\n{state['question']}\n\n"
               f"Ответ агента:\n{draft}")
    try:
        verdict = _parse_json(call_llm(MODEL_CHECK, [SystemMessage(content=PROMPT_REVIEW), HumanMessage(content=context)],
                                       'review', state['news_id'], timeout=left - 1).content)
    except Exception as e:   # не успела - отдаём ответ без самопроверки
        log('review_skipped', news_id=state['news_id'], error=str(e)[:200])
        return {'answer': draft, 'verdict': 'ok'}
    ok = verdict.get('ok', True)
    fixes = state.get('fixes', 0)
    log('review', news_id=state['news_id'], ok=ok, problems=verdict.get('problems', []), attempt=fixes + 1)

    if ok or fixes >= MAX_FIXES or _left(state) < MIN_FOR_FIX:
        return {'answer': draft, 'verdict': 'ok'}

    feedback = ('Самопроверка нашла проблемы в ответе: ' + '; '.join(verdict.get('problems', [])) +
                f"\nЧто сделать: {verdict.get('fix', '')}\n"
                'Исправь ответ. Если не хватает данных, вызови инструменты. Пользователю проверку не упоминай.')
    return {'messages': [HumanMessage(content=feedback)], 'verdict': 'fix', 'fixes': fixes + 1}


#Сборка графа:

def after_agent(state: State) -> str:
    return 'tools' if state['messages'][-1].tool_calls else 'review'


def after_review(state: State) -> str:
    return END if state['verdict'] == 'ok' else 'agent'


builder = StateGraph(State)
builder.add_node('collect', collect)
builder.add_node('check', check)
builder.add_node('agent', agent)
builder.add_node('tools', tools)
builder.add_node('review', review)
builder.add_edge(START, 'collect')
builder.add_edge('collect', 'check')
builder.add_edge('check', 'agent')
builder.add_conditional_edges('agent', after_agent, ['tools', 'review'])
builder.add_edge('tools', 'agent')
builder.add_conditional_edges('review', after_review, ['agent', END])

graph = builder.compile(checkpointer=MemorySaver())   # MemorySaver хранит историю чата


def ask(news_id: int, question: str, user_id: str = 'demo') -> str:
    """Главная функция: вопрос по новости -> ответ агента. Её потом вызовет Django."""
    config = {'configurable': {'thread_id': f'{user_id}:{news_id}'}}   # своя история на каждую новость
    log('question', news_id=news_id, question=question)
    USAGE.update(calls=0, input_tokens=0, output_tokens=0)
    started = time.monotonic()
    result = graph.invoke({'news_id': news_id, 'question': question, 'fixes': 0, 'tool_rounds': 0,
                           'final': False, 'deadline': started + TIME_BUDGET}, config)
    log('usage', news_id=news_id, question=question, **USAGE, seconds=round(time.monotonic() - started, 1),
        cost_rub=_cost(USAGE['input_tokens'], USAGE['output_tokens']))   # итог по вопросу
    return result['answer']


def usage_report():
    """Сводка расхода по журналу: сколько вопросов, токенов и рублей."""
    if not LOG_FILE.exists():
        return 'Журнал пуст.'
    rows = [json.loads(line) for line in LOG_FILE.read_text(encoding='utf-8').splitlines() if line.strip()]
    per_question = [r for r in rows if r['step'] == 'usage']
    if not per_question:
        return 'В журнале ещё нет вопросов с подсчётом токенов.'
    n = len(per_question)
    calls = sum(r['calls'] for r in per_question)
    t_in = sum(r['input_tokens'] for r in per_question)
    t_out = sum(r['output_tokens'] for r in per_question)
    cost = sum(r.get('cost_rub', 0) for r in per_question)
    lines = [f'Вопросов: {n}',
             f'Запросов к нейросети: {calls} (в среднем {calls / n:.1f} на вопрос)',
             f'Токенов: {t_in + t_out} (входящих {t_in}, исходящих {t_out}), '
             f'в среднем {(t_in + t_out) / n:.0f} на вопрос']
    if PRICE_IN or PRICE_OUT:
        lines.append(f'Стоимость: {cost:.2f} ₽, в среднем {cost / n:.3f} ₽ на вопрос, '
                     f'на 500 ₽ хватит примерно на {int(500 / (cost / n)) if cost else "∞"} вопросов')
    else:
        lines.append('Стоимость не считается: задай PRICE_IN_RUB_PER_1K и PRICE_OUT_RUB_PER_1K.')
    return '\n'.join(lines)


if __name__ == '__main__':
    if '--graph' in sys.argv:
        print(graph.get_graph().draw_mermaid())   # схема для презентации (вставить в mermaid.live)
    elif '--usage' in sys.argv:
        print(usage_report())                     # сколько потрачено токенов и рублей
    elif len(sys.argv) > 1:
        print(ask(news_id=1, question=sys.argv[1]))
    else:
        print('Чат по новости 1. Пустая строка - выход.')
        if isinstance(LLM, DemoModel):
            print('(демо-режим: ключа API нет, нейросеть не подключена)')
        while question := input('\nВы: ').strip():
            print('Агент:', ask(news_id=1, question=question))
