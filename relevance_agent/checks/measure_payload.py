"""Offline size of the LLM request for live_news/selected.json; no network, no paid calls."""
import json
import sys
from pathlib import Path
from langchain_core.messages import AIMessage
from relevance_agent.graph import build_graph
from relevance_agent.loaders import load_news
from relevance_agent.single_contracts import SingleNewsInput

ROOT = Path(__file__).resolve().parent / 'live_news'


class CaptureModel:
    """Records the request and returns an empty answer, so the graph never calls a provider."""
    model_name = 'offline-capture'

    def __init__(self):
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return AIMessage(content='{}')


def main():
    company = sys.argv[1] if len(sys.argv) > 1 else 'U_57706'
    model = CaptureModel()
    pipeline = build_graph(model)
    for news in load_news(ROOT / 'selected.json'):
        model.messages.clear()
        req = SingleNewsInput(company_id=company, news=news)
        req.options.max_refinement_rounds = 0
        pipeline.invoke({'request': req}, {'recursion_limit': 10})
        system, human = model.messages[0]
        print(json.dumps(dict(news_id=news.id, system_chars=len(system.content),
                              user_chars=len(human.content),
                              total_chars=len(system.content) + len(human.content)), ensure_ascii=False))


if __name__ == '__main__':
    main()
