import socket
import unittest
from unittest.mock import patch

import httpx

from relevance_agent.articles import ArticleError, ArticleTools, extract_article, public_target
from relevance_agent.contracts import NewsItem, RunOptions
from relevance_agent.loaders import InputDataError

TEXT = 'Компания сообщила о расширении производства и новых поставках оборудования. ' * 6
HTML = f'<html><title>Новость</title><nav>Меню</nav><article><h1>Событие</h1><p>{TEXT}</p><aside>Реклама</aside></article><footer>Подвал</footer></html>'.encode()


class BytesStream(httpx.SyncByteStream):
    def __init__(self, data): self.data = data
    def __iter__(self): yield self.data


def response(status=200, content=HTML, headers=None):
    return httpx.Response(status, headers={'content-type':'text/html; charset=utf-8', **(headers or {})}, stream=BytesStream(content))


def news(uid='n1', url='https://news.example/article'):
    return NewsItem(id=uid,title='Новость',summary='Исходное описание',link=url,published='',source='Тест')


class ArticleTests(unittest.TestCase):
    def setUp(self):
        resolver=patch('relevance_agent.articles.socket.getaddrinfo',return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('93.184.216.34',443))])
        resolver.start();self.addCleanup(resolver.stop)

    def tools(self, handler, **kwargs):
        client=httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return ArticleTools([news(),news('n2')],client=client,**kwargs)

    def test_extract_and_cache_preserves_summary(self):
        calls=[]
        def handler(request):
            calls.append(request)
            self.assertEqual(request.url.host,'93.184.216.34')
            self.assertEqual(request.headers['host'],'news.example')
            self.assertEqual(request.extensions['sni_hostname'],'news.example')
            return response()
        tools=self.tools(handler)
        result=tools.fetch_article('n1')
        self.assertEqual(result.status,'ok')
        self.assertIn('расширении производства',result.text)
        self.assertNotIn('Реклама',result.text)
        result.text='изменён'
        second=tools.fetch_article('n2')
        self.assertEqual(second.news_id,'n2')
        self.assertNotEqual(second.text,'изменён')
        self.assertEqual(len(calls),1)
        self.assertEqual(tools.news['n1'].summary,'Исходное описание')

    def test_redirect_uses_original_hostname(self):
        def handler(request):
            return response(302,headers={'location':'/new'}) if request.url.path=='/article' else response()
        result=self.tools(handler).fetch_article('n1')
        self.assertEqual(result.final_url,'https://news.example/new')

    def test_private_redirect_blocked(self):
        def resolver(host,*args,**kwargs):
            return [(socket.AF_INET,socket.SOCK_STREAM,6,'',('127.0.0.1' if host=='127.0.0.1' else '93.184.216.34',443))]
        with patch('relevance_agent.articles.socket.getaddrinfo',side_effect=resolver):
            result=self.tools(lambda r:response(302,headers={'location':'http://127.0.0.1/private'})).fetch_article('n1')
        self.assertEqual(result.status,'unavailable')
        self.assertIn('запрещены',result.error)

    def test_bad_urls_and_mixed_dns(self):
        for url in ['file:///etc/passwd','https://user:secret@news.example/','http://news.example:22/','']:
            with self.subTest(url=url),self.assertRaises(ArticleError):public_target(url)
        with patch('relevance_agent.articles.socket.getaddrinfo',return_value=[
            (2,1,6,'',('93.184.216.34',443)),(2,1,6,'',('10.0.0.1',443))]):
            with self.assertRaises(ArticleError):public_target('https://news.example')

    def test_status_errors(self):
        for status in [401,403,404,429,500]:
            with self.subTest(status=status):
                result=self.tools(lambda r:response(status)).fetch_article('n1')
                self.assertEqual(result.status,'unavailable')
                self.assertIn(str(status),result.error)
                self.assertIsNone(result.text)

    def test_timeout(self):
        def handler(request):raise httpx.ReadTimeout('timeout')
        self.assertEqual(self.tools(handler).fetch_article('n1').status,'unavailable')

    def test_redirect_loop_and_limit(self):
        for location,limit in [('/article',5),('/new',0)]:
            result=self.tools(lambda r:response(302,headers={'location':location}),max_redirects=limit).fetch_article('n1')
            self.assertEqual(result.status,'unavailable')

    def test_bad_content_and_size(self):
        for headers in [{'content-type':'application/pdf'},{'content-length':'3000000'}, {'content-encoding':'br'}]:
            with self.subTest(headers=headers):
                self.assertEqual(self.tools(lambda r:response(headers=headers)).fetch_article('n1').status,'unavailable')
        self.assertEqual(self.tools(lambda r:response(),max_bytes=20).fetch_article('n1').status,'unavailable')

    def test_truncation(self):
        result=self.tools(lambda r:response(),options=RunOptions(max_article_chars=50)).fetch_article('n1')
        self.assertEqual(result.status,'ok');self.assertTrue(result.truncated);self.assertEqual(len(result.text),50)

    def test_budget_and_unknown_id(self):
        tools=self.tools(lambda r:self.fail('Network must not run'),options=RunOptions(max_article_fetches=0))
        self.assertIn('лимит',tools.fetch_article('n1').error)
        with self.assertRaises(InputDataError):tools.fetch_article('missing')

    def test_empty_or_challenge_page(self):
        for content in [b'<html><nav>links</nav></html>',b'<html><title>Just a moment</title></html>']:
            self.assertEqual(self.tools(lambda r:response(content=content)).fetch_article('n1').status,'unavailable')

    def test_cyrillic_encoding_and_old_markup(self):
        html=f'<html><div><h1>Заголовок</h1><p>{TEXT}</p></div></html>'.encode('cp1251')
        self.assertIn('Компания сообщила',extract_article(html,'windows-1251'))

    def test_div_paragraphs_do_not_return_market_widgets(self):
        html = f'<main><p>Валютный виджет {"X" * 200}</p><div class="article_text_wrapper"><div class="doc-text-block__paragraph">{TEXT}</div><p>Ошибка графика</p></div></main>'.encode()
        result = extract_article(html)
        self.assertIn('Компания сообщила', result)
        self.assertNotIn('Ошибка графика', result)
        self.assertNotIn('Валютный виджет', result)

    def test_articlebody_and_hidden_noise(self):
        html=f'<article><div itemprop="articleBody"><p>{TEXT}</p><p hidden>Скрыто</p></div><div class="related"><p>Шум</p></div></article>'.encode()
        result=extract_article(html)
        self.assertNotIn('Скрыто',result);self.assertNotIn('Шум',result)


if __name__=='__main__':unittest.main()
