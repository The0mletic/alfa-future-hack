(function () {
    'use strict';

    /* ---------- Фильтр ленты ---------- */

    function initFeed(feed) {
        var chips = feed.querySelectorAll('[data-filter]');
        var cards = feed.querySelectorAll('.news-card');
        var empty = feed.querySelector('[data-feed-empty]');

        chips.forEach(function (chip) {
            chip.addEventListener('click', function () {
                var filter = chip.dataset.filter;
                var visible = 0;

                chips.forEach(function (c) {
                    c.setAttribute('aria-pressed', String(c === chip));
                });
                cards.forEach(function (card) {
                    var show = filter === 'all' || card.dataset.category === filter;
                    card.hidden = !show;
                    if (show) visible += 1;
                });
                if (empty) empty.hidden = visible > 0;
            });
        });
    }

    /* ---------- Табы (сегментированный контрол) ---------- */

    function initTabs(tablist) {
        var tabs = Array.prototype.slice.call(tablist.querySelectorAll('[role="tab"]'));

        function select(tab, focus) {
            tabs.forEach(function (t) {
                var active = t === tab;
                var panel = document.getElementById(t.getAttribute('aria-controls'));
                t.setAttribute('aria-selected', String(active));
                t.tabIndex = active ? 0 : -1;
                panel.hidden = !active;
                if (active) panel.parentElement.scrollTop = 0;
            });
            if (focus) tab.focus();
        }

        tabs.forEach(function (tab, i) {
            tab.addEventListener('click', function () {
                select(tab, false);
            });
            tab.addEventListener('keydown', function (e) {
                var next = null;
                if (e.key === 'ArrowRight') next = tabs[(i + 1) % tabs.length];
                if (e.key === 'ArrowLeft') next = tabs[(i - 1 + tabs.length) % tabs.length];
                if (e.key === 'Home') next = tabs[0];
                if (e.key === 'End') next = tabs[tabs.length - 1];
                if (next) {
                    e.preventDefault();
                    select(next, true);
                }
            });
        });
    }

    /* ---------- Мини-чат с AI-аналитиком ---------- */

    // Демо-ответы, пока не подключён бэкенд (data-endpoint пустой).
    var DEMO_ANSWERS = [
        {
            match: /финанс|кредит|долг|деньг|ставк|выручк/i,
            text: 'Я оцениваю влияние через три канала: процентные расходы, стоимость закупок ' +
                'и спрос со стороны клиентов. Ключевые риски и возможности по этой новости собраны ' +
                'в панели выше, а детальный расчёт по вашим договорам появится после подключения модели.'
        },
        {
            match: /делать|действ|месяц|рекоменд|план/i,
            text: 'На ближайший месяц я бы предложил:\n' +
                '1. Сверить риски из панели с действующими договорами и бюджетом.\n' +
                '2. Выбрать возможности, которые реально реализовать за квартал.\n' +
                '3. Обсудить выбранные шаги с персональным менеджером.'
        },
        {
            match: /основан|источник|почему|откуда/i,
            text: 'Вывод опирается на профиль компании, текст новости и аналитические материалы ' +
                'Альфа-Банка. Признаки компании перечислены в блоке «Основано на», источники — ' +
                'в конце статьи.'
        }
    ];

    var DEMO_FALLBACK = 'Это демо-режим: модель ещё не подключена. ' +
        'Когда бэкенд будет готов, здесь появится ответ с учётом профиля компании и аналитики банка.';

    function demoAnswer(question) {
        for (var i = 0; i < DEMO_ANSWERS.length; i += 1) {
            if (DEMO_ANSWERS[i].match.test(question)) return DEMO_ANSWERS[i].text;
        }
        return DEMO_FALLBACK;
    }

    function getCookie(name) {
        var match = document.cookie.match(new RegExp('(?:^|; )' + name + '=([^;]*)'));
        return match ? decodeURIComponent(match[1]) : '';
    }

    function initChat(chat) {
        var messages = chat.querySelector('[data-chat-messages]');
        var suggest = chat.querySelector('[data-chat-suggest]');
        var form = chat.querySelector('[data-chat-form]');
        var input = chat.querySelector('[data-chat-input]');
        var send = chat.querySelector('[data-chat-send]');
        var toggle = chat.querySelector('[data-chat-toggle]');
        var endpoint = chat.dataset.endpoint;
        var busy = false;

        function setCollapsed(collapsed) {
            chat.classList.toggle('is-collapsed', collapsed);
            toggle.setAttribute('aria-expanded', String(!collapsed));
            toggle.querySelector('.visually-hidden').textContent = collapsed ? 'Развернуть чат' : 'Свернуть чат';
        }

        function scrollToEnd() {
            messages.scrollTop = messages.scrollHeight;
        }

        function addMessage(role, text) {
            var el = document.createElement('div');
            el.className = 'msg msg--new msg--' + role;
            el.textContent = text;
            messages.appendChild(el);
            scrollToEnd();
            return el;
        }

        function addTyping() {
            var el = document.createElement('div');
            el.className = 'msg msg--new msg--ai msg--typing';
            el.setAttribute('aria-label', 'AI-аналитик печатает');
            el.innerHTML = '<i></i><i></i><i></i>';
            messages.appendChild(el);
            scrollToEnd();
            return el;
        }

        function autosize() {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 120) + 'px';
        }

        function syncSend() {
            send.disabled = busy || !input.value.trim();
        }

        function requestAnswer(question) {
            if (!endpoint) {
                return new Promise(function (resolve) {
                    setTimeout(function () {
                        resolve(demoAnswer(question));
                    }, 900);
                });
            }
            return fetch(endpoint, {
                method: 'POST',
                headers: {'Content-Type': 'application/json', 'X-CSRFToken': getCookie('csrftoken')},
                body: JSON.stringify({message: question, news_id: chat.dataset.newsId})
            }).then(function (res) {
                if (!res.ok) throw new Error(res.status);
                return res.json();
            }).then(function (data) {
                return data.answer;
            });
        }

        function ask(question) {
            question = question.trim();
            if (!question || busy) return;

            busy = true;
            setCollapsed(false);
            suggest.hidden = true;
            addMessage('user', question);
            input.value = '';
            autosize();
            syncSend();

            var typing = addTyping();
            requestAnswer(question)
                .then(function (answer) {
                    typing.remove();
                    addMessage('ai', answer);
                })
                .catch(function () {
                    typing.remove();
                    addMessage('ai', 'Не получилось получить ответ. Попробуйте ещё раз чуть позже.');
                })
                .then(function () {
                    busy = false;
                    syncSend();
                });
        }

        form.addEventListener('submit', function (e) {
            e.preventDefault();
            ask(input.value);
        });

        input.addEventListener('input', function () {
            autosize();
            syncSend();
        });

        input.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
                e.preventDefault();
                ask(input.value);
            }
        });

        suggest.addEventListener('click', function (e) {
            var btn = e.target.closest('.suggest');
            if (btn) ask(btn.textContent);
        });

        toggle.addEventListener('click', function () {
            setCollapsed(!chat.classList.contains('is-collapsed'));
        });

        // На невысоких экранах чат стартует свёрнутым, чтобы не сжимать панель рисков
        if (window.matchMedia('(min-width: 1024px) and (max-height: 819px)').matches) {
            setCollapsed(true);
        }
    }

    document.querySelectorAll('[data-feed]').forEach(initFeed);
    document.querySelectorAll('[data-tabs]').forEach(initTabs);
    document.querySelectorAll('[data-chat]').forEach(initChat);
})();
