# deciders

Сервер решений на [Cloudflare Clef](https://huggingface.co/Cloudflare/clef). Принимает состояние и вопросы, возвращает ответ с вероятностями. Генерации текста нет: один проход модели и голова схемы.

Одна видеокарта. Запускайте один сервис.

| Сервис | Как считается | Порт |
|---|---|---|
| `clef` | [Ollama](https://ollama.com/library/clef), модель `clef` | `127.0.0.1:$PORT` |
| `clef-flash` | `Cloudflare/clef-flash` через SGLang | `0.0.0.0:$PORT` |

## Что нужно

Docker с NVIDIA GPU. Для `clef` нужен Ollama 0.35.1 или новее: это версия образа `ollama/ollama`. Для `clef-flash` — доступ в Hugging Face.

На RTX 4500 Ada полный Clef влезает не целиком: часть слоёв остаётся на CPU, окно тега — 16384 токена. Запрос длиннее этого Ollama отклоняет, а не обрезает.

## Настройка

```bash
cp .env.example .env
```

В `.env`:

- `PORT` — порт на хосте. У `clef` он проброшен только на `127.0.0.1`, внутри контейнера Ollama слушает 11434. У `clef-flash` тот же номер открыт на всех интерфейсах.
- `CLEF_API_KEY` — ключ `clef-flash` (`Authorization: Bearer`). Ollama его не проверяет.
- `HF_TOKEN` — если Hub просит авторизацию для Flash. Раскомментируйте строку.

Окно, число запросов и доля памяти карты у Flash задаются в `clef-flash/config.yaml`. Образ SGLang обрезает запрос на 16384.

## Полный Clef

Том `ollama` должен уже существовать: в нём лежат скачанные веса. На новой машине:

```bash
docker volume create ollama
```

Запуск:

```bash
docker compose up -d clef
docker compose exec clef ollama pull clef
```

Повторный `pull` ничего не качает, если модель уже в томе. Первый запрос после старта грузит её в память и занимает десятки секунд. Следующие идут из уже загруженной модели, она держится 5 минут после последнего запроса.

Логи: `docker compose logs -f clef`. Остановка: `docker compose stop clef`.

## Clef Flash

Профиль `clef-flash` не входит в обычный `up`:

```bash
docker compose up -d clef-flash
```

`clef-flash/entrypoint.py` читает конфиг и заменяет себя на `sglang serve`. Одновременно обрабатывается один запрос.

Не поднимайте `clef` и `clef-flash` вместе: оба занимают видеокарту и хостовый `$PORT`.

## Запросы к полному Clef

`clef` — это Ollama. `clef-flash` отдаёт свой API SGLang.

У каждого вопроса должно быть поле `instructions`, в том числе у `score`. Картинки — base64, без приставки `data:`.

```bash
curl -s "http://127.0.0.1:${PORT}/v1/systemone" \
  -H "Content-Type: application/json" \
  -d '{"model":"clef","state":"Our checkout started returning errors and orders are blocked.","questions":{"department":{"type":"choice","instructions":"Which team should handle the message?","criteria":{"billing":"Payments or invoices","technical":"Bugs or outages"}},"urgency":{"type":"score","instructions":"How soon does this need attention?","criteria":["Can wait","This week","Today"]},"outage":{"type":"noul","instructions":"Is a service down?"}}}'
```

Ответы:

- `noul` — вероятность метки `true`;
- `choice` — выбранный вариант, уверенность и вероятности по `criteria`;
- `score` — математическое ожидание индекса, легенда и вероятности уровней `"0"`, `"1"`, …

`usage.output_tokens` равен 0.
