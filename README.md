# 🧠 Python Ollama Summarizer

Этот проект — это FastAPI-приложение, которое:

- 🔗 Принимает ссылку на статью (URL);
- 📄 Извлекает и очищает HTML-текст;
- 🧩 Разбивает на чанки при необходимости;
- 🤖 Отправляет каждый чанк в локальную LLM через [Ollama](https://ollama.com);
- 📊 Возвращает краткое summary и список тематических тегов.

---

## 🚀 Быстрый старт

### 1. Клонируй проект и перейди в директорию

```bash
git clone https://github.com/yourname/python-ollama-summarizer.git
cd python-ollama-summarizer
```

### 2. Установи зависимости

Создай и активируй виртуальное окружение:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Настрой `.env`

Создай `.env` файл в корне проекта:

```env
OLLAMA_API_URL=http://localhost:11434/api/generate
MODEL_NAME=gemma3:4b-it-qat
MAX_TOKENS=7500
MAX_QUEUE_SIZE=5
```

> 🔄 Ты можешь заменить модель, например, на `gemma3:1b-it-qat` или `mistral`.

### 4. Запусти Ollama и загрузи модель

```bash
ollama serve
ollama pull gemma3:4b-it-qat
```

Убедись, что Ollama работает на `http://localhost:11434`.

---

### 5. Запусти FastAPI-приложение

```bash
uvicorn main:app --reload
```

Открой документацию:
- [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📦 Структура проекта

```
app/
├── api/              # Эндпоинты FastAPI
│   └── summarize.py
├── core/             # Конфигурация и логгирование
│   ├── config.py
│   └── logging.py
├── db/               # Модель и подключение к БД
│   ├── database.py
│   └── models.py
├── schemas/          # Pydantic-схемы
│   └── summary.py
├── services/         # Бизнес-логика
│   ├── chunking.py
│   ├── ollama.py
│   └── summarize.py
main.py               # Точка входа
```

---

## 📤 Пример запроса

POST `/summarize`:

```json
{
  "url": "https://habr.com/ru/articles/909130/"
}
```

GET `/status/{request_id}`:

```json
{
  "status": "success",
  "result": {
    "summary": "Short summary of the article",
    "tags": ["ai", "deep-learning", "ollama"]
  }
}
```

---

## 📚 Зависимости

- `fastapi`
- `uvicorn`
- `requests`
- `readability-lxml`
- `beautifulsoup4`
- `sqlalchemy`
- `pydantic`
- `tiktoken`
- `python-dotenv`

Установи их через `pip install -r requirements.txt`.

---

## 🛠 Дополнительно

- Логи пишутся в `logs/summary.log`
- SQLite создаётся автоматически
- Поддержка чанкинга и повторного суммирования
- Если URL уже обрабатывался успешно — возвратится ошибка `409`

---
u