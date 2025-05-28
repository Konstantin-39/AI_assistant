# Локальный AI ассистент: поиск и анализ корпоративных данных с RAG и LLM

Локальное приложение RAG (Retrieval Augmented Generation), позволяющее работать с данными внутри локальной сети с помощью Ollama и LangChain. Работа осуществляется в веб-интерфейсе Streamlit для простого взаимодействия с документацией.

## Структура проекта
```
ollama_rag/
├── src/                      # Исходный код
│   ├── app/                  # Streamlit
│   │   ├── components/       # Интерфейс приложения
│   │   │   ├── chat.py       # Интерфейс чата
│   │   │   ├── pdf_viewer.py # PDF
│   │   │   └── sidebar.py    # Элементы управления
│   │   └── main.py           # Main app
│   └── core/                 # core
│       ├── document.py       # Обработка документов
│       ├── embeddings.py     # Vector embeddings
│       ├── llm.py            # LLM setup
│       └── rag.py            # RAG pipeline
├── data/                     # Data
│   ├── pdfs/                 # PDF-хранилище
│   │   └── sample/           # PDFs-файлы
│   └── vectors/              # Vector DB storage
├── docs/                     # Документация
└── run.py                    # Запуск приложения
```


## Процесс запуска приложения

1. **Установить Ollama**
   - Посетите [сайт Ollama](https://ollama.ai) для загрузки и установки
   - Загрузите необходимые модели:
     ```bash
     ollama pull gemma2  # или предпочитаемая модель
     ollama pull nomic-embed-text
     ```

2. **Клонировать репозиторий**
   ```bash
   git clone https://github.com/Konstantin-39/AI_assistant
   ```

3. **Настройка среды**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

   Ключевые зависимости и их версии:
   ```txt
   ollama==0.4.4
   streamlit==1.40.0
   pdfplumber==0.11.4
   langchain==0.1.20
   langchain-core==0.1.53
   langchain-ollama==0.0.2
   chromadb==0.4.22
   ```

### Запуск приложения


```bash
python run.py
```
Затем откройте браузер `http://localhost:8501`