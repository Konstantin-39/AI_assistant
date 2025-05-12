# Руководство по установке

Перед установкой Ollama PDF RAG убедитесь, что у вас есть:

1. Установлен Python 3.11
2. pip (установщик пакетов Python)
3. Установлен git
4. Ollama установлена ​​в вашей системе

## Установка Ollama

1. Посетите [сайт Ollama](https://ollama.ai), чтобы загрузить и установить приложение.
2. После установки извлеките необходимые модели:
   ```bash
   ollama pull gemma2  # or your preferred model
   ollama pull nomic-embed-text
   ```

## Установка Ollama PDF RAG

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/tonykipkemboi/ollama_pdf_rag.git
   cd ollama_pdf_rag
   ```

2. Создайте и активируйте виртуальную среду:
   ```bash
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Проверка установки

1. Запустить Ollama в фоновом режиме
2. Запустите приложение:
   ```bash
   python run.py
   ```
3. Откройте браузер `http://localhost:8501`


## Следующие шаги

- Чтобы начать использовать приложение, следуйте [краткому руководству](quickstart.md)
- Подробные инструкции по использованию см. в [руководстве пользователя](../user-guide/pdf-processing.md)
