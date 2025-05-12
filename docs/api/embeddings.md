# Embeddings API

TНа этой странице описаны компоненты встраивания текста, используемые для семантического поиска.

## NomicEmbeddings

```python
class NomicEmbeddings:
    """Manages text embeddings using Nomic's embedding model."""
    
    def __init__(self, model_name: str = "nomic-embed-text"):
        """Initialize embeddings with model name."""
```

### Методы

#### embed_documents
```python
def embed_documents(self, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
```

Parameters:
- `texts`: List of text strings

Returns:
- List of embedding vectors

#### embed_query
```python
def embed_query(self, text: str) -> List[float]:
    """Generate embedding for a single query text."""
```

Parameters:
- `text`: Query text

Returns:
- Embedding vector

## Пример использования

```python
# Initialize embeddings
embeddings = NomicEmbeddings()

# Embed documents
docs = ["First document", "Second document"]
doc_embeddings = embeddings.embed_documents(docs)

# Embed query
query = "Sample query"
query_embedding = embeddings.embed_query(query)
```

## Конфигурация

Настраивание embeddings осуществляется с помощью:

- `Model selection`: Выбор модели
- `Batch size`: Размер batch
- `Normalization`: Нормализация
- `Caching options`: Параметры кэширования

## Производительность

Варианты оптимизации:

- `Batch processing`: Пакетная обработка
- `GPU acceleration`: ускорение графического процессора
- `Caching`: Кэширование
- `Dimensionality`: Размерность

## Лучшие практики

1. **Подготовка текста**
   - Предварительная очистка текста
   - Обработка специальных символов
   - нормализация длинны

2. **Управление ресурсами**
   - Batch одинаковой длинны
   - Мониторинг использования памяти
   - Кэшировать частые запросы

3. **Контроль качества**
   - Проверка embeddings
   - Проверка dimensions
   - Мониторинг оценок сходства
