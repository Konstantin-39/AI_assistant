# LLM Manager API

На этой странице описаны компоненты управления языковой моделью.

## LLMManager

```python
class LLMManager:
    """Manages Ollama language model interactions."""
    
    def __init__(self, model_name: str = "gemma2"):
        """Initialize LLM manager with model name."""
```

### Методы

#### list_models
```python
def list_models() -> List[str]:
    """List available Ollama models."""
```

Returns:
- List of model names

#### get_model
```python
def get_model(self, model_name: str) -> LLM:
    """Get an instance of the specified model."""
```

Parameters:
- `model_name`: Name of the Ollama model

Returns:
- LLM instance

#### generate
```python
def generate(self, prompt: str, **kwargs) -> str:
    """Generate text using the current model."""
```

Parameters:
- `prompt`: Input text
- `**kwargs`: Additional generation parameters

Returns:
- Generated text

## Пример использования

```python
# Initialize manager
manager = LLMManager(model_name="gemma2")

# List available models
models = manager.list_models()

# Generate text
response = manager.generate(
    prompt="Explain RAG in simple terms",
    temperature=0.7,
    max_tokens=500
)
```

## Параметры модели

Настройте поведение модели с помощью:

- `temperature`: Креативность (0.0-1.0)
- `max_tokens`: Длина ответа
- `top_p`: Количества ответов
- `frequency_penalty`: Контроль повторения

## Обработка ошибок

- `Model loading errors`: Ошибки загрузки модели
- `Generation timeouts`: Тайм-ауты генерации
- `Resource constraints`: Ограничения ресурсов
- `API communication issues`: Проблемы с коммуникацией API

## Лучшие практики

1. **Выбор модели**
   - Сопоставьте модель с задачей
   - Рассмотрите использование ресурсов
   - Тест производительности

2. **Настройка параметров**
   - Отрегулируйте температуру
   - Длина контрольного ответа
   - Баланс качество/скорость
