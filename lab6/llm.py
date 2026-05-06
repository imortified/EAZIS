import requests
from config import OLLAMA_URL, MODEL_NAME


def generate_response(query, context, history=None):
    system_prompt = """Ты — полезный ассистент по теме \"Досуг\" на русском языке.
Твоя задача — отвечать на вопросы пользователя, опираясь на предоставленный контекст из документов.
Если в контексте нет ответа на вопрос, честно скажи об этом.
Отвечай кратко, по существу и на русском языке."""

    prompt = f"{system_prompt}\n\nКонтекст из документов:\n{context}\n\n"
    if history:
        prompt += "История диалога:\n"
        for msg in history[-6:]:
            role = "Пользователь" if msg['role'] == 'user' else "Ассистент"
            prompt += f"{role}: {msg['content']}\n"
        prompt += "\n"
    prompt += f"Пользователь: {query}\nАссистент:"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_ctx": 4096, "top_p": 0.9}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return data.get('response', '[Ошибка: пустой ответ от модели]').strip()
    except requests.exceptions.ConnectionError:
        return "[Ошибка: не удалось подключиться к Ollama. Убедитесь, что сервер запущен на localhost:11434]"
    except Exception as e:
        return f"[Ошибка при генерации ответа: {str(e)}]"
