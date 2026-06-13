import json
import re
from langchain_ollama import OllamaLLM

class RAGJudge:
    #model_name_l="IlyaGusev/saiga_llama3_8b"
    model_name_l="qwen2.5:7b"
    def __init__(self, model_name=model_name_l):

        self.llm = OllamaLLM(model=model_name, temperature=0.0)
        print("Модель успешно подключена!")

    def evaluate(self, query: str, rag_answer: str, expected_answer: str, required: list, forbidden: list) -> dict:

        prompt = f"""Ты — беспристрастный эксперт-аудитор RAG-систем. Твоя задача — оценить качество Реального Ответа бота.

Контекст теста:
- Исходный Запрос: "{query}"
- Эталонный ("Золотой") Ответ: "{expected_answer}"
- Список обязательных понятий/фактов: {required}
- Список запрещенных понятий/тем: {forbidden}

Реальный Ответ бота для оценки:
"{rag_answer}"

Инструкция по оценке:
1. Проверь, отражены ли в Реальном Ответе все обязательные факты (учитывай синонимы). Напиши "yes" или "no".
2. Проверь, отсутствуют ли запрещенные понятия. Напиши "yes" (если чисто и запрещенного нет) или "no" (если нашли запрещенку).
3. Выстави общую оценку схожести Реального Ответа с Эталонным по шкале от 1 до 5:
   5 — Идеально совпадает по смыслу и фактам.
   4 — Смысл верный, но упущены мелкие детали.
   3 — Ответ частично верен, но слишком размыт или неполный.
   2 — Ответ почти не связан с эталоном, содержит галлюцинации.
   1 — Полный бред, отказ отвечать или грубая ошибка.

Ответь СТРОГО в формате JSON без лишнего текста, Markdown-разметки или объяснений:
{{
  "required_fulfilled": "yes/no",
  "forbidden_absent": "yes/no",
  "similarity_score": 5,
  "reasoning": "Краткое пояснение оценки на русском языке"
}}
"""

        try:
            generated_text = self.llm.invoke(prompt)

            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return {"error": "Не удалось распарсить JSON из ответа модели", "raw_output": generated_text}

        except Exception as e:
            return {"error": f"Ошибка при обращении к Ollama: {str(e)}"}

