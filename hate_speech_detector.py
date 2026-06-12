from transformers import pipeline

class HateSpeechDetector:
    def __init__(self):
        print("Загрузка локальной модели детектора токсичности...")

        model_name = "cointegrated/rubert-tiny-toxicity"

        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=-1
        )
        print("Модель hate-speech успешно загружена локально!\n" + "="*50)

    def analyze(self, text: str) -> dict:
        #results = self.classifier(text, top_k=None)
        results = self.classifier(text, top_k=None, truncation=True, max_length=512)
        if results and isinstance(results, list) and isinstance(results[0], list):
            results = results[0]
        scores = {item['label']: item['score'] for item in results}
        return scores

    def is_safe(self, text: str, threshold: float = 0.5) -> bool:
        scores = self.analyze(text)
        is_polite = scores.get('non-toxic', 1.0) >= threshold
        #is_polite = scores.get('neutral', 1.0) >= threshold
        has_toxic_elements = (
                scores.get('dangerous', 0.0) > threshold or
                scores.get('insult', 0.0) > threshold or
                scores.get('obscenity', 0.0) > threshold or
                scores.get('threat', 0.0) > threshold
        )
        return is_polite and not has_toxic_elements

