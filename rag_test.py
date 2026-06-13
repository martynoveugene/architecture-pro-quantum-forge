import json
import os
import time
from rag import QueryProcessor

DATASET_PATH = "task7/dataset.json"

def run_test():
    if not os.path.exists(DATASET_PATH):
        print(f"Файл датасета {DATASET_PATH} не найден!")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    processor = QueryProcessor()
    print("QueryProcessor инициализирован")

    total_tests = len(dataset)
    passed_tests = 0

    print(f"\nВсего тестов {total_tests}")
    print("=" * 80)

    for index, case in enumerate(dataset, 1):
        case_id = case.get("id", f"case_{index}")
        query = case["query"]
        required = case.get("required_words", [])
        forbidden = case.get("forbidden_words", [])

        print(f"Тест {index}/{total_tests} [{case_id}] | Запрос: '{query}'")

        start_time = time.perf_counter()

        answer = processor.process_query(query)

        time_run = time.perf_counter() - start_time

        answer_lower = answer.lower()

        missing_words = [word for word in required if word.lower() not in answer_lower]

        found_forbidden = [word for word in forbidden if word.lower() in answer_lower]

        passed = len(missing_words) == 0 and len(found_forbidden) == 0

        if passed:
            passed_tests += 1
            print(f" [ТЕСТ ПРОШЕЛ] Время: {time_run:.2f}с.")
        else:
            print(f" [ТЕСТ НЕ ПРОШЕЛ] Время: {time_run:.2f}с.")
            if missing_words:
                print(f"   Отсутствуют обязательные слова: {missing_words}")
            if found_forbidden:
                print(f"   Обнаружены запрещенные слова: {found_forbidden}")
            print(f"     Ответ бота:\n {answer}")
        print("-" * 80)

    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

    print("\n" + "="*30 + " ИТОГОВЫЙ ОТЧЕТ " + "="*30)
    print(f"Всего тестов запущено : {total_tests}")
    print(f"Успешно пройденных    : {passed_tests}")
    print(f"Провалено тест-кейсов : {total_tests - passed_tests}")
    print(f"Метрика успешности    : {success_rate:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    run_test()
