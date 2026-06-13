import json
import os
import time
from rag import QueryProcessor
from rag_judge import RAGJudge
from datetime import datetime
from pathlib import Path
import logging

class RagTest:
    def __init__(self, data_set_path: str, target_dir: str):
        self.data_set_path = data_set_path
        self.target_dir = Path(target_dir)

        self.target_dir.mkdir(parents=True, exist_ok=True)

        current_date = datetime.now().strftime("%Y-%m-%d")
        self.RESULTS_PATH = self.target_dir / f"test_results_{current_date}.json"
        LOG_FILE_NAME = self.target_dir / f"test_log_{current_date}.log"

        self.logger = logging.getLogger("RAG_Test")
        self.logger.setLevel(logging.INFO)

        log_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        file_handler = logging.FileHandler(LOG_FILE_NAME, encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)

    def run_tests(self):
        if not os.path.exists(self.data_set_path):
            self.logger.error(f"Файл датасета {self.data_set_path} не найден!")
            return

        with open(self.data_set_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        processor = QueryProcessor()
        self.logger.info("QueryProcessor инициализирован")

        judge = RAGJudge()
        self.logger.info("RAGJudge инициализирован")

        total_tests = len(dataset)
        passed_tests = 0
        detailed_results = []

        self.logger.info(f"Всего тестов {total_tests}")
        self.logger.info("=" * 80)

        global_start_time = time.perf_counter()

        for index, case in enumerate(dataset, 1):
            case_id = case.get("id", f"case_{index}")
            query = case["query"]
            required = case.get("required_words", [])
            forbidden = case.get("forbidden_words", [])
            expected_answer = case.get("expected_answer", "")

            self.logger.info(f"Тест {index}/{total_tests} [{case_id}] | Запрос: '{query}'")

            start_time = time.perf_counter()

            answer = processor.process_query(query)

            time_run = time.perf_counter() - start_time

            self.logger.info("  Анализ ответа судьёй")
            judge_verdict = judge.evaluate(
                query=query,
                rag_answer=answer,
                expected_answer=expected_answer,
                required=required,
                forbidden=forbidden
            )

            passed = (
                    judge_verdict.get("required_fulfilled") == "yes" and
                    judge_verdict.get("forbidden_absent") == "yes" and
                    judge_verdict.get("similarity_score", 0) >= 4
            )

            if passed:
                passed_tests += 1
                status = "PASSED"
                self.logger.info(f" [ТЕСТ ПРОШЕЛ] Время: {time_run:.2f}с.")
            else:
                self.logger.info(f" [ТЕСТ НЕ ПРОШЕЛ] Время: {time_run:.2f}с.")
                status = "FAILED"
                self.logger.info(f"     Оценка схожести: {judge_verdict.get('similarity_score')}/5")
                self.logger.info(f"     Причина: {judge_verdict.get('reasoning')}")
            self.logger.info("-" * 80)

            case_result = {
                "id": case_id,
                "query": query,
                "status": status,
                "execution_time_seconds": round(time_run, 4),
                "expected_answer": expected_answer,
                "rag_answer": answer,
                "judge_verdict": judge_verdict
            }
            detailed_results.append(case_result)

        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        total_time = time.perf_counter() - global_start_time

        report = {
            "summary": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate_percent": round(success_rate, 2),
                "total_time_seconds": round(total_time, 2)
            },
            "test_cases": detailed_results
        }

        with open(self.RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)

        self.logger.info("="*30 + " ИТОГОВЫЙ ОТЧЕТ " + "="*30)
        self.logger.info(f"Всего тестов запущено : {total_tests}")
        self.logger.info(f"Успешно пройденных    : {passed_tests}")
        self.logger.info(f"Провалено тест-кейсов : {total_tests - passed_tests}")
        self.logger.info(f"Метрика успешности    : {success_rate:.2f}%")
        self.logger.info(f"Общее время тестов    : {total_time:.2f}с.")
        self.logger.info("=" * 80)

