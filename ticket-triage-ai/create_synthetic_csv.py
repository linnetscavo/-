"""Создание исключительно СИНТЕТИЧЕСКИХ учебных заявок."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from triage import COLUMNS, SEED

DEPARTMENTS = {
    "Учебная оплата": ["Оплата заказа", "платёж", "списание средств", "проверка счёта"],
    "Учебная доставка": ["Доставка заказа", "посылка", "маршрут доставки", "пункт выдачи"],
    "Учебное качество": ["Качество товара", "комплектация товара", "упаковка", "дефект изделия"],
    "Учебная IT-поддержка": ["Работа портала", "вход в кабинет", "страница сайта", "доступ к порталу"],
    "Учебные возвраты": ["Возврат товара", "заявление на возврат", "обмен изделия", "возврат заказа"],
}
URGENCY = {
    "низкий": ["Плановый вопрос без срочности.", "Нужна консультация, работа продолжается."],
    "стандартный": ["Обычный запрос в рабочем порядке.", "Требуется проверка по стандартному порядку."],
    "высокий": ["Часть операций недоступна, есть временный обход.", "Серьёзный сбой, работа частично нарушена."],
    "критический": ["Полная остановка операций, обходного решения нет.", "Массовая недоступность, нужна немедленная эскалация."],
}
COUNTS_PER_DEPARTMENT = {"низкий": 600, "стандартный": 1200, "высокий": 480, "критический": 120}


def generate():
    rng = np.random.default_rng(SEED)
    assignments = [(dept, priority) for dept in DEPARTMENTS
                   for priority, count in COUNTS_PER_DEPARTMENT.items() for _ in range(count)]
    rng.shuffle(assignments)
    rows = []
    for i, (dept, priority) in enumerate(assignments):
        choices = DEPARTMENTS[dept]
        item = rng.choice(choices[1:])
        urgency = rng.choice(URGENCY[priority])
        message = f"Синтетическое обращение: {item}. {urgency} Код учебного сценария {i:05d}."
        rows.append([f"SYN-{i:05d}", choices[0], message, dept, priority])
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["data_origin"] = "SYNTHETIC"
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/tickets_synthetic.csv"))
    args = parser.parse_args()
    frame = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig", lineterminator="\n")
    metadata = dict(mode="SYNTHETIC", seed=SEED, rows=len(frame), departments=list(DEPARTMENTS),
                    priority_counts=frame.priority.value_counts().to_dict(),
                    sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                    source="Шаблоны create_synthetic_csv.py; реальные обращения не использованы")
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SYNTHETIC:", args.output, frame.shape)


if __name__ == "__main__":
    main()
