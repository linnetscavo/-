"""Локальный интерфейс диспетчера с журналом SQLite."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import joblib
from triage import TASKS


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("""CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP,
        mode TEXT NOT NULL, model_hash TEXT NOT NULL, proposal TEXT NOT NULL,
        final TEXT, resolved TEXT)""")
    return db


def recommend(db, bundle, model_hash, subject, body):
    text = (subject.strip() + " " + body.strip()).strip()
    if not text:
        raise ValueError("Введите тему или текст")
    proposal = {}
    for task in TASKS:
        model = bundle["models"][task]
        probs = model.predict_proba([text])[0]
        i = probs.argmax()
        proposal[task] = {"label": str(model.classes_[i]), "confidence": float(probs[i])}
    with db:
        row = db.execute("INSERT INTO decisions(mode,model_hash,proposal) VALUES (?,?,?)",
                         (bundle["info"]["mode"], model_hash, json.dumps(proposal, ensure_ascii=False)))
    return row.lastrowid, proposal


def resolve(db, ticket, final, labels):
    if set(final) != set(TASKS) or any(final[t] not in labels[t] for t in TASKS):
        raise ValueError("Проверьте выбранные метки")
    with db:
        cur = db.execute("UPDATE decisions SET final=?,resolved=CURRENT_TIMESTAMP WHERE id=? AND final IS NULL",
                         (json.dumps(final, ensure_ascii=False), ticket))
        if cur.rowcount != 1:
            raise ValueError("Решение уже сохранено или номер отсутствует")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--db", default="private/decisions.sqlite")
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    fingerprint = hashlib.sha256(args.model.read_bytes()).hexdigest()
    print("Режим:", bundle["info"]["mode"])
    db = connect(args.db)
    try:
        ticket, prediction = recommend(db, bundle, fingerprint, input("Тема: "), input("Текст: "))
        print("Номер рекомендации:", ticket)
        final = {}
        for task, title in zip(TASKS, ["Подразделение", "Приоритет"]):
            rec = prediction[task]
            options = bundle["labels"][task]
            print(f"{title}: {rec['label']} ({rec['confidence']:.1%})")
            print(" / ".join(options))
            while True:
                answer = input("Enter — подтвердить, или введите метку: ").strip() or rec["label"]
                if answer in options:
                    final[task] = answer
                    break
                print("Выберите метку из списка")
        resolve(db, ticket, final, bundle["labels"])
        print("Решение сохранено.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
