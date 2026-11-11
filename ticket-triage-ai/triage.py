"""Воспроизводимый эксперимент для двух независимых задач."""
from pathlib import Path
import hashlib
import json

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

SEED = 42
PRIORITIES = ["низкий", "стандартный", "высокий", "критический"]
TASKS = ["department", "priority"]
COLUMNS = ["id", "subject", "text", *TASKS]
ALIASES = dict(zip(["ID", "тема", "текст", "ответственное подразделение", "приоритет"], COLUMNS))


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def normalize_priorities(values):
    return values.str.lower().replace({
        "low": "низкий", "medium": "стандартный", "standard": "стандартный",
        "high": "высокий", "critical": "критический", "средний": "стандартный",
    })


def is_synthetic(frame):
    data = frame.rename(columns=ALIASES)
    markers = data["id"].astype(str).str.startswith("SYN-")
    if "data_origin" in data:
        markers = markers | data.data_origin.eq("SYNTHETIC")
    if markers.any() and not markers.all():
        raise ValueError("Нельзя смешивать синтетические и исторические строки")
    return bool(markers.all() and len(data))


def validate(frame):
    frame = frame.rename(columns=ALIASES)
    if frame.columns.has_duplicates or set(COLUMNS) - set(frame.columns):
        raise ValueError(f"Нужны однозначные столбцы {COLUMNS}")
    data = frame[COLUMNS].fillna("").astype(str).apply(lambda col: col.str.strip())
    if (data[["id", *TASKS]] == "").any().any() or data.id.duplicated().any():
        raise ValueError("Проверьте пустые ID, метки и повторяющиеся ID")
    data["x"] = (data.subject + " " + data.text).str.strip()
    if data.x.eq("").any():
        raise ValueError("Тема и текст одновременно пусты")
    data.priority = normalize_priorities(data.priority)
    if set(data.priority) != set(PRIORITIES) or data.department.nunique() < 2:
        raise ValueError("Нужны все четыре допустимых приоритета и минимум два подразделения")
    return data.sort_values("id").set_index("id")


def synthetic():
    """Только искусственные шаблоны, никаких клиентских данных."""
    rows = []
    for i in range(384):
        dept, item = [("Учебная сеть", "маршрутизатор"), ("Учебный склад", "накладная"),
                      ("Учебный доступ", "пароль")][i % 3]
        level = (i // 3) % 4
        symptom = ["плановый вопрос", "обычная задача", "частичный сбой", "полная остановка"][level]
        rows.append([f"SYN-{i:04d}", item, f"{symptom} учебный сценарий {i}", dept, PRIORITIES[level]])
    return pd.DataFrame(rows, columns=COLUMNS)


def split(data, demo, test_ids=None):
    size = 2000
    if demo and len(data) == 384:
        size = 64
    elif len(data) != 12000:
        raise ValueError("Ожидаются 12 000 строк либо 384 строки малого синтетического smoke test")
    if test_ids is None:
        pool, test = train_test_split(data, test_size=size, stratify=data.priority, random_state=SEED)
    else:
        ids = list(test_ids)
        if len(ids) != size or len(set(ids)) != size or set(ids) - set(data.index):
            raise ValueError("Некорректный список ID отложенного теста")
        test, pool = data.loc[ids], data.drop(ids)
    train, val = train_test_split(pool, test_size=0.2, stratify=pool.priority, random_state=SEED)
    for task in TASKS:
        if set(train[task]) != set(data[task]):
            raise ValueError(f"В train отсутствует класс {task}; проверьте редкие метки")
    for part in [val, test]:
        if "критический" not in part.priority.values:
            raise ValueError("Невозможно оценить recall: нет критических заявок")
    seen = set()
    for part in [train, val, test]:
        keys = set(part.x.str.casefold().str.split().str.join(" "))
        if seen & keys:
            raise ValueError("Повтор текста между выборками: проверьте дубли до эксперимента")
        seen.update(keys)
    return train, val, test


def score(model, part, task, labels):
    truth = part[task]
    pred = model.predict(part.x)
    metrics = {f"{avg}_f1": f1_score(truth, pred, labels=labels, average=avg, zero_division=0)
               for avg in ["macro", "weighted"]}
    matrix = pd.DataFrame(confusion_matrix(truth, pred, labels=labels), index=labels, columns=labels)
    if task == "priority":
        crit = matrix.loc["критический"]
        metrics["critical_recall"] = float(crit["критический"] / crit.sum())
    return metrics, matrix, pred


def experiment(frame, out, demo=False, test_ids=None):
    data = validate(frame)
    demo = demo or is_synthetic(frame)
    train, val, test = split(data, demo, test_ids)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    pd.concat([pd.Series(name, index=part.index, name="split")
               for name, part in zip(["train", "validation", "test"], [train, val, test])]).to_csv(out / "split.csv")
    data.loc[train.index, TASKS].apply(pd.Series.value_counts).fillna(0).to_csv(out / "train_counts.csv")
    models, labels, candidates, chosen = {}, {}, [], {}
    for task in TASKS:
        labels[task] = PRIORITIES if task == "priority" else sorted(train[task].unique())
        dummy = DummyClassifier(strategy="most_frequent").fit(train.x, train[task])
        best = None
        for c in [0.5, 2.0]:
            model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                  LogisticRegression(C=c, max_iter=1000, random_state=SEED))
            model.fit(train.x, train[task])
            metrics, _, _ = score(model, val, task, labels[task])
            candidates.append(dict(task=task, C=c, **metrics))
            rank = (metrics["macro_f1"], metrics.get("critical_recall", 0))
            if best is None or rank > best:
                best, chosen[task], models[task] = rank, c, model
        models["dummy_" + task] = dummy
    pd.DataFrame(candidates).to_csv(out / "validation.csv", index=False)
    scores = []
    for task in TASKS:
        for method, key in [("majority", "dummy_" + task), ("tfidf_lr", task)]:
            metrics, matrix, pred = score(models[key], test, task, labels[task])
            scores.append(dict(task=task, method=method, **metrics))
            matrix.rename_axis("actual").to_csv(out / f"matrix_{method}_{task}.csv")
            if task == "priority":
                wrong = test.priority.eq("критический") & (pred != test.priority)
                audit = test.loc[wrong, ["department", "priority"]].copy()
                audit["predicted"] = pred[wrong]
                audit.to_csv(out / f"missed_{method}.csv")
    scores = pd.DataFrame(scores)
    scores.to_csv(out / "test_metrics.csv", index=False)
    lr = scores[scores.method == "tfidf_lr"]
    passed = bool(lr.macro_f1.ge(0.72).all() and lr.critical_recall.dropna().iloc[0] >= 0.80)
    digest = hashlib.sha256(data.to_csv().encode()).hexdigest()
    info = dict(mode="SYNTHETIC" if demo else "historical", seed=SEED, selected_C=chosen,
                data_sha256=digest, acceptance=None if demo else passed)
    (out / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    joblib.dump(dict(models={t: models[t] for t in TASKS}, labels=labels, info=info), out / "model.joblib")
    return scores
