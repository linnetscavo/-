"""Проверка CSV до обучения без автоматического удаления строк."""
import pandas as pd
from triage import ALIASES, COLUMNS, PRIORITIES, normalize_priorities


def inspect_data(frame, departments=None, min_chars=20):
    data = frame.rename(columns=ALIASES).copy()
    if data.columns.has_duplicates or set(COLUMNS) - set(data.columns):
        raise ValueError(f"Проверьте однозначные столбцы {COLUMNS}")
    clean = data[COLUMNS].fillna("").astype(str).apply(lambda col: col.str.strip())
    clean.priority = normalize_priorities(clean.priority)
    combined = (clean.subject + " " + clean.text).str.strip()
    keys = combined.str.casefold().str.split().str.join(" ")
    flags = pd.DataFrame(index=data.index)
    flags["empty_id"] = clean.id.eq("")
    flags["duplicate_id"] = clean.id.duplicated(keep=False)
    flags["duplicate_row"] = clean.duplicated(keep=False)
    flags["duplicate_text"] = keys.duplicated(keep=False)
    flags["empty_subject"] = clean.subject.eq("")
    flags["empty_text"] = clean.text.eq("")
    flags["empty_input"] = combined.eq("")
    flags["short_input"] = combined.str.len().between(1, min_chars - 1)
    flags["unknown_priority"] = ~clean.priority.isin(PRIORITIES)
    flags["empty_department"] = clean.department.eq("")
    if departments is not None:
        flags["unknown_department"] = ~clean.department.isin(departments)
    issues = flags.copy()
    issues.insert(0, "id", clean.id)
    issues = issues.loc[flags.any(axis=1)]
    fields = pd.DataFrame({"dtype": data[COLUMNS].dtypes.astype(str),
                           "null_count": data[COLUMNS].isna().sum(),
                           "blank_count": clean[COLUMNS].eq("").sum()})
    fields["blank_share"] = fields.blank_count / len(data) if len(data) else 0.0
    return dict(shape=data.shape, fields=fields, counts=flags.sum(), issues=issues,
                priority_counts=clean.priority.value_counts(dropna=False),
                department_counts=clean.department.value_counts(dropna=False),
                text_lengths=combined.str.len().describe(),
                department_dictionary_checked=departments is not None)
