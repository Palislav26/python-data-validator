import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Issue:
    row_index: Optional[int]   # None = dataset-level issue
    column: Optional[str]
    issue: str
    value: Any


def add_issue(issues: List[Issue], row_index: Optional[int], column: Optional[str], issue: str, value: Any):
    issues.append(Issue(row_index=row_index, column=column, issue=issue, value=value))


def validate_dataframe(
    df: pd.DataFrame,
    *,
    required_cols: List[str],
    expected_types: Dict[str, str],
    ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
    allowed_values: Dict[str, List[Any]],
    unique_key_cols: Optional[List[str]] = None,
    email_cols: Optional[List[str]] = None,
) -> Tuple[List[Issue], Dict[str, Any]]:
    issues: List[Issue] = []
    summary: Dict[str, Any] = {}

    # Required columns exist
    missing_required_cols = [c for c in required_cols if c not in df.columns]
    for c in missing_required_cols:
        add_issue(issues, None, c, "Missing required column", None)

    present_required_cols = [c for c in required_cols if c in df.columns]

    # Missing values in required columns
    for c in present_required_cols:
        missing_mask = df[c].isna()
        for idx in df.index[missing_mask]:
            add_issue(issues, int(idx), c, "Missing value", None)

    # Duplicate check
    if unique_key_cols:
        key_cols = [c for c in unique_key_cols if c in df.columns]
        for c in unique_key_cols:
            if c not in df.columns:
                add_issue(issues, None, c, "Missing key column for duplicate check", None)

        if key_cols:
            dup_mask = df.duplicated(subset=key_cols, keep=False)
            for idx in df.index[dup_mask]:
                add_issue(issues, int(idx), ",".join(key_cols), "Duplicate key", df.loc[idx, key_cols].to_dict())
    else:
        dup_mask = df.duplicated(keep=False)
        for idx in df.index[dup_mask]:
            add_issue(issues, int(idx), None, "Duplicate row (entire row)", None)

    # Type checks
    for col, type_name in expected_types.items():
        if col not in df.columns:
            continue

        non_missing = df[col].dropna()
        if type_name in ("int", "float"):
            converted = pd.to_numeric(non_missing, errors="coerce")
            bad = converted.isna()
        elif type_name == "datetime":
            converted = pd.to_datetime(non_missing, errors="coerce", utc=False)
            bad = converted.isna()
        elif type_name == "string":
            bad = non_missing.apply(lambda x: isinstance(x, (list, dict, set, tuple)))
        else:
            add_issue(issues, None, col, f"Unknown expected type '{type_name}'", None)
            continue

        for idx in bad[bad].index:
            add_issue(issues, int(idx), col, f"Type mismatch (expected {type_name})", df.at[idx, col])

    # Range checks
    for col, (min_val, max_val) in ranges.items():
        if col not in df.columns:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")

        if min_val is not None:
            bad = (numeric < min_val).fillna(False)
            for idx in df.index[bad]:
                add_issue(issues, int(idx), col, f"Value below min ({min_val})", df.at[idx, col])

        if max_val is not None:
            bad = (numeric > max_val).fillna(False)
            for idx in df.index[bad]:
                add_issue(issues, int(idx), col, f"Value above max ({max_val})", df.at[idx, col])

    # Allowed values
    for col, allowed in allowed_values.items():
        if col not in df.columns:
            continue
        bad = (~df[col].isin(allowed)) & (~df[col].isna())
        for idx in df.index[bad]:
            add_issue(issues, int(idx), col, f"Value not allowed (allowed={allowed})", df.at[idx, col])

    # Email check
    if email_cols:
        for col in email_cols:
            if col not in df.columns:
                continue
            s = df[col].dropna().astype(str)
            bad_mask = ~s.str.contains("@", na=False)
            for idx in s.index[bad_mask]:
                add_issue(issues, int(idx), col, "Invalid email (missing '@')", df.at[idx, col])

    summary["rows"] = int(len(df))
    summary["cols"] = int(len(df.columns))
    summary["total_issues"] = int(len(issues))
    summary["issue_types"] = pd.Series([i.issue for i in issues]).value_counts().to_dict() if issues else {}

    return issues, summary


def issues_to_df(issues: List[Issue]) -> pd.DataFrame:
    return pd.DataFrame([{
        "row_index": i.row_index,
        "column": i.column,
        "issue": i.issue,
        "value": i.value
    } for i in issues])

# Page UX/UI
st.set_page_config(page_title="Pandas Data Validator", layout="wide")
st.title("📄 Pandas Data Validator")
st.write("Drag & drop a CSV file and I’ll validate it with simple rules.")

uploaded = st.file_uploader("Drop your CSV here", type=["csv"])

with st.sidebar:
    st.header("Rules (edit these)")
    required_cols_text = st.text_input("Required columns (comma-separated)", "id,email,age,country")
    unique_key_cols_text = st.text_input("Unique key columns (comma-separated)", "id")

    st.subheader("Type expectations")
    st.caption("Format: column:type  (type = int/float/datetime/string)")
    expected_types_text = st.text_area("Expected types", "id:int\nage:int\nemail:string")

    st.subheader("Ranges")
    st.caption("Format: column:min:max  (leave min/max blank if not needed)")
    ranges_text = st.text_area("Ranges", "age:0:120")

    st.subheader("Allowed values")
    st.caption("Format: column=value1,value2,value3")
    allowed_values_text = st.text_area("Allowed values", "country=SK,CZ,AT,HU")

    email_cols_text = st.text_input("Email columns (comma-separated)", "email")


def parse_expected_types(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in [l.strip() for l in text.splitlines() if l.strip()]:
        if ":" not in line:
            continue
        col, t = line.split(":", 1)
        out[col.strip()] = t.strip()
    return out


def parse_ranges(text: str) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for line in [l.strip() for l in text.splitlines() if l.strip()]:
        parts = line.split(":")
        if len(parts) != 3:
            continue
        col, mn, mx = parts
        mn_val = float(mn) if mn.strip() != "" else None
        mx_val = float(mx) if mx.strip() != "" else None
        out[col.strip()] = (mn_val, mx_val)
    return out


def parse_allowed_values(text: str) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    for line in [l.strip() for l in text.splitlines() if l.strip()]:
        if "=" not in line:
            continue
        col, vals = line.split("=", 1)
        out[col.strip()] = [v.strip() for v in vals.split(",") if v.strip() != ""]
    return out


if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("Preview")
    st.dataframe(df.head(30), use_container_width=True)

    required_cols = [c.strip() for c in required_cols_text.split(",") if c.strip()]
    unique_key_cols = [c.strip() for c in unique_key_cols_text.split(",") if c.strip()]
    email_cols = [c.strip() for c in email_cols_text.split(",") if c.strip()]

    expected_types = parse_expected_types(expected_types_text)
    ranges = parse_ranges(ranges_text)
    allowed_values = parse_allowed_values(allowed_values_text)

    issues, summary = validate_dataframe(
        df,
        required_cols=required_cols,
        expected_types=expected_types,
        ranges=ranges,
        allowed_values=allowed_values,
        unique_key_cols=unique_key_cols if unique_key_cols else None,
        email_cols=email_cols if email_cols else None,
    )

    st.subheader("Summary")
    st.json(summary)

    st.subheader("Issues")
    if issues:
        issues_df = issues_to_df(issues)
        st.dataframe(issues_df, use_container_width=True)

        csv_bytes = issues_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download issues.csv", data=csv_bytes, file_name="issues.csv", mime="text/csv")
    else:
        st.success("No issues found ✅")

else:
    st.info("Upload a CSV to start.")

