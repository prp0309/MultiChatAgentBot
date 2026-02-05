import os
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import pandas as pd
from dotenv import load_dotenv


try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv() #loading environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DRY_RUN = (OPENAI_API_KEY == "") or (OpenAI is None) #safety switch to not crash



# Small utilities
def _safe_lower(x: Any) -> str:
    return str(x).strip().lower()
    
def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None
        
def _to_int(x: Any) -> Optional[int]:
    f = _to_float(x)
    if f is None:
        return None
    try:
        return int(round(f))
    except Exception:
        return None

def load_csv_flexible(path: str) -> pd.DataFrame:
    candidates: List[pd.DataFrame] = []
    for h in (0, 1, 2):
        try:
            df = pd.read_csv(path, header=h, low_memory=False)
            # Drop empty columns early
            df = df.dropna(axis=1, how="all")
            # Clean column names
            df.columns = [str(c).strip() for c in df.columns]
            candidates.append(df)
        except Exception:
            continue
            
    if not candidates:
        # Last resort: read without headers
        df = pd.read_csv(path, header=None, low_memory=False)
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
        return df
        
    def score(df: pd.DataFrame) -> int:
        non_null = int(df.notna().sum().sum())
        unnamed = sum(str(c).startswith("Unnamed") for c in df.columns) * 50
        col_bonus = int(len(df.columns) * 5)
        return non_null - unnamed + col_bonus        
    best = max(candidates, key=score)
    #lastcleanup
    best = best.dropna(axis=1, how="all")
    best.columns = [str(c).strip() for c in best.columns]
    return best

#function to write short summaries
def summarization(df: pd.DataFrame) -> str:
    df2 = df.copy()
    rows, cols = df2.shape
    num_cols = df2.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df2.columns if df2[c].dtype == "object"]

    summary_parts: List[str] = []
    summary_parts.append(f"I loaded {rows} rows and {cols} columns.")
    if cols:
        show_cols = ", ".join(df2.columns[:12])
        summary_parts.append(f"Columns look like: {show_cols}" + (" ..." if cols > 12 else ""))

    if num_cols:
        c = num_cols[0]
        summary_parts.append(
            f"One numeric column '{c}' ranges from {df2[c].min()} to {df2[c].max()}."
        )

    if cat_cols:
        c = cat_cols[0]
        top_vals = df2[c].astype(str).value_counts().head(3).to_dict()
        summary_parts.append(f"Top values in '{c}' look like {top_vals}.")

    return " ".join(summary_parts)
    
def llm_text(prompt: str, model: str = "gpt-4o-mini") -> str:
    if DRY_RUN:
        return "[DRY_RUN] Add OPENAI_API_KEY in .env to enable LLM summaries."
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "Be practical and grounded. Do not invent numbers."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

#NORMALIZATION
@dataclass
class NormalizedData:
    """
    Unifies two modes:
    - schema="sales":  date/amount/qty mapped
    - schema="general": providing useful Q&A and summaries
    """
    df: pd.DataFrame
    source_name: str
    schema: str  # "sales" or "general."

    @classmethod
    def from_csv(cls, path: str) -> "NormalizedData":
        raw = load_csv_flexible(path)
        name = os.path.basename(path)

        #trying normalization first
        sales_df = cls._try_normalize(raw)

        if sales_df is not None and len(sales_df) > 0:
            sales_df["source_file"] = name
            return cls(df=sales_df, source_name=name, schema="sales")

        #else falling back to general table
        general_df = raw.copy()
        general_df["source_file"] = name
        return cls(df=general_df, source_name=name, schema="generic")

    @staticmethod
    def _find_col(raw: pd.DataFrame, patterns: List[str]) -> Optional[str]:
        cols = list(raw.columns)
        cols_l = [_safe_lower(c) for c in cols]
        for pat in patterns:
            for c, cl in zip(cols, cols_l):
                if re.search(pat, cl):
                    return c
        return None

    @classmethod
    def _try_normalize(cls, raw: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        trying to map date/amount/qty, treating as sales-like.
        Otherwise, return None so the pipeline uses generic mode.
        """
        df = raw.copy()
        date_col = cls._find_col(df, [r"\bdate\b", r"order\s*date", r"transaction\s*date"])
        amt_col = cls._find_col(df, [r"\bamount\b", r"gross\s*amt", r"net\s*amt", r"\btotal\b", r"revenue"])
        qty_col = cls._find_col(df, [r"\bqty\b", r"\bquantity\b", r"\bpcs\b", r"\bunits?\b"])

        if amt_col is None or date_col is None:
            return None
        #mapping product,region
        product_col = cls._find_col(df, [r"sku", r"product", r"item", r"style", r"asin", r"description"])
        region_col = cls._find_col(df, [r"state", r"region", r"city", r"ship.*state", r"ship.*city"])

        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df[date_col], errors="coerce")
        out["amount"] = df[amt_col].apply(_to_float)

        if qty_col is not None:
            out["qty"] = df[qty_col].apply(_to_int)
        else:
            out["qty"] = None

        out["product"] = df[product_col].astype(str) if product_col is not None else "unknown"
        out["region"] = df[region_col].astype(str) if region_col is not None else "unknown"

        status_col = cls._find_col(df, [r"\bstatus\b", r"order\s*status"])
        if status_col is not None:
            out["status"] = df[status_col].astype(str)
        out = out.dropna(subset=["amount"], how="all")

        #if everything is none, assuming non--sales data
        if len(out) == 0:
            return None

        return out
        
#AGENT
class DataAgent:
    def __init__(self, data: NormalizedData):
        self.data = data

    def _clean_query(self, q: str) -> str:
        q = q.lower().strip()
        q = re.sub(r"\s+", " ", q)

        #simple WORD mapping so users don't need exact wording
        replacements = {
            "best region": "top region",
            "highest region": "top region",
            "best state": "top region",
            "highest state": "top region",
            "best city": "top region",
            "highest city": "top region",
            "most sales": "top",
            "highest sales": "top",
            "most revenue": "top",
            "highest revenue": "top",
        }
        for a, b in replacements.items():
            q = q.replace(a, b)

        return q

    def run(self, question: str) -> str:
        q = self._clean_query(question)
        df = self.data.df

        if self.data.schema == "sales":
            if "top" in q and "product" in q:
                g = df.groupby("product")["amount"].sum().sort_values(ascending=False)
                return f"Top product by revenue is {g.index[0]} with {float(g.iloc[0]):.2f}."

            if "top" in q and ("region" in q or "state" in q or "city" in q):
                g = df.groupby("region")["amount"].sum().sort_values(ascending=False)
                return f"Top region by revenue is {g.index[0]} with {float(g.iloc[0]):.2f}."

            if "revenue" in q and "region" in q:
                g = df.groupby("region")["amount"].sum().sort_values(ascending=False).head(10)
                lines = [f"{k}: {float(v):.2f}" for k, v in g.items()]
                return "Revenue by region:\n" + "\n".join(lines)

            if "revenue" in q and "product" in q:
                g = df.groupby("product")["amount"].sum().sort_values(ascending=False).head(10)
                lines = [f"{k}: {float(v):.2f}" for k, v in g.items()]
                return "Revenue by product:\n" + "\n".join(lines)

            if "total" in q and ("revenue" in q or "sales" in q or "amount" in q):
                return f"Total revenue is {float(df['amount'].fillna(0).sum()):.2f}."

            if "total" in q and ("qty" in q or "units" in q):
                if df["qty"].notna().any():
                    return f"Total units sold is {int(df['qty'].fillna(0).sum())}."
                return "This file does not have a quantity column mapped."

            return "I can answer totals, top product, top region, and revenue breakdowns for this sales dataset."

        #non-sales mode
        if "columns" in q or "schema" in q:
            return f"This file has {len(df)} rows and {len(df.columns)} columns. Columns: {', '.join(df.columns[:20])}" + (" ..." if len(df.columns) > 20 else "")

        if "shape" in q or "rows" in q:
            return f"Loaded {len(df)} rows and {len(df.columns)} columns."
            
        # Simple numeric column summary
        if "summary" in q or "stats" in q:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if not num_cols:
                return "This file has no obvious numeric columns to summarize."
            c = num_cols[0]
            return f"Quick stats for '{c}': min={df[c].min()}, max={df[c].max()}, mean={round(float(df[c].mean()),2)}"

        return "This file looks like a general data table. I can describe columns, shapes, and all basic numeric stats."

#summarization agents
class SummaryAgent:
    def __init__(self, data: NormalizedData):
        self.data = data

    def _sales_stats(self) -> Dict[str, Any]:
        df = self.data.df
        out: Dict[str, Any] = {
            "rows": int(len(df)),
            "date_min": str(df["date"].min().date()) if df["date"].notna().any() else None,
            "date_max": str(df["date"].max().date()) if df["date"].notna().any() else None,
            "total_revenue": round(float(df["amount"].fillna(0).sum()), 2) if df["amount"].notna().any() else None,
            "total_qty": int(df["qty"].fillna(0).sum()) if "qty" in df.columns and df["qty"].notna().any() else None,
        }

        if df["amount"].notna().any():
            by_region = df.groupby("region")["amount"].sum().sort_values(ascending=False).head(5)
            by_product = df.groupby("product")["amount"].sum().sort_values(ascending=False).head(5)
            out["top_regions"] = [{"region": k, "revenue": round(float(v), 2)} for k, v in by_region.items()]
            out["top_products"] = [{"product": k, "revenue": round(float(v), 2)} for k, v in by_product.items()]

        if "status" in df.columns and df["status"].notna().any():
            s = df["status"].astype(str).str.lower()
            out["cancel_rate"] = round(float((s == "cancelled").mean()), 4)

        return out

    def run(self, style: str = "short") -> str:
        if self.data.schema == "generic":
            base = summarization(self.data.df)

            
            if DRY_RUN:
                return base

            prompt = f"""
I already computed the key points from the data.
I want you to help me phrase this into a {style} summary that I can share with my team.
Please stick to the facts below and don’t add anything new.

Notes:
{base}
"""
            return llm_text(prompt)

        stats = self._sales_stats()

        if DRY_RUN:
            # Simple deterministic summary without LLM
            parts = [f"Rows: {stats['rows']}."]
            if stats["date_min"] and stats["date_max"]:
                parts.append(f"Range: {stats['date_min']} to {stats['date_max']}.")
            if stats["total_revenue"] is not None:
                parts.append(f"Total revenue: {stats['total_revenue']}.")
            if stats.get("top_products"):
                parts.append(f"Top product: {stats['top_products'][0]['product']}.")
            if stats.get("top_regions"):
                parts.append(f"Top region: {stats['top_regions'][0]['region']}.")
            return " ".join(parts)

        prompt = f"""
I’ve pulled out the important numbers from this dataset.
Help me turn this into a {style} summary that sounds natural when I explain it to someone.
Only use the facts below and don’t make up anything.

Facts:
{json.dumps(stats, indent=2)}
"""
        return llm_text(prompt)

#summarization-numerical Q&A
class RouterAgent:
    def __init__(self, data_agent: DataAgent, summary_agent: SummaryAgent):
        self.data_agent = data_agent
        self.summary_agent = summary_agent

    def _normalize(self, text: str) -> str:
        t = text.lower().strip()
        t = re.sub(r"\s+", " ", t)
        return t

    def route(self, user_input: str) -> Tuple[str, str]:
        text = self._normalize(user_input)

        summary_keywords = [
            "summarize", "summary", "report", "overall",
            "insight", "insights", "overview", "high level", "brief"
        ]

        if any(k in text for k in summary_keywords):
            style = "short"
            if "detailed" in text or "long" in text:
                style = "detailed"
            return ("SummaryAgent", self.summary_agent.run(style=style))

        return ("DataAgent", self.data_agent.run(user_input))
        
