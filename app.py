import io
import json
import math
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import phonenumbers
import streamlit as st

# Attempt optional library imports with safe fallbacks
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# --- Page Configuration ---
st.set_page_config(
    page_title="LA Lead Classifier",
    page_icon="🍲",
    layout="wide"
)

# --- F&B Keyword Dictionary (EN, LO, ZH) ---
FB_KEYWORDS = {
    "English": [
        "restaurant", "cafe", "coffee", "baking", "bakery", "bar", "bistro", "pub", 
        "tea", "boba", "bubble tea", "bbq", "barbecue", "grill", "hotpot", "noodle", 
        "pizza", "burger", "sushi", "dining", "eatery", "food", "kitchen", "lounge"
    ],
    "Lao": [
        "ຮ້ານອາຫານ", "ຮ້ານກາເຟ", "ກາເຟ", "ຮ້ານກິນດື່ມ", "ເຂົ້າ", "ເຝີ", 
        "ປິ້ງ", "ສິນດາດ", "ຊາໂມກຕາ", "ຂະຫນົມປັງ", "ຂອງຫວານ", "ເຄື່ອງດື່ມ", "ອາຫານທະເລ"
    ],
    "Chinese": [
        "餐厅", "饭店", "餐馆", "咖啡", "咖啡厅", "茶", "奶茶", "珍珠奶茶", 
        "烧烤", "火锅", "面馆", "小吃", "甜品", "酒馆", "酒吧", "海鲜", "美食"
    ]
}

# --- Helper Functions ---
def normalize_la_phone(phone: Any) -> str:
    """
    Standardizes Laos phone numbers to a clean national number string 
    (stripping country code +856, leading zeros, and formatting characters).
    """
    if pd.isna(phone) or not phone:
        return ""
    
    clean_str = re.sub(r"[^\d+]", "", str(phone))
    
    try:
        parsed = phonenumbers.parse(clean_str, "LA")
        if phonenumbers.is_valid_number(parsed):
            return str(parsed.national_number)
    except Exception:
        pass

    digits = re.sub(r"\D", "", clean_str)
    if digits.startswith("856"):
        digits = digits[3:]
    digits = digits.lstrip("0")
    return digits if 7 <= len(digits) <= 10 else ""

def is_fb_category(category: Any) -> bool:
    """Checks if the category string matches any EN, LO, or ZH F&B keyword."""
    if pd.isna(category) or not category:
        return False
    
    text = str(category).lower()
    for keywords in FB_KEYWORDS.values():
        for kw in keywords:
            if kw.lower() in text:
                return True
    return False

def classify_and_dedupe(scraped_df: pd.DataFrame, master_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Cleans, classifies F&B status, and deduplicates against master data.
    """
    df = scraped_df.copy()
    
    grid_col = next((c for c in df.columns if "grid" in c.lower() or "place_id" in c.lower()), None)
    phone_col = next((c for c in df.columns if "phone" in c.lower() or "tel" in c.lower()), None)
    cat_col = next((c for c in df.columns if "cat" in c.lower() or "type" in c.lower()), None)

    df["_clean_grid"] = df[grid_col].astype(str).str.strip() if grid_col else ""
    df["_clean_phone"] = df[phone_col].apply(normalize_la_phone) if phone_col else ""

    master_grid_set = set()
    master_phone_set = set()

    if master_df is not None and not master_df.empty:
        m_grid_col = next((c for c in master_df.columns if "grid" in c.lower() or "place_id" in c.lower()), None)
        m_phone_col = next((c for c in master_df.columns if "phone" in c.lower() or "tel" in c.lower()), None)

        if m_grid_col:
            master_grid_set = set(master_df[m_grid_col].dropna().astype(str).str.strip())
        if m_phone_col:
            master_phone_set = set(master_df[m_phone_col].apply(normalize_la_phone).dropna())
            master_phone_set.discard("")

    def evaluate_row(row):
        category_val = row[cat_col] if cat_col else ""
        is_fb = is_fb_category(category_val)
        
        grid_match = row["_clean_grid"] in master_grid_set if row["_clean_grid"] else False
        phone_match = row["_clean_phone"] in master_phone_set if row["_clean_phone"] else False
        is_duplicate = grid_match or phone_match

        if not is_fb:
            status = "Non-F&B"
        elif is_duplicate:
            status = "Duplicate (Master)"
        else:
            status = "Qualified Lead"

        dedupe_reason = []
        if grid_match: dedupe_reason.append("GRID Match")
        if phone_match: dedupe_reason.append("Phone Match")
        
        return pd.Series([
            status, 
            is_fb, 
            is_duplicate, 
            ", ".join(dedupe_reason) if dedupe_reason else "None",
            row["_clean_phone"]
        ])

    df[["Lead_Status", "Is_FB", "Is_Duplicate", "Dedupe_Reason", "Normalized_Phone"]] = df.apply(evaluate_row, axis=1)
    df.drop(columns=["_clean_grid", "_clean_phone"], inplace=True)
    return df

# --- Streamlit UI Setup ---
st.title("🍲 LA Lead Classifier — Nationwide Laos")
st.markdown("Clean, classify, and deduplicate scraped F&B leads against master data.")

st.sidebar.header("📁 Upload Datasets")
scraped_file = st.sidebar.file_uploader("Upload Scraped Leads (CSV/XLSX)", type=["csv", "xlsx"])
master_file = st.sidebar.file_uploader("Upload Master Dataset (Optional CSV/XLSX)", type=["csv", "xlsx"])

if scraped_file:
    scraped_df = pd.read_csv(scraped_file) if scraped_file.name.endswith(".csv") else pd.read_excel(scraped_file)
    master_df = None
    if master_file:
        master_df = pd.read_csv(master_file) if master_file.name.endswith(".csv") else pd.read_excel(master_file)

    st.subheader("Raw Data Preview")
    st.dataframe(scraped_df.head(5), use_container_width=True)

    if st.button("⚡ Process & Classify Leads", type="primary"):
        with st.spinner("Processing leads with O(1) hash matching..."):
            processed_df = classify_and_dedupe(scraped_df, master_df)
            
        st.success("Classification Complete!")

        col1, col2, col3, col4 = st.columns(4)
        total_leads = len(processed_df)
        qualified = len(processed_df[processed_df["Lead_Status"] == "Qualified Lead"])
        duplicates = len(processed_df[processed_df["Lead_Status"] == "Duplicate (Master)"])
        non_fb = len(processed_df[processed_df["Lead_Status"] == "Non-F&B"])

        col1.metric("Total Input", total_leads)
        col2.metric("Qualified Leads", qualified)
        col3.metric("Master Duplicates", duplicates)
        col4.metric("Filtered Non-F&B", non_fb)

        st.subheader("📥 Export Data")
        csv_data = processed_df.to_csv(index=False).encode('utf-8')
        qualified_only_csv = processed_df[processed_df["Lead_Status"] == "Qualified Lead"].to_csv(index=False).encode('utf-8')

        d_col1, d_col2 = st.columns(2)
        d_col1.download_button("Download All Results (CSV)", csv_data, "classified_leads_all_la.csv", "text/csv")
        d_col2.download_button("Download Qualified Leads Only (CSV)", qualified_only_csv, "qualified_fb_leads_la.csv", "text/csv")

        st.subheader("Classified Results Preview")
        st.dataframe(processed_df, use_container_width=True)
else:
    st.info("Please upload a scraped dataset in the sidebar to begin.")
