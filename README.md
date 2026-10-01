# 🍲 Lead Classifier — Laos Nationwide

A high-performance **Streamlit** application designed to clean, classify, and deduplicate F&B (Food & Beverage) sales leads scraped from Google Maps or Apify against a master database (Salesforce / Backend Accounts).

---

## 🚀 Key Features

- **Nationwide Laos Scope (`LA`):** Filters leads exclusively for Laos without splitting or fragmenting data by individual provinces.
- **Multi-Language Category Matching:** Pre-configured with F&B keywords in **English**, **Lao** (e.g., ຮ້ານອາຫານ, ຮ້ານກາເຟ, ເຝີ, ສິນດາດ), and **Chinese** (e.g., 烧烤, 珍珠奶茶店, 火锅).
- **High-Speed Hash Matching ($O(1)$ Complexity):** Instantly deduplicates incoming leads against large master datasets (80,000+ records) using pre-indexed hash sets for GRID IDs and standardized Lao phone numbers (+856 / 020 / 030).
- **Instant Export:** Download classified results directly as a clean CSV file.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/thongsouksouvankham-lang/Lao-Lead-Classification-tool.git](https://github.com/thongsouksouvankham-lang/Lao-Lead-Classification-tool.git)
   cd Lao-Lead-Classification-tool
