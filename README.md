# Multi-Agent Sales Data Chatbot

This is a small multi-agent chatbot I built to explore how LLMs and simple agent routing can work together on CSV data. The app lets me load noisy sales or finance files, ask questions in plain English, and get either direct numeric answers or short summaries. Under the hood, I keep the math deterministic and only use the LLM for phrasing when it actually helps.

---

## What this app does (at a glance)

You can load a CSV file, either from a preset folder or by uploading one.  
You can ask questions like “best region”, “total revenue”, or “give me insights”.  
The app routes the query to the right agent, computes answers using pandas, and optionally uses an LLM to phrase a summary.  
It runs locally using Streamlit and works even without an API key in a dry-run mode.

---

## Requirements

### Hardware
A normal laptop is enough. I tested this on a MacBook.  
No GPU is required since all heavy lifting is done with pandas and small prompts.

### Software
Python 3.10 or 3.11  
pip  
Homebrew (on macOS)  
CMake (needed if native deps like pyarrow are pulled in)  
Streamlit  

### Python dependencies
All Python dependencies are listed in `requirements.txt`.

---

## Setup and installation

### 1. System setup (macOS)

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install CMake (needed for some Python wheels like pyarrow)
brew install cmake

###  Setting up Python Environment
cd multi_agent_sales_demo
python3 -m venv .venv
source .venv/bin/activate

###Installing Python dependencies
pip install -r requirements.txt

###RUN THE APP
streamlit run app.py
###if streamlit not found
streamlit -m run app.py

###FROM VSCODE
Activate the .venv in the terminal.
Run:
python -m streamlit run app.py

---
```
## Assumptions

The CSV has headers and at least some structured columns.
If the file looks like sales data, the app tries to normalize it into date, amount, region, and product.
If it does not look like sales data, the app switches to a generic mode and still produces a fallback summary.
The LLM is not trusted for calculations. All numbers come from pandas.

---
## Limitations

This is not a production-grade system.
The intent routing is keyword-based and not learned.
Large CSVs will work, but the UI can feel slow since everything runs locally.
No persistent storage or indexing is used yet. Everything is in memory.

---
## Possible improvements

Add embeddings and a vector index for document-level search.
Add caching for repeated queries.
Improve intent detection using a small classifier instead of keywords.
Support Excel files and PDFs.
Add basic authentication if this is ever exposed outside local use.

---
