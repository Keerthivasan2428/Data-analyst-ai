import json
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="AI Data Analyst & Visualisation Engine", layout="wide"
)

st.title("📊 AI Data Analyst with Schema & Visualisation Engine")
st.write(
    "Upload any dataset (CSV/Excel/JSON) to query rows, understand schema, and"
    " generate dynamic charts."
)

# Get API Key from Streamlit Secrets
try:
  api_key = st.secrets["GEMINI_API_KEY"]
except Exception as e:
  st.error("API Key missing from Streamlit Secrets.")
  api_key = None

uploaded_file = st.file_uploader(
    "Upload your dataset (CSV, Excel or JSON)", type=["csv", "xlsx", "json"]
)

if uploaded_file is not None:
  if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
  elif uploaded_file.name.endswith(".xlsx"):
    df = pd.read_excel(uploaded_file)
  else:
    df = pd.read_json(uploaded_file)

  st.subheader("📋 Dataset Preview & Schema")
  st.dataframe(df.head())

  st.info(
      f"Shape: {df.shape[0]} rows, {df.shape[1]} columns | Columns detected:"
      f" {list(df.columns)}"
  )

  user_query = st.text_input(
      "Ask a question (e.g., 'Show average bmi by diabetes outcome'):"
  )

  if user_query and api_key:
    with st.spinner("Analyzing data structure and generating insights..."):
      schema_info = df.dtypes.to_string()
      full_data_text = df.to_string()

      prompt = f"""
            You are an expert data analyst. 
            
            SCHEMA INFORMATION:
            {schema_info}
            
            ENTIRE DATASET:
            {full_data_text}
            
            USER QUESTION: {user_query}
            
            Analyze the dataset and return a VALID JSON object with this exact structure:
            {{
              "text_response": "Detailed text analysis answering the user question.",
              "is_chart_data": true or false,
              "chart_type": "bar" or "line" or "none",
              "x_column": "exact column name for X axis or null",
              "y_column": "exact column name for Y axis or null",
              "data": [
                {{"x_value": "Category1", "y_value": 100}},
                {{"x_value": "Category2", "y_value": 250}}
              ]
            }}
            Return ONLY valid JSON. No markdown code blocks, just raw JSON string.
            """

      # Direct REST API call to Gemini
      url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={api_key}"
      headers = {"Content-Type": "application/json"}
      payload = {"contents": [{"parts": [{"text": prompt}]}]}

      try:
        res = requests.post(url, headers=headers, json=payload)
        res_json = res.json()

        if "error" in res_json:
          st.error(f"API Error: {res_json['error'].get('message')}")
        else:
          raw_text = (
              res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
          )

          if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
          if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

          result_json = json.loads(raw_text.strip())

          st.success("### Analysis Result")
          st.write(result_json.get("text_response"))

          if result_json.get("is_chart_data") and result_json.get("data"):
            chart_df = pd.DataFrame(result_json["data"])
            if not chart_df.empty:
              chart_df = chart_df.set_index("x_value")
              st.subheader("📈 Generated Visualisation")

              chart_type = result_json.get("chart_type", "bar")
              if chart_type == "line":
                st.line_chart(chart_df)
              else:
                st.bar_chart(chart_df)

      except json.JSONDecodeError:
        st.warning(
            "The model responded with plain text instead of JSON. Here is the"
            " raw output:"
        )
        st.write(raw_text)
      except Exception as e:
        st.error(f"An error occurred: {e}")
