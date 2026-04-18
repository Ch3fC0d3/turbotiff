import os
import io
import base64
import requests
import pandas as pd
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

def _ascii_safe_preview(df: pd.DataFrame, rows: int = 5) -> str:
    """Render OCR previews without crashing on a Windows charmap console."""
    preview = df.head(rows).to_string()
    return preview.encode("ascii", errors="backslashreplace").decode("ascii")

def extract_survey_from_pdf(pdf_path: str, pages_list=None, is_curve=False):
    """
    Extracts tables from a scanned PDF using Gemini API.
    """
    print(f"Initializing Gemini Vision API for {pdf_path}...")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please add it to your .env file.")

    model_id = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
    model_name = model_id if model_id.startswith("models/") else f"models/{model_id}"

    # Load PDF
    doc = fitz.open(pdf_path)
    if pages_list is None:
        pages_list = list(range(len(doc)))

    all_dataframes = []

    system_prompt = (
        "You are an expert directional survey parser. Your task is to extract tabular data from directional well surveys. "
        "The survey typically contains columns like Measured Depth (MD), Inclination (INC), Azimuth (AZI), "
        "True Vertical Depth (TVD), +N/-S, +E/-W. "
        "Return ONLY a raw, perfectly formatted CSV string. "
        "Do not include any Markdown tags like ```csv. "
        "Include standard column headers on the first line (e.g. MD, INC, AZI, TVD, N/S, E/W). "
        "Do NOT include table titles or any text outside of the table itself. "
        "If the page does not contain survey data, simply return the exact word: EMPTY"
    )

    if is_curve:
        system_prompt = (
            "You are an expert well log curve parser. Your task is to extract tabular data from digitized well curves. "
            "The data typically contains a Depth column and one or more curve columns (like GR, RES, DEN, NEU, etc). "
            "Return ONLY a raw, perfectly formatted CSV string. "
            "Do not include any Markdown tags like ```csv. "
            "Include standard column headers on the first line. "
            "Do NOT include table titles or any text outside of the table itself. "
            "If the page does not contain curve data, simply return the exact word: EMPTY"
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}"

    for page_num in pages_list:
        if page_num < 0 or page_num >= len(doc):
            print(f"Skipping invalid page number: {page_num}")
            continue

        page = doc.load_page(page_num)
        
        # Render page to image (DPI 150 is usually enough for OCR)
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("jpeg")
        base64_image = base64.b64encode(img_bytes).decode("utf-8")

        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": base64_image
                            }
                        },
                        {
                            "text": "Extract the table from this image as a CSV."
                        }
                    ]
                }
            ]
        }

        print(f"Processing page {page_num + 1} / {len(doc)} via Gemini...")
        
        try:
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
            
            if response.status_code != 200:
                print(f"Gemini API error on page {page_num + 1}: {response.status_code} - {response.text}")
                continue

            response_json = response.json()
            
            candidates = response_json.get("candidates", [])
            if not candidates:
                print(f"No candidates returned for page {page_num + 1}")
                continue
                
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                continue
                
            csv_text = parts[0].get("text", "").strip()

            # Clean up markdown formatting if the model ignored instructions
            if csv_text.startswith("```csv"):
                csv_text = csv_text[6:]
            if csv_text.startswith("```"):
                csv_text = csv_text[3:]
            if csv_text.endswith("```"):
                csv_text = csv_text[:-3]
            
            csv_text = csv_text.strip()

            if csv_text.upper() == "EMPTY" or not csv_text:
                print(f"Page {page_num + 1} is empty or contains no relevant data.")
                continue

            # Read the CSV text into a DataFrame
            df = pd.read_csv(io.StringIO(csv_text))
            if not df.empty:
                print(f"Successfully extracted {len(df)} rows from page {page_num + 1}")
                print(_ascii_safe_preview(df))
                all_dataframes.append(df)
            else:
                print(f"DataFrame for page {page_num + 1} was empty.")

        except Exception as e:
            print(f"Failed to process page {page_num + 1}: {e}")

    if not all_dataframes:
        return None

    # Combine all pages
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df

if __name__ == "__main__":
    import sys
    test_pdf = sys.argv[1] if len(sys.argv) > 1 else "sample_survey.pdf"
    
    if os.path.exists(test_pdf):
        dfs = extract_survey_from_pdf(test_pdf)
        if dfs:
            print("\nSuccessfully pulled table into a Pandas DataFrame!")
            dfs[0].to_csv("extracted_table_preview.csv", index=False)
            print("Saved the first table preview as extracted_table_preview.csv")
    else:
        print(f"File '{test_pdf}' not found. Please provide a valid file path.")
