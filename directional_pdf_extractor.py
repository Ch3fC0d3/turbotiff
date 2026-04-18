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
    Extracts tables from a scanned PDF using GPT-4o Vision API.
    """
    print(f"Initializing GPT-4o Vision API for {pdf_path}...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please add it to your .env file.")

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
            "You are an expert well log curve parser. Your task is to extract tabular data from printed well logs. "
            "The data usually contains a Depth column followed by various curve measurements (like Gamma Ray, Resistivity, Porosity, etc). "
            "Return ONLY a raw, perfectly formatted CSV string. "
            "Do not include any Markdown tags like ```csv. "
            "Include standard column headers on the first line. The first column MUST be DEPTH. "
            "Do NOT include table titles or any text outside of the table itself. "
            "If the page does not contain curve table data, simply return the exact word: EMPTY"
        )

    for page_num in pages_list:
        print(f"Processing Page {page_num}...")
        page = doc[page_num]
        
        # Render page to an image
        # High DPI for better OCR (zoom 2)
        matrix = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=matrix)
        
        # Convert to base64
        img_bytes = pix.tobytes("jpeg")
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.0
        }
        
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            result_text = response.json()['choices'][0]['message']['content'].strip()
            
            if result_text == "EMPTY" or ("EMPTY" in result_text and len(result_text) < 10):
                print(f"  No survey data found on page {page_num}.")
                continue
                
            # Clean up potential markdown formatting if the model disobeys
            if result_text.startswith("```csv"):
                result_text = result_text[6:]
            elif result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
                
            result_text = result_text.strip()
            
            # Parse into pandas robustly (skip bad lines)
            df = pd.read_csv(io.StringIO(result_text), on_bad_lines='skip')
            
            if not df.empty:
                print(f"\n--- Found Table on Page {page_num} ---")
                print(_ascii_safe_preview(df))
                all_dataframes.append(df)
                
        except Exception as e:
            print(f"  Error parsing page {page_num}: {e}")

    if not all_dataframes:
        print("No tables detected by GPT-4o.")
        
    return all_dataframes

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
