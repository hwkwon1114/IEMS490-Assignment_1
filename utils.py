import os
import pandas as pd
import re
from langchain_google_genai import ChatGoogleGenerativeAI

def setup_api_key():
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = "YOUR_KEY_HERE"
        # Load Dataset from Pandas

def data_sample():

    splits = {'train': 'main/train-00000-of-00001.parquet', 'test': 'main/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])
    df_small = df.sample(n=150, random_state=32)
    df_small = df_small.reset_index(drop=True)
    return df_small

def load_data_for_part3(dev_size=40, test_size=150, random_state=32):

    try:
        print("Loading full GSM8K dataset for Part 3 split...")
        splits = {'train': 'main/train-00000-of-00001.parquet'}
        full_df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])
        
        # Create the development set for optimization
        dev_set = full_df.sample(n=dev_size, random_state=random_state)
        
        # Create the test set from the remaining data to ensure no overlap
        remaining_df = full_df.drop(dev_set.index)
        test_set = remaining_df.sample(n=test_size, random_state=random_state)
        
        print(f"Created disjoint development set ({len(dev_set)} samples) and test set ({len(test_set)} samples).")
        
        return dev_set.reset_index(drop=True), test_set.reset_index(drop=True)
        
    except Exception as e:
        print(f"Error loading data for Part 3: {e}")
        return pd.DataFrame(), pd.DataFrame()



def parser(text):
    try:
        # 1. Isolate the part after the prefix
        answer_prefix = "####"
        answer_part = text.split(answer_prefix)[-1]

        # 2. Clean the string
        #    This removes commas, $, spaces, and everything
        #    except digits and one decimal point.
        cleaned_string = re.sub(r'[^\d.]', '', answer_part)
        
        # 3. Handle cases where the string is empty after cleaning
        if not cleaned_string:
            return None

        # 4. Convert to a float, with error handling
        return float(cleaned_string)
        
    except (IndexError, ValueError):
        # IndexError happens if .split() fails (no "####")
        # ValueError happens if float() fails on a bad string
        return None
    

#initialize model
def initialize_model():
    chat = ChatGoogleGenerativeAI(temperature=0.0, model="gemini-2.5-flash-lite", max_tokens=None,request_timeout=180,max_retries=3)
    return chat

if __name__ == '__main__':
    print("--- Running Setup ---")
    setup_api_key()
    df_sampled = data_sample()
    chat_model = initialize_model()
    print("\n--- Setup Complete ---")
    print(f"Data loaded with shape: {df_sampled.shape}")
    print("Chat model is ready.")
    print("\n--- Testing Parser ---")
    test_string = "The final answer is #### $1,234.50"
    parsed_value = parser(test_string)
    print(f"Parsing '{test_string}' -> {parsed_value}")
