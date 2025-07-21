# PACKAGE IMPORTS
from openai import OpenAI
import torch
from IPython.display import display, HTML
import os
import pandas as pd
import json
import sys
from dotenv import load_dotenv
from together import Together

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
together_key = os.getenv("TOGETHERAI_API_KEY")


# Function to use GPT‑4
client = OpenAI(api_key=openai_key)

def query_gpt4(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=300
    )
    return response.choices[0].message.content


# Funtction to use LLaMA but issues related to device capacity.
def query_llama(prompt, max_tokens=64, temperature=0.8, top_k=50, top_p=0.95):
    """Given parameters and prompt:   """
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        echo=False,
        top_k=top_k,
        top_p=top_p
    )
    return output["choices"][0]["text"]