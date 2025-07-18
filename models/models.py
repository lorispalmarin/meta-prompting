from openai import OpenAI
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# GPT‑4
client = OpenAI(api_key="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")

def query_gpt4(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=300
    )
    return response.choices[0].message.content

# LLama
def query_llama(prompt, max_tokens=64, temperature=0.8, top_k=50, top_p=0.95):
    """
    Interroga LLaMA e restituisce solo il testo generato.
    """

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        echo=False,
        top_k=top_k,
        top_p=top_p
    )
    return output["choices"][0]["text"]