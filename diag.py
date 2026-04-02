"""Quick diagnostic for GPT-OSS and Gemini response structures."""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")

from groq import Groq
c = Groq(api_key=os.environ["GROQ_API_KEY"])

r = c.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=30, temperature=0.0,
    reasoning_effort="low", reasoning_format="hidden"
)
txt = r.choices[0].message.content or ""
print(f"GPT-OSS content repr: {repr(txt)}")

from google import genai as gg
client = gg.Client(api_key=os.environ["GEMINI_API_KEY"])
resp = client.models.generate_content(
    model="gemini-2.5-flash", contents="What is 2+2?",
    config=gg.types.GenerateContentConfig(max_output_tokens=50, temperature=0.0)
)
print(f"Gemini resp type: {type(resp)}")
print(f"Gemini text attr: {repr(getattr(resp, 'text', 'NO_TEXT_ATTR'))}")
if hasattr(resp, "candidates") and resp.candidates:
    part = resp.candidates[0].content.parts[0]
    print(f"Gemini via candidates[0].content.parts[0].text: {repr(part.text)}")
