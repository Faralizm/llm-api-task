import os
from dotenv import load_dotenv
from openai import OpenAI

# .env faylından API açarını yükləyirik
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("ERROR: 'OPENAI_API_KEY' not found!")

client = OpenAI(api_key=api_key)

# 2. Checkpoint-dən olan Prompt-larimiz
system_prompt = """
You are a professional customer support assistant. Analyze the incoming user inquiry and respond strictly in the following JSON format:
{
  "category": "the category of the inquiry (e.g., shipping, billing, technical, refund)",
  "sentiment": "the emotional state of the user (e.g., positive, neutral, angry)",
  "suggested_reply": "a polite, professional, and helpful response text in the customer's language"
}
RULE: Do not return any introduction, explanation, markdown blocks, or extra text. Return ONLY the raw JSON object.
"""

user_input = "I tried to log in but it keeps giving me a 500 internal server error."

print("Sending request to OpenAI with streaming enabled...\n")

# 3. Checkpoint: Streaming-in reallaşdırılması (stream=True)
response_stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User: \"{user_input}\"\nAssistant:"}
    ],
    temperature=0.2,
    stream=True # Cavabın axınla gəlməsini aktiv edirik
)

full_response_text = ""

print("--- Live Stream Response ---")
# Hər gələn simvol hissəsini (chunk) canlı olaraq ekrana yazdırırıq
for chunk in response_stream:
    chunk_text = chunk.choices[0].delta.content
    if chunk_text is not None:
        print(chunk_text, end="", flush=True)
        full_response_text += chunk_text
print("\n---------------------------")