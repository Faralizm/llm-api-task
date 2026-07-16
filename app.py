import os
import time
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

# İnstall API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("ERROR: 'OPENAI_API_KEY' not found!")

client = OpenAI(api_key=api_key)

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


# Checkpoint 4: retry logic
def call_llm_with_retry(system_prompt, user_prompt, max_retries=3, delay=2):
    """
    API xətaları, Rate limit və ya Timeout zamanı 3 cəhd edən funksiya.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                stream=True
            )
            return response
            
        except RateLimitError as e:
            print(f"\n[Retry {attempt}/{max_retries}] Rate limit! Gözlənilir: {delay}s...")
            time.sleep(delay)
        except APITimeoutError as e:
            print(f"\n[Retry {attempt}/{max_retries}] Timeout! Gözlənilir: {delay}s...")
            time.sleep(delay)
        except APIError as e:
            print(f"\n[Retry {attempt}/{max_retries}] API Error: {e}. Gözlənilir: {delay}s...")
            time.sleep(delay)
        except Exception as e:
            print(f"\n[Gözlənilməyən Xəta]: {e}")
            raise e
            
    raise Exception("Yenidən cəhd limitini aşdı.")


print("Sorğu göndərilir...\n")

try:
    response_stream = call_llm_with_retry(system_prompt, f"User: \"{user_input}\"\nAssistant:")
    
    full_response_text = ""
    print("--- Live Stream Response ---")
    for chunk in response_stream:
        chunk_text = chunk.choices[0].delta.content
        if chunk_text is not None:
            print(chunk_text, end="", flush=True)
            full_response_text += chunk_text
    print("\n---------------------------")
    
except Exception as e:
    print(f"\n❌ Proqram işləmədi: {e}")