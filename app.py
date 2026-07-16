import os
import time
import json  # JSON obyektlərini oxumaq (parse etmək) üçün əlavə etdik
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

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


def call_llm_with_retry(system_prompt, user_prompt, max_retries=3, delay=2):
    for attempt in range(1, max_retries + 1):
        try:
            # response_format vasitəsilə OpenAI-ın mütləq JSON qaytarmasını məcbur edirik
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},  # JSON rejimini aktivləşdiririk
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                stream=False  # Parsing və struktur yoxlaması üçün bu addımda axını müvəqqəti bağlayırıq
            )
            return response
            
        except (RateLimitError, APITimeoutError, APIError) as e:
            print(f"\n[Retry {attempt}/{max_retries}] API Error: {e}. Gözlənilir: {delay}s...")
            time.sleep(delay)
        except Exception as e:
            raise e
            
    raise Exception("Yenidən cəhd limiti aşdı.")


print("Modeldən gələn cavab yoxlanılır...\n")

try:
    response = call_llm_with_retry(system_prompt, f"User: \"{user_input}\"\nAssistant:")
    raw_content = response.choices[0].message.content
    
    print("--- Modeldən Gələn Xam Mətn ---")
    print(raw_content)
    print("-------------------------------\n")
    
    # Checkpoint 5: JSON Parsing və Validasiya
    try:
        # Gələn mətni Python lüğətinə (dict) çeviririk
        parsed_json = json.loads(raw_content)
        
        # Tələb olunan açarların mövcudluğunu yoxlayırıq (Schema Validation)
        required_keys = ["category", "sentiment", "suggested_reply"]
        is_valid = all(key in parsed_json for key in required_keys)
        
        if is_valid:
            print("✅ Validasiya Uğurlu: Düzgün strukturda JSON aşkar edildi!")
            print(f"Kategoriya: {parsed_json['category'].upper()}")
            print(f"Emosional Vəziyyət (Sentiment): {parsed_json['sentiment'].upper()}")
            print(f"Təklif Olunan Cavab: {parsed_json['suggested_reply']}")
        else:
            print("❌ Validasiya Uğursuz: Bəzi tələb olunan JSON sahələri tapılmadı.")
            
    except json.JSONDecodeError:
        print("❌ Validasiya Uğursuz: Gələn cavab düzgün JSON formatında deyil.")

except Exception as e:
    print(f"\n❌ Tətbiq xətası: {e}")