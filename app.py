import os
import time
import json
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
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                stream=False
            )
            return response
            
        except (RateLimitError, APITimeoutError, APIError) as e:
            print(f"\n[Retry {attempt}/{max_retries}] API Error: {e}. Waiting {delay}s...")
            time.sleep(delay)
        except Exception as e:
            raise e
            
    raise Exception("Failed after maximum retries.")


print("Sorğu göndərilir və xərclər hesablanır...\n")

try:
    response = call_llm_with_retry(system_prompt, f"User: \"{user_input}\"\nAssistant:")
    raw_content = response.choices[0].message.content
    
    print("--- Modeldən Gələn Cavab ---")
    print(raw_content)
    print("----------------------------\n")
    
    # 5. Checkpoint: JSON Parsing və Validasiya
    try:
        parsed_json = json.loads(raw_content)
        required_keys = ["category", "sentiment", "suggested_reply"]
        is_valid = all(key in parsed_json for key in required_keys)
        
        if is_valid:
            print("✅ Validasiya Uğurlu!")
        else:
            print("❌ Validasiya Uğursuz!")
    except json.JSONDecodeError:
        print("❌ JSON Oxunarkən Xəta Baş Verdi!")

    # 6. Checkpoint: Token və Xərc Hesabatı (Cost/Token Logging)
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    # gpt-4o-mini rəsmi qiymətləri (1 milyon token üçün):
    # Giriş (Input): $0.150 / M
    # Çıxış (Output): $0.600 / M
    input_cost = (prompt_tokens / 1_000_000) * 0.150
    output_cost = (completion_tokens / 1_000_000) * 0.600
    total_cost = input_cost + output_cost

    print("\n================ TOKEN & COST REPORT ================")
    print(f"📥 Giriş Tokenləri (Prompt Tokens): {prompt_tokens}")
    print(f"📤 Çıxış Tokenləri (Completion Tokens): {completion_tokens}")
    print(f"📊 Ümumi Token Sayı (Total Tokens): {total_tokens}")
    print(f"💵 Təxmini Sorğu Qiyməti (Total Cost): ${total_cost:.8f}")
    print("=====================================================")

except Exception as e:
    print(f"\n❌ Xəta baş verdi: {e}")