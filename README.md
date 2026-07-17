# LLM API Integration Task

This repository contains a production-ready implementation of an LLM-powered customer support analysis backend using the OpenAI SDK. The system orchestrates prompt engineering, response streaming, resilient error handling, strict output schema validation, and real-time transaction cost logging.

## 🚀 Features Implemented
- **Secure Configuration:** Environment variables managed carefully with `python-dotenv` and protected via `.gitignore`.
- **Structured Prompts:** Built-in persona framing combined with Few-Shot examples forcing the model to classify inquiries into JSON objects.
- **Streaming UI Optimization:** Chunk-by-chunk real-time tokens rendering for optimized user-perceived speed.
- **Fail-safe Logic:** Automatic retry capabilities handling `RateLimitError` and `APITimeoutError` gracefully up to 3 attempts.
- **Output Validation:** Post-execution parser that decodes JSON strings safely and guarantees schema requirements are strictly met before output.
- **Cost Monitoring:** Programmatic cost estimation based on input/output token counters relative to standard `gpt-4o-mini` pricing tiers.

---

## 🛠️ Setup & Installation Instructions

Follow these quick steps to set up and run the script locally:

### 1. Clone the Repository
```bash
git clone [https://github.com/Faralizm/llm-api-task.git](https://github.com/Faralizm/llm-api-task.git)
cd llm-api-task
