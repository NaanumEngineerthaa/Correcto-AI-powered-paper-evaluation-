# test_llm.py

from llama_cpp import Llama

# Path to your downloaded GGUF model
model_path = "models/llama.gguf/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Load the model (this may take 20-60 seconds)
llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4)

# Sample test prompt
prompt = "You are a teacher. Evaluate this answer:\nKey: Water boils at 100 degrees Celsius.\nStudent: Water gets hot and steamy when heated."

# Get response
response = llm(prompt, max_tokens=128)
print("=== LLM Response ===")
print(response["choices"][0]["text"].strip())