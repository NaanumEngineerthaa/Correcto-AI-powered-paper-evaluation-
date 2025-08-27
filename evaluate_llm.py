from llama_cpp import Llama

# Load the model
llm = Llama(model_path=r"models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=2048)

def evaluate_answer(llm, key_answer, student_answer):
    prompt = f"""You are an AI teacher. Compare the student's answer with the correct answer and give a mark from 0 to 100 (only the number, no explanation).

Correct Answer: {key_answer}
Student Answer: {student_answer}

Score:"""

    response = llm(prompt, max_tokens=5, temperature=0.0, stop=["\n"])
    score = response["choices"][0]["text"].strip()
    return score




key = "The heart pumps blood throughout the body using blood vessels."
student = "Heart helps in circulating blood in body."

score = evaluate_answer(llm, key, student)
print(f"Mark: {score}/100")