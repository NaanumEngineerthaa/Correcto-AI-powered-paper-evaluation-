from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import easyocr
import fitz
from evaluate_llm import llm, evaluate_answer  # llama-related functions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

reader = easyocr.Reader(['en'])  # initialize EasyOCR

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def extract_text_from_image(image_path):
    result = reader.readtext(image_path, detail=0)
    return " ".join(result).strip()

# Use LLM to find the best matching QA pair from key answer set
def find_relevant_key_answer(llm, full_key_text, student_answer):
    prompt = f"""
You are a helpful assistant. The following is a list of multiple question-and-answer pairs from a textbook or key file. Based on the student's answer, pick the most relevant question-answer pair that matches what the student has attempted to answer.

Key Answers:
{full_key_text}

Student Answer:
{student_answer}

Return only the most relevant question and its answer.
"""
    response = llm(prompt)
    if isinstance(response, dict) and 'text' in response:
        return response['text'].strip()
    else:
        return str(response).strip()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        key_pdf = request.files.get('key_pdf')
        student_img = request.files.get('student_image')

        key_text = request.form.get('key_text', '').strip()
        student_text = request.form.get('student_text', '').strip()

        key_answer = ""
        student_answer = ""

        if key_pdf:
            key_pdf_filename = secure_filename(key_pdf.filename)
            key_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], key_pdf_filename)
            key_pdf.save(key_pdf_path)
            key_answer = extract_text_from_pdf(key_pdf_path)
        elif key_text:
            key_answer = key_text

        if student_img:
            student_img_filename = secure_filename(student_img.filename)
            student_img_path = os.path.join(app.config['UPLOAD_FOLDER'], student_img_filename)
            student_img.save(student_img_path)
            student_answer = extract_text_from_image(student_img_path)
        elif student_text:
            student_answer = student_text

        if not key_answer or not student_answer:
            return jsonify({'error': 'Missing key or student answer input.'}), 400

        # Get the most relevant key QA pair using LLM
        matched_key_answer = find_relevant_key_answer(llm, key_answer, student_answer)

        # Evaluate score using matched QA
        score = evaluate_answer(llm, matched_key_answer, student_answer)

        return jsonify({
            'key_answer': matched_key_answer,
            'student_answer': student_answer,
            'score': score
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
