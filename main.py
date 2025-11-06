import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import random

# ------------------------- CONFIG -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_FILE = "input.txt"  # Your text file
NUM_KEYWORDS = 5
NUM_OPTIONS = 4
# ----------------------------------------------------------

# ------------------------- SUMMARY -------------------------
def summarize_text(text):
    """Generates a summary of the input text using BART."""
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(DEVICE)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# ------------------------- KEYWORDS -------------------------
def extract_keywords(text, num_keywords=5):
    """Extracts keywords from text using sentence-transformers embeddings."""
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=DEVICE)
    sentences = text.split(".")
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Simple heuristic: pick top frequent words
    words = [word.lower() for word in text.replace(".", "").replace(",", "").split()]
    freq = {}
    for w in words:
        if len(w) > 3:
            freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq, key=freq.get, reverse=True)
    return sorted_words[:num_keywords]

# ------------------------- QUESTIONS -------------------------
def generate_questions(text):
    """Generates base questions using valhalla/t5-small-qg-hl."""
    tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl")
    model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-hl").to(DEVICE)

    # Format input for QG: highlight sentences
    sentences = text.split(".")
    questions = []
    for sent in sentences:
        if len(sent.strip()) < 10:
            continue
        input_text = f"generate question: {sent.strip()} </s>"
        inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_length=64)
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if question:
            questions.append({"question": question, "answer": sent.strip()})
    return questions

# ------------------------- MULTIPLE CHOICE -------------------------
def generate_mcq_options(correct_answer, keywords, num_options=4):
    """Creates multiple-choice options including correct answer and distractors."""
    options = [correct_answer]
    distractors = [kw for kw in keywords if kw.lower() != correct_answer.lower()]
    
    while len(options) < num_options:
        if distractors:
            option = random.choice(distractors)
            options.append(option)
            distractors.remove(option)
        else:
            options.append("Option" + str(len(options)+1))
    random.shuffle(options)
    return options

# ------------------------- MAIN -------------------------
def main():
    try:
        with open(TEXT_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: {TEXT_FILE} not found!")
        return

    print("\n--- Summary ---")
    summary = summarize_text(text)
    print(summary)

    print("\n--- Keywords ---")
    keywords = extract_keywords(text, NUM_KEYWORDS)
    print(keywords)

    print("\n--- Questions ---")
    questions = generate_questions(text)
    for idx, q in enumerate(questions, 1):
        options = generate_mcq_options(q["answer"], keywords, NUM_OPTIONS)
        print(f"\n{idx}. {q['question']}")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")

if __name__ == "__main__":
    main()
