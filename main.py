import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_FILE = "input.txt"  # Your text file

def summarize_text(text, max_len=150, min_len=30):
    """Generates a summary of the input text using BART."""
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(DEVICE)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=min_len,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def extract_keywords(text, num_keywords=5):
    """Extracts keywords from text using simple frequency heuristic."""
    words = [word.lower() for word in text.replace(".", "").replace(",", "").split()]
    freq = {}
    for w in words:
        if len(w) > 3:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq, key=freq.get, reverse=True)
    return sorted_words[:num_keywords]

def generate_questions(text, max_questions=5):
    """Generates base questions using valhalla/t5-small-qg-hl."""
    tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl")
    model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-hl").to(DEVICE)

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
        if len(questions) >= max_questions:
            break
    return questions

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

def main():
    try:
        with open(TEXT_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: {TEXT_FILE} not found!")
        return

    # --- User inputs ---
    summary_choice = input("Do you want a longer or shorter summary? (long/short): ").strip().lower()
    if summary_choice == "long":
        max_len, min_len = 250, 50
    else:
        max_len, min_len = 100, 20

    num_keywords = int(input("How many keywords do you want to extract? "))
    num_questions = int(input("How many questions do you want to generate? "))
    num_options = int(input("How many multiple-choice options per question? "))

    # --- Generate content ---
    print("\n--- Summary ---")
    summary = summarize_text(text, max_len=max_len, min_len=min_len)
    print(summary)

    print("\n--- Keywords ---")
    keywords = extract_keywords(text, num_keywords=num_keywords)
    print(keywords)

    print("\n--- Questions ---")
    questions = generate_questions(text, max_questions=num_questions)
    for idx, q in enumerate(questions, 1):
        options = generate_mcq_options(q["answer"], keywords, num_options=num_options)
        print(f"\n{idx}. {q['question']}")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")

if __name__ == "__main__":
    main()
