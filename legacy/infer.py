import os
import re
import json
import csv
import torch
from tqdm import tqdm

from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor
from qwen_omni_utils import process_mm_info


MODEL_PATH = "/data-nfs/gpu2/um202574318/models/Qwen2.5-Omni-7B"

DEV_JSONL = "/data-nfs/gpu2/um202574318/data/DCASE2026-Task5-DevSet/dev.jsonl"
AUDIO_ROOT = "/data-nfs/gpu2/um202574318/data/DCASE2026-Task5-DevSet"

OUTPUT_DIR = "/data-nfs/gpu2/um202574318/outputs"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "qwen_omni_7b_base_dev.csv")


SYSTEM_PROMPT = (
    "You are an audio understanding model that answers multiple choice questions "
    "based on audio content."
)


def normalize_text(x):
    return re.sub(r"\s+", " ", str(x).strip().lower()) if x is not None else ""


def build_question_prompt(question, choices):
    options = [str(c) for c in choices]
    return (
        f"{question} "
        f"Please choose the answer from the following options: {options}. "
        f"Output the final answer in <answer> </answer>."
    )


def clean_answer(pred, choices):
    if pred is None or len(choices) == 0:
        return choices[0] if choices else ""

    raw = str(pred).strip()

    # 优先解析 <answer>...</answer>
    m = re.search(r"<answer>(.*?)</answer>", raw, flags=re.I | re.S)
    if m:
        raw = m.group(1).strip()

    raw = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    raw = raw.strip("\"'“”‘’").strip()

    raw_norm = normalize_text(raw)

    # exact option text
    for c in choices:
        if raw_norm == normalize_text(c):
            return c

    # letter fallback
    upper = raw.upper().strip()
    letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    if upper in letter_map and letter_map[upper] < len(choices):
        return choices[letter_map[upper]]

    m = re.search(r"\b([ABCD])\b", upper)
    if m:
        idx = letter_map[m.group(1)]
        if idx < len(choices):
            return choices[idx]

    # contains option text
    sorted_choices = sorted(choices, key=lambda x: len(str(x)), reverse=True)

    for c in sorted_choices:
        c_norm = normalize_text(c)
        if c_norm and c_norm in raw_norm:
            return c

    for c in sorted_choices:
        c_norm = normalize_text(c)
        if raw_norm and raw_norm in c_norm:
            return c

    return choices[0]


def infer_one(model, processor, audio_path, question, choices):
    prompt = build_question_prompt(question, choices)

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    audios, images, videos = process_mm_info(
        messages,
        use_audio_in_video=False,
    )

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            num_beams=1,
            use_audio_in_video=False,
            return_audio=False,
        )

    input_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[:, input_len:]

    pred_raw = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    pred = clean_answer(pred_raw, choices)

    return pred, pred_raw


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    print("Loading model...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print("Loading dev set...")
    data = []
    with open(DEV_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Total samples: {len(data)}")

    rows = []

    for item in tqdm(data):
        qid = str(item["id"])
        question = item["question_text"]
        choices = item["multi_choice"]

        audio_path = item["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(AUDIO_ROOT, audio_path)

        if not os.path.exists(audio_path):
            print(f"[MISSING AUDIO] {qid}: {audio_path}")
            pred = choices[0]
            pred_raw = "[MISSING_AUDIO]"
        else:
            try:
                pred, pred_raw = infer_one(
                    model=model,
                    processor=processor,
                    audio_path=audio_path,
                    question=question,
                    choices=choices,
                )
            except Exception as e:
                print(f"[ERROR] {qid}: {repr(e)}")
                pred = choices[0]
                pred_raw = f"[ERROR] {repr(e)}"
                torch.cuda.empty_cache()

        rows.append({
            "question": qid,
            "answer": pred,
            "pred_raw": pred_raw,
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "answer", "pred_raw"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()