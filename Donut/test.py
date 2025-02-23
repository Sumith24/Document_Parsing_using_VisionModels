import os
import time
import json
import torch
from PIL import Image
from difflib import SequenceMatcher
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Load pre-trained model
model_name = "naver-clova-ix/donut-base-finetuned-cord-v2"
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_evaluations(extracted, ground_truth):
    """
    Compute accuracy using similarity.
    """
    return SequenceMatcher(None, extracted.lower(), ground_truth.lower()).ratio()

def parse_receipt_data(image_path):
    """
    Extract structured data from a receipt image using Donut model.
    """
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # Generate tokenized output
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    start_time = time.time()
    #generate output
    outputs = model.generate(pixel_values.to(device),
                            decoder_input_ids=decoder_input_ids.to(device), 
                            max_length=model.decoder.config.max_position_embeddings,
                            early_stopping=True, 
                            pad_token_id=processor.tokenizer.pad_token_id,
                            eos_token_id=processor.tokenizer.eos_token_id,
                            use_cache=True,
                            num_beams=1,
                            bad_words_ids=[[processor.tokenizer.unk_token_id]],
                            return_dict_in_generate=True,
                            output_scores=True
                            )
    end_time = time.time()
    
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.sep_token, "")
    result= processor.token2json(sequence)
    processing_time = round(end_time - start_time, 2)
    
    result["filename"] = os.path.basename(image_path)
    result["processing_time"] = processing_time
    
    return result

def process_receipts(input_folder, output_file):
    """
    Process all receipt images in a folder and save structured output.
    """
    output_jsons = []
    
    for file in os.listdir(input_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            receipt_data = parse_receipt_data(os.path.join(input_folder, file))
            output_jsons.append(receipt_data)
    
    with open(output_file, "w") as f:
        json.dump(output_jsons, f, indent=4)
    
    print(f"Extraction completed. Results saved in {output_file}")

if __name__ == "__main__":
    input_folder = r"/content/"
    output_file = "extracted_receipts.json"
    process_receipts(input_folder, output_file)
