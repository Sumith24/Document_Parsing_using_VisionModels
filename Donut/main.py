import os
import time
import json
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Load pre-trained model
model_name = "naver-clova-ix/donut-base-finetuned-cord-v2"
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

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

def get_evaluations(ground_truth, extracted_path):
    """
    Compute accuracy using similarity.
    """
    c_name_count = 0
    t_price_count = 0
    total = 0
    time_tkn = list()
    for parsed_file in os.listdir(extracted_path):
        for grwdth_file in os.listdir(ground_truth):
            if grwdth_file.endswith(".json"):
                if os.path.splitext(parsed_file)[0].split("_")[-1]==grwdth_file.split("_")[0]:
                    total+=1
                    with open(os.path.join(extracted_path, parsed_file), "r") as pf:
                        p_data = json.load(pf)
                        if isinstance(p_data["menu"], list):
                            parsed_company_name = p_data["menu"][0]["nm"]
                        else:
                            parsed_company_name = p_data["menu"]["nm"]
                        parsed_total_price = p_data["total"]["total_price"]
                    with open(os.path.join(ground_truth, grwdth_file), "r") as gf:
                        g_data = json.load(gf)
                        ground_company_name = g_data["Company_Name"]
                        ground_total_price = g_data["total_price"]
                    if parsed_company_name==ground_company_name:
                        c_name_count += 1
                    if parsed_total_price==ground_total_price:
                        t_price_count += 1
                    time_tkn.append(p_data["processing_time"])

    avg_processing_time = sum(time_tkn)/len(time_tkn)
    print(f"\nOut of {total} images, Model extracted {c_name_count}-Company_Name correctly")
    print(f"Out of {total} images, Model extracted {t_price_count}-total_price correctly")
    print(f"Average time taken to process a single image is {round(avg_processing_time, 2)} sec")
    accuarcy = (c_name_count+t_price_count)/(2*total)
    print(f"\nModel accuracy is {round(accuarcy, 2)}")

def process_receipts(input_folder, output_path):
    """
    Process all receipt images in a folder and save structured output.
    """
    for file in os.listdir(input_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            receipt_data = parse_receipt_data(os.path.join(input_folder, file))
            output_file = os.path.join(output_path,"parsed_json_"+os.path.basename(file).split(".")[0]+".json")
            with open(output_file, "w") as f:
                json.dump(receipt_data, f, indent=4)
            print(f"Extraction completed. Results saved in {output_file}")

if __name__ == "__main__":
    input_folder = "../Image_data/"
    output_path = "output/"
    # process_receipts(input_folder, output_path)
    get_evaluations(input_folder, output_path)

