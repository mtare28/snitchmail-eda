import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re
from urllib.parse import urlparse
import math
import time
import pandas as pd
from langdetect import detect
import torch
import awswrangler as wr

from deep_translator import GoogleTranslator
from transformers import (pipeline, AutoTokenizer 
    , BertForSequenceClassification, BertTokenizer
    , AutoTokenizer, AutoModelForSequenceClassification
    , AutoModelForCausalLM)
from peft import PeftModel, PeftConfig
from pydantic import BaseModel, field_validator

class PhishingResponse(BaseModel):
    is_phishing: int

    @field_validator('is_phishing', mode='before')
    @classmethod
    def extract_boolean(cls, v):
        # Find first TRUE/FALSE match (case-insensitive)
        if isinstance(v, str):
            match = re.search(r'\b(true|false)', v, re.IGNORECASE)
            if match:
                return 1 if match.group(1).lower() == 'true' else 0
        return 0  # Default if no valid match

def load_dataframe(src):
    df = wr.s3.read_parquet(path=src)
    print(f"Dataframe loaded of size: {len(df)} from {src}!")
    return df

def clean_email_text(text):
    if not isinstance(text, str):
        return ""
    
    def extract_domain(match):
        url = match.group(0)
        try:
            # Only process if it looks like a URL
            if url.startswith(('http://', 'https://')):
                parsed = urlparse(url)
                if parsed.netloc:
                    return f"{parsed.scheme}://{parsed.netloc}"
                else:
                    return url  # Return as is if netloc is empty
            elif url.startswith('www.'):
                parsed = urlparse('http://' + url)
                if parsed.netloc:
                    return parsed.netloc
                else:
                    return url  # Return as is if netloc is empty
            else:
                return url  # Not a URL, return as is
        except Exception:
            return url  # On any parsing error, return as is

    # Improved URL pattern: only match URLs, not tokens like LINK:UNSUBSCRIBE
    url_pattern = re.compile(r'(https?://[a-zA-Z0-9./?=_\-#%&]+|www\.[a-zA-Z0-9./?=_\-#%&]+)')
    text = url_pattern.sub(extract_domain, text)
    
    # Remove special Unicode characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F\u200c\xa0]', ' ', text)
    # Collapse multiple whitespaces
    return re.sub(r'\s+', ' ', text).strip()

def translate_text(text):
    try:
        if detect(text) != 'en':
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except Exception:
        return text  # Return original if translation fails

def extract_parent_domains(text):
    """
    Extracts unique parent domains (scheme://netloc or netloc for www.) from URLs in the input text.
    Returns a list of domains.
    """
    if not isinstance(text, str):
        return []

    def get_parent_domain(url):
        try:
            if url.startswith(('http://', 'https://')):
                parsed = urlparse(url)
                if parsed.netloc:
                    return f"{parsed.scheme}://{parsed.netloc}"
            elif url.startswith('www.'):
                parsed = urlparse('http://' + url)
                if parsed.netloc:
                    return parsed.netloc
        except Exception:
            pass
        return None

    # URL regex pattern
    url_pattern = re.compile(r'(https?://[a-zA-Z0-9./?=_\-#%&]+|www\.[a-zA-Z0-9./?=_\-#%&]+)')
    matches = url_pattern.findall(text)
    domains = set()
    for url in matches:
        domain = get_parent_domain(url)
        if domain:
            domains.add(domain)
    return ' '.join(list(domains))


def initialize_model(model_name, tokenizer_name=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = pipeline(
            model=model_name,
            tokenizer=tokenizer,
            truncation=True,
            device=0
        )
    else:
        # 2.
        # model = BertForSequenceClassification.from_pretrained(model_name).to(device)
        # tokenizer = BertTokenizer.from_pretrained(model_name)
        # model.eval()

        # 3.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        model.eval()

        # 4.
        # config = PeftConfig.from_pretrained(model_name)
        # base_model_name = config.base_model_name_or_path
        
        # tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side='left')
        # base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        # model_with_lora = PeftModel.from_pretrained(base_model, model_name)
    
        # model = model_with_lora.to(device)
    print(f"Model initialized: {model_name}!")
    return model, tokenizer


def process_batch(model, df_batch, batch_size, label):
    # Make a deep copy to ensure independence
    batch = df_batch.copy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    score_label = '_'.join(label.split('_')[:-1]) + '_score'

    if label == "ealvaradob_bert_finetuned_label":
        # Clean text
        batch['cleaned_text'] = batch['email_text'].apply(clean_email_text)
        batch['subject_body'] = (
            batch['subject'].fillna('') + ' ' + batch['cleaned_text'].fillna('')
        ).str.strip()
        # Translate non-English text
        batch['translated_text'] = batch['subject_body'].apply(translate_text)
        # batch['url_domains'] = batch['email_text'].apply(extract_parent_domains)

    texts = [str(text) if text is not None else "" for text in batch['subject_body'].tolist()]

    if label == "ealvaradob_bert_finetuned_label":
        # Run phishing detection
        predictions = model(texts, batch_size=batch_size)
        
        # Extract labels
        batch[label] = [1 if pred['label'] == "phishing" else 0 for pred in predictions]
        batch[score_label] = [round(pred['score'], 6) if pred['label'] == 'phishing' else round(1 - pred['score'], 6) for pred in predictions]
        
    elif label == 'elslay_bert_finetuned_label':
        preds = []
        confidences = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                # Tokenize batch
                inputs = tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    truncation=True,
                    padding='max_length',
                    max_length=512
                )
                # Move tensors to GPU
                inputs = {k: v.to(device) for k, v in inputs.items()}
                # Model inference
                outputs = model(**inputs)
                logits = outputs.logits

                # 2
                probs = torch.softmax(logits, dim=-1)
                batch_preds = torch.argmax(probs, dim=-1).cpu().numpy()
                preds.extend(batch_preds)
                confs = [round(prob, 6) if batch_preds[i] == 1 else round(1 - prob, 6) for i, prob in enumerate(probs.max(dim=-1).values.cpu().numpy())]
                confidences.extend(confs)
        # Map predictions to label names
        batch[label] = preds
        batch[score_label] = confidences
    elif label == 'cybersectony_distilbert_finetuned_label':
        preds = []
        confidences = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                # Tokenize batch
                inputs = tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    truncation=True,
                    padding='max_length',
                    max_length=512
                )
                # Move tensors to GPU
                inputs = {k: v.to(device) for k, v in inputs.items()}
                # Model inference
                outputs = model(**inputs)
                logits = outputs.logits
                
                # 3
                # label names
                label_names = ["legitimate_email", "phishing_url", "legitimate_url", "phishing_url_alt"]

                logits = model(**inputs).logits
                predictions = torch.softmax(logits, dim=-1)  # [batch_size, num_labels]
                probs = predictions.cpu().tolist()
                
                for prob_list in probs:
                    # Map probabilities to labels
                    labels = dict(zip(label_names, prob_list))
                    # Determine most likely classification
                    max_label = max(labels.items(), key=lambda x: x[1])
                    pred = 0 if max_label[0] in ('legitimate_email', 'legitimate_url') else 1
                    conf = round(max_label[1], 6) if pred == 1 else round(1 - max_label[1], 6)
                    preds.append(pred)
                    confidences.append(conf)
        # Map predictions to label names
        batch[label] = preds
        batch[score_label] = confidences
    # elif label == "AcuteShrewdSecurity_phishsense_llama_finetuned_label":
    #     responses = []
    #     for i in range(0, len(texts), batch_size):
    #         batch_texts = texts[i:i+batch_size]
    #         prompts = [
    #             f"Classify as phishing (TRUE) or not (FALSE):\n{text}\n<<ANSWER>>:"
    #             for text in batch_texts
    #         ]
    #         inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    #         if torch.cuda.is_available():
    #             inputs = {k: v.to(device) for k, v in inputs.items()}
    #         with torch.no_grad():
    #             outputs = model.generate(
    #                 **inputs,
    #                 max_new_tokens=5,
    #                 temperature=0.01,
    #                 pad_token_id=tokenizer.eos_token_id
    #             )
    #         # Decode and extract only the model's answer
    #         decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #         # Extract everything after the last "Answer:" in the output
    #         for prompt, output in zip(prompts, decoded):
    #             # Remove the prompt part from the output
    #             answer = output.split("<<ANSWER>>:")[-1].strip()
    #             responses.append(PhishingResponse(is_phishing=answer).is_phishing)
    #     # Map predictions to label names
    #     batch[label] = responses
        
    #     # Map predictions to label names
    #     batch[label] = preds
    #     batch[score_label] = confidences
    return batch

def save_to_s3(dst_filepath, df):
    wr.s3.to_parquet(
        df=df,
        path=dst_filepath,
        index=False
    )
    print(f"Saved Final Dataframe of size: {len(df)} to {dst_filepath}!")
    

if __name__ == "__main__":
    # 1. Load data
    data_path = "s3://accsec-ai-prod-snitchmail/processed/phish/user_data.parquet"
    # data_path = "s3://accsec-ai-prod-snitchmail/processed/phish/labeled_user_data.parquet"
    # data_df = load_dataframe(data_path)

    # data_df = pd.read_csv("csv_files/phishnet_flagged_records_05-06May.csv")
    # data_df = pd.read_csv("csv_files/labeled_phishnet_May5-6.csv")

    models = [
            'ealvaradob/bert-finetuned-phishing',
          'ElSlay/BERT-Phishing-Email-Model',
          'cybersectony/phishing-email-detection-distilbert_v2.4.1'
    ]
    labels = [
        'ealvaradob_bert_finetuned_label',
          'elslay_bert_finetuned_label',
          'cybersectony_distilbert_finetuned_label'
    ]
    tokenizers = [
        "bert-large-uncased",
        None, None
    ]
    
    # Model 4
    # model_name = 'AcuteShrewdSecurity/Llama-Phishsense-1B'
    # label = 'AcuteShrewdSecurity_phishsense_llama_finetuned_label'
    # model, tokenizer = initialize_model(model_name)
    
    # src = 'csv_files/w_label/good_user_emails_w_sub.parquet'
    src = 'csv_files/w_label/phish_user_emails_2.parquet'
    data_df = pd.read_parquet(src)
    total_rows = len(data_df)
    print(f"Dataframe loaded of size: {total_rows} from {src}!")

    for model_name, tokenizer_name, label in zip(models, tokenizers, labels):
        
        model, tokenizer = initialize_model(model_name, tokenizer_name)
        
        # 3. Process dataframe in batches
        inference_batch_size = 16  # For model inference
        df_chunk_size = 1000       # Number of rows per DataFrame chunk (tune as needed)
        
        results = []
        num_chunks = math.ceil(total_rows / df_chunk_size)
        
        start_time = time.time()
        for i in range(num_chunks):
            start_idx = i * df_chunk_size
            end_idx = min((i + 1) * df_chunk_size, total_rows)
            chunk = data_df.iloc[start_idx:end_idx].copy()
            processed_chunk = process_batch(model, chunk, inference_batch_size, label)
            results.append(processed_chunk)
            elapsed = time.time() - start_time
            print(f"[{time.strftime('%X')}] Processed batch {i + 1}/{num_chunks} ({end_idx}/{total_rows} rows) in {elapsed:.1f}s")
    
        # 4. Combine results and save
        data_df = pd.concat(results)
    
        # 5. Save dataframe to S3 location as parquet file
        # output_file = "s3://accsec-ai-prod-snitchmail/processed/phish/labeled_user_data.parquet"
        output_file = 'csv_files/w_label/phish_user_emails_urls.parquet'
        # output_file = 'csv_files/w_label/good_user_emails_urls.parquet'
        # save_to_s3(output_file, final_df)
        data_df.to_parquet(output_file, index=False)
        print(f'{model_name} finished inferencing and saved data to {output_file}')

    