import pandas as pd
from email.header import decode_header, make_header
from langdetect import detect
import geoip2.database
from email import policy
from email.parser import BytesParser
import io
import time
import multiprocessing as mp
import logging
from bs4 import XMLParsedAsHTMLWarning
import warnings
import awswrangler as wr
import codecs

wr.engine.set("python")
logging.getLogger('nltk').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def get_country(ip):
    if not ip:
        return 'N/A'
    try:
        with geoip2.database.Reader('./GeoIP2-Country.mmdb') as reader:
            return reader.country(ip).country.name
    except Exception:
        return 'N/A'

def detect_lang(text):
    try:
        return detect(text) if pd.notnull(text) and text else 'N/A'
    except Exception:
        return 'N/A'

def extract_data_from_raw_mime(raw_mime, msg_id, s3_uri):
    raw_mime = raw_mime.encode('utf-8')
    msg = BytesParser(policy=policy.default).parsebytes(raw_mime)
    
    content_type = msg.get_content_type()
    text_body = ""
    html_body = ""

    
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            charset = part.get_content_charset('utf-8') or 'utf-8'  # Fallback to UTF-8
            charset = charset.lower().strip().replace('3d', '').replace('=', '')
            # Fallback if still invalid
            try:
                codecs.lookup(charset)
            except LookupError:
                charset = 'utf-8'
            content_disposition = part.get("Content-Disposition", "")
            if content_type == "text/plain" and "attachment" not in content_disposition:
                text_body = part.get_payload(decode=True).decode(charset, errors='replace')
            elif content_type == "text/html":
                html_body = part.get_payload(decode=True).decode(charset, errors='replace')
    else:
        content_type = msg.get_content_type()
        charset = msg.get_content_charset('utf-8') or 'utf-8'  # Fallback to UTF-8
        charset = charset.lower().strip().replace('3d', '').replace('=', '')
        # Fallback if still invalid
        try:
            codecs.lookup(charset)
        except LookupError:
            charset = 'utf-8'
        if content_type == "text/plain":
            text_body = msg.get_payload(decode=True).decode(charset, errors='replace')
        elif content_type == "text/html":
            html_body = msg.get_payload(decode=True).decode(charset, errors='replace')

    # Optional: Save HTML to a file
    if html_body:
        # os.makedirs("html_files", exist_ok=True)
        # with open(f"html_output/{filename}.html", "w", encoding='utf-8') as f:
        #     f.write(html_body)
        # s3_uri = "s3://accsec-ai-prod-snitchmail/processed/phish/html_files/"  # ensure trailing slash
        # s3_uri = "s3://accsec-ai-prod-snitchmail/processed/not_phish/html_files/"
        
        # Upload the file
        dest_path = s3_uri + f"{msg_id}.html"
        buffer = io.BytesIO(html_body.encode('utf-8'))
        wr.s3.upload(
            local_file=buffer,
            path=dest_path
        )
        # print(f"Uploaded to {dest_path}")

    return {
        # "from_address": from_,
        # "to_address": to,
        # "subject": subject,
        # "Date": date,
        "Content-Type": content_type,
        "email_text": text_body
    }

def process_5k_users(args):
    start = time.time()
    file_group, batch_start, batch_end, dest_s3_path, html_s3_path = args
    batch_id = f"{batch_start}-{batch_end}"
    parquet_path = f"{dest_s3_path}users_sample_{batch_id}.parquet"

    dest_files =  wr.s3.list_objects(dest_s3_path)
    if parquet_path in dest_files:
        return
    
    # Read only the required files for this batch
    batch_df = wr.s3.read_parquet(path=file_group)
    
    good_users_5k = pd.DataFrame()
    good_users_5k['user_id'] = batch_df['event.payload.userid']
    good_users_5k['msg_id'] = batch_df['event.payload.msgid']
    good_users_5k['sg_event_id'] = batch_df['event.payload.sg_event_id']
    good_users_5k['subject'] = batch_df['event.payload.subject'].apply(
        lambda x: str(make_header(decode_header(x))) if pd.notnull(x) else None)
    good_users_5k['email_from'] = batch_df['event.payload.from']
    good_users_5k['email_to'] = batch_df['event.payload.email']
    good_users_5k['email_date'] = batch_df['event.payload.date']
    good_users_5k['originating_ip'] = batch_df['event.payload.originating_ip']
    good_users_5k['originating_ip_country'] = batch_df['event.payload.originating_ip'].apply(get_country)
    good_users_5k['lang'] = good_users_5k['subject'].apply(detect_lang)
    good_users_5k_processed_cols = batch_df.apply(
        lambda row:
            extract_data_from_raw_mime(row['raw_mime'],
                                       row['event.payload.msgid'],
                                       html_s3_path), axis=1
    )
    good_users_5k_processed_cols = good_users_5k_processed_cols.apply(pd.Series)
    # Concatenate the extracted columns with original DataFrame
    good_users_5k = pd.concat([good_users_5k, good_users_5k_processed_cols], axis=1)

    wr.s3.to_parquet(good_users_5k, parquet_path)
    print("============================================================")
    print(f"\nParquet file stored for batch: {batch_id} at {parquet_path}\n")
    print(f"Time taken: {str(round((time.time() - start) / 60, 3))} minutes.")
    print("============================================================")


if __name__ == "__main__":
    # s3_dir = "s3://accsec-ai-prod-snitchmail/processed/not_phish/unique_user_email_data/"
    s3_dir = "s3://accsec-ai-prod-snitchmail/processed/not_phish/unique_user_email_part2/"
    # dest_s3_dir = "s3://accsec-ai-prod-snitchmail/processed/not_phish/processed_unique_user_email_data/"
    dest_s3_dir = "s3://accsec-ai-prod-snitchmail/processed/not_phish/processed_unique_user_email_data_part2/"
    html_s3_dir = "s3://accsec-ai-prod-snitchmail/processed/not_phish/html_files/"

    # 1. List and group files (2 files = 1000 rows)
    good_user_files = sorted([f for f in wr.s3.list_objects(s3_dir) if f.endswith('.parquet')])
    existing_files = wr.s3.list_objects(dest_s3_dir)
    
    print(f"Total files to process: {len(good_user_files) - 10*len(existing_files)}.")
    file_groups = [good_user_files[i:i+10] for i in range(0, len(good_user_files), 10)]

    # 2. Create batches with metadata
    batches = [
        (group, idx * 5000, idx * 5000 + 4999, dest_s3_dir, html_s3_dir)
        for idx, group in enumerate(file_groups)
    ]
    
    # 3. Run in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(process_5k_users, batches)