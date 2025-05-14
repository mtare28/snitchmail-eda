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
import argostranslate.translate


def extract_data_from_raw_mime(raw_mime):
    if isinstance(raw_mime, str):
        raw_mime = raw_mime.encode('utf-8')
    msg = BytesParser(policy=policy.default).parsebytes(raw_mime)
    
    content_type = msg.get_content_type()
    text_body = ""
    html_body = ""

    def decode_part(part):
        charset = part.get_content_charset('utf-8') or 'utf-8'
        charset = charset.lower().strip().replace('3d', '').replace('=', '')
        try:
            codecs.lookup(charset)
        except LookupError:
            charset = 'utf-8'
        return part.get_payload(decode=True).decode(charset, errors='replace')

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = part.get("Content-Disposition", "")
            if content_type == "text/plain" and "attachment" not in content_disposition:
                text_body = decode_part(part)
            elif content_type == "text/html":
                html_body = decode_part(part)
    else:
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            text_body = decode_part(msg)
        elif content_type == "text/html":
            html_body = decode_part(msg)

    # If only HTML is present, convert to plain text
    if not text_body and html_body:
        soup = BeautifulSoup(html_body, 'html.parser')
        text_body = soup.get_text(separator=' ', strip=True)

    return text_body

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

def translate_text(text, lang):
    lang = lang.split('-')[0]
    translator = translators.get(lang)
    if translator is not None:
        try:
            return translator.translate(text)
        except Exception:
            return text
    return text


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
    return ' '.join(list(domains))s


if __name__ == "__main__":
    # read data
    df_path = ''
    df = pd.read_csv(df_path)

    # Build a dictionary of translators for each language in your DataFrame
    installed_languages = argostranslate.translate.get_installed_languages()
    to_code = "en"
    
    # Create a mapping: {from_lang_code: translator_object}
    translators = {}
    for lang in set(df['lang']):
        lang = lang.split('-')[0]
        try:
            from_lang = next(l for l in installed_languages if l.code == lang)
            to_lang = next(l for l in installed_languages if l.code == to_code)
            translators[lang] = from_lang.get_translation(to_lang)
        except StopIteration:
            translators[lang] = None  # No model installed for this pair

    # extract body, html, clean text, translate text
    df['email_text'] = df['mime'].apply(extract_data_from_raw_mime)
    df['url_domains'] = df['email_text'].apply(extract_parent_domains)
    df['cleaned_text'] = df['email_text'].apply(clean_text)
    df['translated_text'] = df.apply(lambda x: translate_text(
        translate_text(clean_text(x['cleaned_text']), x['lang'])
    )
                                     
    df['subject'] = df.apply(lambda x: translate_text(
        translate_text(clean_text(x['subject']))
        , x['lang'])
    )

    # write to repo
    dst_path = ''
    df.to_csv(dst_path, index=False)
    
    