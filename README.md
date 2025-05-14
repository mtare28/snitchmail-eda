# snitchmail-eda
Repo to perform EDA for SnitchMail

## Python Files

### process_good_user_emails.py

* Picks up json files from Accsec S3 location
* Processes 10 at a time (500 emails each)
* Extract raw columns
* Get country from IP address (`GeoIP2-Country.mmdb` mapping)
* Get language from subject text
* Extract email text from raw mime
* Extract HTML from raw mime and store as a file

### process_raw_mime.py

* Process 1 parquet DF at a time
* Process email text better
    * Extract email text from raw mime
    * Extract URL domains from email text
    * Extract clean text from email text with link domains
    * Translate clean text
    * Translate subject

### llm_model_inference.py

* Load Source dataframe from parquet file
* Create `subject_body` column from `subject` and `translated_text` column
* Alternately create `url_domains` column from `email_text` column
* Pass the text/URL domains to the model as input for inference
* Save the label and Confidence score given by model
* Save file to local or S3 location as parquet


## Notebooks

### File data.ipynb

* Fetch data from Sendgrid S3 location and dump in S3
* Process Suspended Users
    * Read raw mime
    * Extract text and html
    * Save HTML file to S3
    * Prepare combined parque
    * Save to Accsec S3

### File opensource_model_inference.ipynb

* Explore the performance of 3 open source models on phish user (5.5K) and good user data (20K)
    * ealvaradob_bert
    * elslay_bert
    * cybersectony_distilbert

### File generate_sheet_for_labeling.ipynb

* Generate csv file with links to HTML files stored in drive folder
* Later to be downloaded and shared with Reviewers for labeling
* Filename to File ID map in `phish_html_files_map.json`

### File compare_embeddings.ipynb

* Create embeddings for Open Source Dataset (Phish and Not-Phish)
* Download our data embeddings from S3
* Compare using Mean Embeddings
* Compare using Mean Cosine Distance
* Visualize Embeddings using UMap
* Visualize Embeddings using PCA and KMeans Clustering