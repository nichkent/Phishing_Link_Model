### IMPORTANT NOTE ---------------------------------
### DO NOT RUN THIS FILE IN A NON-SECURE ENVIRONMENT!!!
### IT MAY DOWNLOAD MALWARE, WHICH NOBODY WANTS!!!
###
#---------------------------------------------------

import pandas as pd
import os
import subprocess
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import urllib.robotparser
import re
import csv
import socket
import ssl
import requests
import logging
from multiprocessing import Process, Queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def is_ip_address(domain):
    # Check if the domain is an IP address
    try:
        socket.inet_aton(domain)
        return True
    except socket.error:
        return False

def get_domain(url):
    # Extract the domain from the URL
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    # Remove port number if present
    domain = domain.split(':')[0]
    return domain.lower()

def is_allowed_by_robots(user_agent, url, timeout=5):
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    try:
        response = requests.get(robots_url, headers={'User-Agent': user_agent}, timeout=timeout)
        if response.status_code == 200:
            rp.parse(response.text.splitlines())
            can_fetch = rp.can_fetch(user_agent, url)
            return can_fetch
        else:
            logging.info(f"robots.txt not found or not accessible for {url}. Assuming allowed.")
            return True
    except Exception as e:
        logging.info(f"Exception fetching robots.txt for {url}: {e}. Assuming allowed.")
        return True

def fetch_url(url, output_dir, timeout=5, user_agent='MyBot/1.0'):
    # Create a unique filename based on the URL
    filename = os.path.join(output_dir, f"{hash(url)}.html")
    command = [
        "wget",
        "--no-verbose",
        "--no-clobber",
        f"--timeout={timeout}",
        f"--dns-timeout={timeout}",
        f"--connect-timeout={timeout}",
        f"--read-timeout={timeout}",
        "--tries=1",
        "--max-redirect=1",
        "--execute=robots=on",
        "--user-agent", user_agent,
        "--output-document", filename,
        url
    ]
    logging.info(f"Fetching URL: {url}")
    try:
        subprocess.run(command, check=True, timeout=timeout + 5)
        logging.info(f"Successfully fetched URL: {url}")
        return filename
    except subprocess.TimeoutExpired:
        logging.warning(f"Timeout expired when fetching URL {url}")
        return None
    except subprocess.CalledProcessError as e:
        logging.warning(f"Error fetching URL {url}: {e}")
        return None

def extract_features(url, html_content):
    features = {}
    logging.info(f"Extracting features from URL: {url}")
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        parsed_url = urlparse(url)
        domain = get_domain(url)

        # 1. URL-Based Features
        features['url_length'] = len(url)
        features['num_subdomains'] = domain.count('.')
        features['is_ip_address'] = int(is_ip_address(domain))
        features['uses_https'] = int(parsed_url.scheme == 'https')
        features['has_at_symbol'] = int('@' in url)
        features['has_hyphen_in_domain'] = int('-' in domain)
        features['num_digits_in_url'] = sum(c.isdigit() for c in url)
        features['double_slash_redirect'] = int(url.rfind('//') > 6)
        features['https_in_url'] = int('https' in parsed_url.netloc.lower())
        shortening_services = ['bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co']
        features['uses_shortening_service'] = int(any(service in domain for service in shortening_services))

        # 2. HTML Content Features
        features['num_script_tags'] = len(soup.find_all('script'))
        features['num_iframe_tags'] = len(soup.find_all('iframe'))
        features['num_embed_tags'] = len(soup.find_all('embed'))
        features['num_object_tags'] = len(soup.find_all('object'))
        features['num_form_tags'] = len(soup.find_all('form'))
        features['num_anchor_tags'] = len(soup.find_all('a'))

        # Number of external resources
        features['num_external_resources'] = 0
        for tag in soup.find_all(['img', 'script', 'link']):
            src = tag.get('src') or tag.get('href')
            if src and src.startswith('http') and get_domain(src) != domain:
                features['num_external_resources'] += 1

        # Forms pointing to external domains
        forms = soup.find_all('form', action=True)
        features['external_forms'] = sum(
            1 for form in forms if not form['action'].startswith('/') and
            get_domain(form['action']) != domain
        )

        # 3. JavaScript Features
        script_texts = [script.get_text() for script in soup.find_all('script')]
        script_content = ' '.join(script_texts)
        features['has_eval'] = int('eval(' in script_content)
        features['has_escape'] = int('escape(' in script_content)
        features['has_obfuscation'] = int(re.search(r'([a-zA-Z]\d{3,})', script_content) is not None)
        features['has_unescape'] = int('unescape(' in script_content)
        features['has_exec'] = int('exec(' in script_content)
        features['uses_document_cookie'] = int('document.cookie' in script_content)
        features['uses_window_location'] = int('window.location' in script_content)
        features['uses_settimeout'] = int('setTimeout(' in script_content)
        features['uses_setinterval'] = int('setInterval(' in script_content)
        features['uses_prompt'] = int('prompt(' in script_content)
        features['uses_alert'] = int('alert(' in script_content)
        features['uses_confirm'] = int('confirm(' in script_content)

        # 4. Metadata Features
        meta_tags = soup.find_all('meta')
        features['num_meta_tags'] = len(meta_tags)
        features['has_meta_refresh'] = int(any('http-equiv' in meta.attrs and meta.attrs['http-equiv'].lower() == 'refresh' for meta in meta_tags))
        features['has_meta_keywords'] = int(any('name' in meta.attrs and meta.attrs['name'].lower() == 'keywords' for meta in meta_tags))
        features['has_meta_description'] = int(any('name' in meta.attrs and meta.attrs['name'].lower() == 'description' for meta in meta_tags))

        # 5. Image Analysis
        images = soup.find_all('img', src=True)
        features['num_images'] = len(images)
        features['num_external_images'] = sum(
            1 for img in images if img['src'].startswith('http') and get_domain(img['src']) != domain
        )
        features['num_data_images'] = sum(
            1 for img in images if img['src'].startswith('data:')
        )

        # 6. SSL Certificate Features (if HTTPS)
        if parsed_url.scheme == 'https':
            try:
                context = ssl.create_default_context()
                with socket.create_connection((domain, 443), timeout=5) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        cert = ssock.getpeercert()
                        issuer = dict(x[0] for x in cert['issuer'])
                        features['ssl_certificate_issuer'] = issuer.get('organizationName', 'Unknown')
                        features['is_cert_verified'] = int(issuer.get('organizationName') in ['Let\'s Encrypt', 'DigiCert'])
            except Exception as e:
                features['ssl_certificate_issuer'] = 'Error'
                features['is_cert_verified'] = 0
        else:
            features['ssl_certificate_issuer'] = 'Not HTTPS'
            features['is_cert_verified'] = 0

        # 7. Additional Features
        features['has_copyright'] = int(bool(re.search(r'©|&copy;|Copyright', html_content, re.I)))
        features['content_length'] = len(html_content)
        links = soup.find_all('a', href=True)
        features['num_links'] = len(links)
        features['num_external_links'] = sum(
            1 for link in links if link['href'].startswith('http') and get_domain(link['href']) != domain
        )
        features['num_internal_links'] = features['num_links'] - features['num_external_links']
        features['num_empty_links'] = sum(
            1 for link in links if link['href'] in ['#', '']
        )
        features['has_favicon'] = int(bool(soup.find('link', rel='shortcut icon') or soup.find('link', rel='icon')))
        features['has_login_form'] = int(bool(soup.find('input', {'type': 'password'})))
        social_media_domains = ['facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com', 'youtube.com']
        features['has_social_media_links'] = int(any(
            any(social_domain in link['href'] for social_domain in social_media_domains)
            for link in links
        ))
        features['has_meta_redirect'] = int(features['has_meta_refresh'])
        features['num_emails_in_page'] = len(re.findall(r'[\w\.-]+@[\w\.-]+', html_content))

        # 8. Page Title Length
        if soup.title and soup.title.string:
            features['page_title_length'] = len(soup.title.string)
        else:
            features['page_title_length'] = 0

        logging.info(f"Successfully extracted features for URL: {url}")
        return features

    except Exception as e:
        logging.error(f"Error during feature extraction for URL {url}: {e}")
        return None

def extract_features_with_timeout(url, html_content, timeout=10):
    def worker(q):
        features = extract_features(url, html_content)
        q.put(features)

    q = Queue()
    p = Process(target=worker, args=(q,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        logging.warning(f"Timeout occurred while extracting features for URL {url}")
        p.terminate()
        p.join()
        return None
    else:
        return q.get()

def main():
    user_agent = 'MyPhishingDetectorBot/1.0 (+mailto:myemail@example.com)'
    headers = {'User-Agent': user_agent}
    output_file = 'features_dataset.csv'
    fieldnames = ['URL', 'label'] + [
        'url_length', 'num_subdomains', 'is_ip_address', 'uses_https', 'has_at_symbol',
        'has_hyphen_in_domain', 'num_digits_in_url', 'double_slash_redirect', 'https_in_url',
        'uses_shortening_service', 'num_script_tags', 'num_iframe_tags', 'num_embed_tags',
        'num_object_tags', 'num_form_tags', 'num_anchor_tags', 'num_external_resources',
        'external_forms', 'has_eval', 'has_escape', 'has_obfuscation', 'has_unescape',
        'has_exec', 'uses_document_cookie', 'uses_window_location', 'uses_settimeout',
        'uses_setinterval', 'uses_prompt', 'uses_alert', 'uses_confirm', 'num_meta_tags',
        'has_meta_refresh', 'has_meta_keywords', 'has_meta_description', 'num_images',
        'num_external_images', 'num_data_images', 'ssl_certificate_issuer', 'is_cert_verified',
        'has_copyright', 'content_length', 'num_links', 'num_external_links',
        'num_internal_links', 'num_empty_links', 'has_favicon', 'has_login_form',
        'has_social_media_links', 'has_meta_redirect', 'num_emails_in_page',
        'page_title_length'
    ]

    # Load the original dataset
    df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')

    # Desired sample size
    sample_size = 1000

    # Calculate class proportions
    phishing_ratio = df['label'].mean()
    legitimate_ratio = 1 - phishing_ratio

    # Number of samples per class
    phishing_count = int(phishing_ratio * sample_size)
    legitimate_count = sample_size - phishing_count

    # Stratified sampling
    phishing_sample = df[df['label'] == 1].sample(n=phishing_count, random_state=42)
    legitimate_sample = df[df['label'] == 0].sample(n=legitimate_count, random_state=42)

    # Combine and shuffle
    sample_df = pd.concat([phishing_sample, legitimate_sample]).sample(frac=1, random_state=42)

    # Create a directory to store fetched HTML pages
    output_dir = 'html_pages'
    os.makedirs(output_dir, exist_ok=True)

    # Open the output CSV file in write mode
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over each URL in the sample
        for count, (idx, row) in enumerate(sample_df.iterrows(), 1):
            url = row['URL']
            label = row['label']

            logging.info(f"Processing URL {count}/{len(sample_df)}: {url}")

            features = {}
            features['URL'] = url
            features['label'] = label

            try:
                # Check robots.txt
                allowed = is_allowed_by_robots(user_agent, url, timeout=5)
                if not allowed:
                    logging.info(f"Disallowed by robots.txt: {url}")
                    # Assign default values for features that cannot be extracted
                    default_feature_values = {fname: 0 for fname in fieldnames if fname not in ['URL', 'label']}
                    features.update(default_feature_values)
                    writer.writerow(features)
                    csvfile.flush()
                    continue

                # Fetch the HTML content
                html_file = fetch_url(url, output_dir, timeout=5, user_agent=user_agent)
                if html_file and os.path.exists(html_file):
                    with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                        html_content = f.read()
                    # Extract features with timeout
                    extracted_features = extract_features_with_timeout(url, html_content, timeout=10)
                    if extracted_features is None:
                        logging.info(f"Skipping URL {url} due to feature extraction timeout.")
                        # Assign default values
                        default_feature_values = {fname: 0 for fname in fieldnames if fname not in ['URL', 'label']}
                        features.update(default_feature_values)
                    else:
                        # Update features
                        features.update(extracted_features)
                else:
                    logging.info(f"Failed to fetch URL {url}")
                    # Assign default values
                    default_feature_values = {fname: 0 for fname in fieldnames if fname not in ['URL', 'label']}
                    features.update(default_feature_values)

                # Write features to CSV
                writer.writerow(features)
                # Flush after each write to ensure data is saved
                csvfile.flush()

            except Exception as e:
                logging.error(f"Error processing URL {url}: {e}")
                # Assign default values
                default_feature_values = {fname: 0 for fname in fieldnames if fname not in ['URL', 'label']}
                features.update(default_feature_values)
                # Write features to CSV
                writer.writerow(features)
                csvfile.flush()
                continue

    logging.info("Feature extraction complete. Saved to features_dataset.csv")

if __name__ == "__main__":
    main()
