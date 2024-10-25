### IMPORTANT NOTE ---------------------------------
### DO NOT RUN THIS FILE IN A NON-SECURE ENVIRONMENT!!!
### IT MAY DOWNLOAD MALWARE, WHICH NOBODY WANTS!!!
###
#---------------------------------------------------

# Imports
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

### The following functions are all for feature extraction ###
def is_ip_address(domain):
    """
    is_ip_address Checks if the current host is an IP address

    :param domain:string Domain name of the URL.
    :return: returns true or false if the domain name is associated with an IP address.
    """
    # Check if the domain is an IP address
    try:
        socket.inet_aton(domain)
        return True
    except socket.error:
        return False

def get_domain(url):
    """
    get_domain takes a url and returns the domain name in lowercase.

    :param url:string Name of the current URL.
    :return: domain name in lowercase.
    """
    # Extract the domain from the URL
    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    # Remove port number if present
    domain = domain.split(':')[0]

    return domain.lower()

def is_allowed_by_robots(user_agent, url, timeout=5):
    """
    is_allowed_by_robots checks the robots.txt page of the site to ensure that this scripts webscraping is in compliance.

    :param user_agent:String Name of my webscrapper.
    :param url:String Current URL.
    :param timeout:Int How long it will check for a response before stopping.
    :return: Returns True or False if the webscraper is allowed by robots.txt or not.
    """
    # Check for the presense of the robots.txt file.
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()

    try:
        # Wait for a response from the website
        response = requests.get(robots_url, headers={'User-Agent': user_agent}, timeout=timeout)

        # If the webpage loads
        if response.status_code == 200:
            # Parse robots.txt
            rp.parse(response.text.splitlines())
            can_fetch = rp.can_fetch(user_agent, url)

            # Return True or False based on the robots.txt file
            return can_fetch
        else:
            # Assume allowed if robots.txt is not present
            logging.info(f"robots.txt not found or not accessible for {url}. Assuming allowed.")
            return True
    except Exception as e:
        # If an exception is thrown, as in, the file is corrupted. Assume allowed.
        logging.info(f"Exception fetching robots.txt for {url}: {e}. Assuming allowed.")
        return True

def fetch_url(url, output_dir, timeout=5, user_agent='MyBot/1.0'):
    """
    fetch_url Grabs the current URL from the dataset for processing.

    :param url:String Current URL.
    :param output_dir:String Stores the output directory of the URL.
    :param timeout:Int How long before the Scraper moves on.
    :param user_agent:String Name of the Web Scraper.
    :return: The filename or none if the dataset is not found.
    """
    # Create a unique filename based on the URL
    filename = os.path.join(output_dir, f"{hash(url)}.html")

    # Create the command line request
    command = [
        "wget",                             # Grab the first pages code
        "--no-verbose",                     # Make the responses short
        "--no-clobber",                     # Append mode to a file
        f"--timeout={timeout}",             # Timeout after a default of 5 seconds performing the command
        f"--dns-timeout={timeout}",         # Timeout after a default of 5 seconds performing the command
        f"--connect-timeout={timeout}",     # Timeout after a default of 5 seconds performing the command
        f"--read-timeout={timeout}",        # Timeout after a default of 5 seconds performing the command
        "--tries=1",                        # Try to extract features only once
        "--max-redirect=1",                 # Grab the first redirect page
        "--execute=robots=on",              # Follow robots.txt
        "--user-agent", user_agent,         # Give the webpage the name of the user-agent
        "--output-document", filename,      # Filename to output the information to
        url
    ]

    # Print to the terminal
    logging.info(f"Fetching URL: {url}")

    try:
        # Run until 10 seconds have passed
        subprocess.run(command, check=True, timeout=timeout + 5)

        # Tell the user if the url could be found
        logging.info(f"Successfully fetched URL: {url}")

        # Return the output file if found
        return filename
    except subprocess.TimeoutExpired:
        logging.warning(f"Timeout expired when fetching URL {url}")
        return None
    except subprocess.CalledProcessError as e:
        logging.warning(f"Error fetching URL {url}: {e}")
        return None

def extract_features(url, html_content):
    """
    extract_features takes a url and extracts features from the HTML content based on the first redirect page's code.
    :param url:String Current URL.
    :param html_content:String Content of the HTML page received from wget.
    :return: Features found in the HTML page. Or none if no features are present.
    """
    # Initialize features set
    features = {}

    # Inform the user
    logging.info(f"Extracting features from URL: {url}")

    try:
        # Use beautifulSoup to parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')
        # Grab the URL
        parsed_url = urlparse(url)
        # Grab the domain
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
        # Most common shortening services
        shortening_services = ['bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co']
        features['uses_shortening_service'] = int(any(service in domain for service in shortening_services))

        # 2. HTML Content Features
        features['num_script_tags'] = len(soup.find_all('script'))
        features['num_iframe_tags'] = len(soup.find_all('iframe'))
        features['num_embed_tags'] = len(soup.find_all('embed'))
        features['num_object_tags'] = len(soup.find_all('object'))
        features['num_form_tags'] = len(soup.find_all('form'))
        features['num_anchor_tags'] = len(soup.find_all('a'))

        ### Number of external resources
        # Assume 0 initially
        features['num_external_resources'] = 0

        ### Find tags in HTML
        for tag in soup.find_all(['img', 'script', 'link']):
            src = tag.get('src') or tag.get('href')
            # If found increment num_external_resources
            if src and src.startswith('http') and get_domain(src) != domain:
                features['num_external_resources'] += 1

        ### Forms pointing to external domains
        forms = soup.find_all('form', action=True)
        # Count all forms pointing to external domains
        features['external_forms'] = sum(1 for form in forms if not form['action'].startswith('/') and get_domain(form['action']) != domain)

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
        features['num_external_images'] = sum(1 for img in images if img['src'].startswith('http') and get_domain(img['src']) != domain)
        features['num_data_images'] = sum(1 for img in images if img['src'].startswith('data:'))

        # 6. SSL Certificate Features (if HTTPS)
        # Check for https, if not then mark as 0
        if parsed_url.scheme == 'https':
            try:
                # Create default context for ssl request
                context = ssl.create_default_context()
                # Create a socket, timeout if it takes too long
                with socket.create_connection((domain, 443), timeout=5) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        # Grab the cert
                        cert = ssock.getpeercert()
                        # Grab the cert issuer
                        issuer = dict(x[0] for x in cert['issuer'])
                        # Verify the cert issuer
                        features['ssl_certificate_issuer'] = issuer.get('organizationName', 'Unknown')
                        features['is_cert_verified'] = int(issuer.get('organizationName') in ['Let\'s Encrypt', 'DigiCert'])
            except Exception as e:
                # If exception mark as 0
                features['ssl_certificate_issuer'] = 'Error'
                features['is_cert_verified'] = 0
        else:
            features['ssl_certificate_issuer'] = 'Not HTTPS'
            features['is_cert_verified'] = 0

        # 7. Additional Features
        # Check for copyright mark
        features['has_copyright'] = int(bool(re.search(r'©|&copy;|Copyright', html_content, re.I)))

        # Grab html Length
        features['content_length'] = len(html_content)
        links = soup.find_all('a', href=True)

        # Grab number of links on the page
        features['num_links'] = len(links)

        # Grab number of external links on the page
        features['num_external_links'] = sum(1 for link in links if link['href'].startswith('http') and get_domain(link['href']) != domain)

        # Grab number of internal links on the page
        features['num_internal_links'] = features['num_links'] - features['num_external_links']

        # Grab number of empty links on the page
        features['num_empty_links'] = sum(1 for link in links if link['href'] in ['#', ''])

        # Check for if the page has a favicon or not
        features['has_favicon'] = int(bool(soup.find('link', rel='shortcut icon') or soup.find('link', rel='icon')))

        # Check for a login form
        features['has_login_form'] = int(bool(soup.find('input', {'type': 'password'})))

        # Check if the domain is a well known social media site
        social_media_domains = ['facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com', 'youtube.com']
        features['has_social_media_links'] = int(any(
            any(social_domain in link['href'] for social_domain in social_media_domains)
            for link in links
        ))

        # Check for redirects in the meta data
        features['has_meta_redirect'] = int(features['has_meta_refresh'])

        # Check for emails present in the page
        features['num_emails_in_page'] = len(re.findall(r'[\w\.-]+@[\w\.-]+', html_content))

        # 8. Page Title Length
        if soup.title and soup.title.string:
            features['page_title_length'] = len(soup.title.string)
        else:
            features['page_title_length'] = 0

        logging.info(f"Successfully extracted features for URL: {url}")

        # Return all features found for the dataset
        return features

    except Exception as e:
        # Otherwise return nothing for the features
        logging.error(f"Error during feature extraction for URL {url}: {e}")
        return None

def extract_features_with_timeout(url, html_content, timeout=10):
    """
    extract_features_with_timeout Extract the found features from the website. Returns the data in a Queue object for storage.

    :param url:String
    :param html_content:String
    :param timeout:Int
    :return: Returns the found feature data from the website in a Queue object.
    """
    # Define a worker function for frequent calling within the function extract_features_with_timeout
    def worker(q):
        # Use extract_features to extract the features from the html_content
        features = extract_features(url, html_content)

        # Put the found features in a queue
        q.put(features)

    # Create the queue object
    q = Queue()

    # Process the queue object using the worker function defined above
    p = Process(target=worker, args=(q,))
    p.start()

    # Try to join all the data for 10 secs.
    p.join(timeout)

    # If still running after 10 sec kill the run and just return what was processed.
    if p.is_alive():
        logging.warning(f"Timeout occurred while extracting features for URL {url}")
        p.terminate()
        p.join()
        return None
    else:
        return q.get()

def main():
    # Define the name of the webscraper
    user_agent = 'MyPhishingDetectorBot/1.0 (+mailto:nicholasiankent@gmail.com)'

    # Create the header for the web scraper
    headers = {'User-Agent': user_agent}

    # Define the output file
    output_file = 'features_dataset.csv'

    # Define th fields we are looking for and what to join the new features on
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

    # Inform the user
    logging.info("Feature extraction complete. Saved to features_dataset.csv")

# Call main
if __name__ == "__main__":
    main()
