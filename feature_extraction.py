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
import re
import csv
import socket
import ssl
import requests

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
    return domain

def fetch_url(url, output_dir):
    # Create a unique filename based on the URL
    filename = os.path.join(output_dir, f"{hash(url)}.html")
    command = [
        "wget",
        "--no-verbose",          # Reduce output
        "--no-clobber",          # Do not overwrite existing files
        "--timeout=10",          # Set timeout to 10 seconds
        "--tries=2",             # Retry twice if the download fails
        "--max-redirect=1",      # Limit to the first redirect
        "--output-document", filename,
        url
    ]
    try:
        subprocess.run(command, check=True)
        return filename
    except subprocess.CalledProcessError as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def extract_features(url, html_content):
    features = {}
    soup = BeautifulSoup(html_content, 'html.parser')
    parsed_url = urlparse(url)
    domain = get_domain(url)

    # 1. URL-Based Features
    features['url_length'] = len(url)
    features['num_subdomains'] = domain.count('.')
    features['is_ip_address'] = int(is_ip_address(domain))
    features['uses_https'] = int(parsed_url.scheme == 'https')

    # Presence of '@' symbol in URL
    features['has_at_symbol'] = int('@' in url)
    # Presence of '-' symbol in domain
    features['has_hyphen_in_domain'] = int('-' in domain)
    # Number of digits in URL
    features['num_digits_in_url'] = sum(c.isdigit() for c in url)
    # Presence of redirect ('//') in URL path
    features['double_slash_redirect'] = int(url.rfind('//') > 6)

    # 2. Domain Age (Placeholder: Requires WHOIS lookup, which may not be feasible)
    # features['domain_age_days'] = get_domain_age(domain)

    # 3. HTML Content Features
    features['num_script_tags'] = len(soup.find_all('script'))
    features['num_iframe_tags'] = len(soup.find_all('iframe'))
    features['num_embed_tags'] = len(soup.find_all('embed'))
    features['num_object_tags'] = len(soup.find_all('object'))
    features['num_form_tags'] = len(soup.find_all('form'))

    # Number of external resources
    features['num_external_resources'] = 0
    for tag in soup.find_all(['img', 'script', 'link']):
        src = tag.get('src') or tag.get('href')
        if src and src.startswith('http') and get_domain(src) != domain:
            features['num_external_resources'] += 1

    # Presence of forms with action pointing to external domains
    forms = soup.find_all('form', action=True)
    features['external_forms'] = sum(
        1 for form in forms if not form['action'].startswith('/') and
        get_domain(form['action']) != domain
    )

    # Obfuscated JavaScript code detection
    script_texts = [script.get_text() for script in soup.find_all('script')]
    script_content = ' '.join(script_texts)
    features['has_eval'] = int('eval(' in script_content)
    features['has_escape'] = int('escape(' in script_content)
    features['has_obfuscation'] = int(re.search(r'([a-zA-Z]\d{3,})', script_content) is not None)
    features['has_unescape'] = int('unescape(' in script_content)
    features['has_exec'] = int('exec(' in script_content)

    # 4. JavaScript Features
    features['uses_document_cookie'] = int('document.cookie' in script_content)
    features['uses_window_location'] = int('window.location' in script_content)
    features['uses_settimeout'] = int('setTimeout(' in script_content)
    features['uses_setinterval'] = int('setInterval(' in script_content)
    features['uses_prompt'] = int('prompt(' in script_content)
    features['uses_alert'] = int('alert(' in script_content)
    features['uses_confirm'] = int('confirm(' in script_content)

    # 5. Metadata Features
    meta_tags = soup.find_all('meta')
    features['num_meta_tags'] = len(meta_tags)
    features['has_meta_refresh'] = int(any('http-equiv' in meta.attrs and meta.attrs['http-equiv'].lower() == 'refresh' for meta in meta_tags))
    features['has_meta_keywords'] = int(any('name' in meta.attrs and meta.attrs['name'].lower() == 'keywords' for meta in meta_tags))
    features['has_meta_description'] = int(any('name' in meta.attrs and meta.attrs['name'].lower() == 'description' for meta in meta_tags))

    # 6. Image Analysis
    images = soup.find_all('img', src=True)
    features['num_images'] = len(images)
    features['num_external_images'] = sum(
        1 for img in images if img['src'].startswith('http') and get_domain(img['src']) != domain
    )
    # Use of data URLs for images
    features['num_data_images'] = sum(
        1 for img in images if img['src'].startswith('data:')
    )

    # 7. SSL Certificate Features (if HTTPS)
    if parsed_url.scheme == 'https':
        try:
            cert = ssl.get_server_certificate((domain, 443))
            x509 = ssl._ssl._test_decode_cert(cert)
            issuer = x509.get('issuer')
            features['ssl_certificate_issuer'] = str(issuer)
            # Check for common issuers (simplified)
            features['is_cert_verified'] = int('Let\'s Encrypt' in str(issuer) or 'DigiCert' in str(issuer))
        except Exception as e:
            features['ssl_certificate_issuer'] = 'Error'
            features['is_cert_verified'] = 0
    else:
        features['ssl_certificate_issuer'] = 'Not HTTPS'
        features['is_cert_verified'] = 0

    # 8. Presence of Copyright Info
    features['has_copyright'] = int(bool(re.search(r'©|&copy;|Copyright', html_content, re.I)))

    # 9. Content Length
    features['content_length'] = len(html_content)

    # 10. Number of Links
    links = soup.find_all('a', href=True)
    features['num_links'] = len(links)
    features['num_external_links'] = sum(
        1 for link in links if link['href'].startswith('http') and get_domain(link['href']) != domain
    )
    features['num_internal_links'] = features['num_links'] - features['num_external_links']

    # 11. Number of Empty Links
    features['num_empty_links'] = sum(
        1 for link in links if link['href'] in ['#', '']
    )

    # 12. Favicon
    features['has_favicon'] = int(bool(soup.find('link', rel='shortcut icon') or soup.find('link', rel='icon')))

    # 13. Robots.txt
    robots_url = parsed_url.scheme + '://' + domain + '/robots.txt'
    try:
        robots_response = requests.get(robots_url, timeout=5)
        features['has_robots_txt'] = int(robots_response.status_code == 200)
    except:
        features['has_robots_txt'] = 0

    # 14. Sitemap.xml
    sitemap_url = parsed_url.scheme + '://' + domain + '/sitemap.xml'
    try:
        sitemap_response = requests.get(sitemap_url, timeout=5)
        features['has_sitemap_xml'] = int(sitemap_response.status_code == 200)
    except:
        features['has_sitemap_xml'] = 0

    # 15. Email Addresses in Page
    features['num_emails_in_page'] = len(re.findall(r'[\w\.-]+@[\w\.-]+', html_content))

    # 16. WHOIS Registration Length (Placeholder)
    # features['whois_registration_length'] = get_whois_registration_length(domain)

    # 17. Page Title Length
    if soup.title:
        features['page_title_length'] = len(soup.title.string)
    else:
        features['page_title_length'] = 0

    # 18. Presence of Login Forms
    features['has_login_form'] = int(bool(soup.find('input', {'type': 'password'})))

    # 19. Presence of Social Media Links
    social_media_domains = ['facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com', 'youtube.com']
    features['has_social_media_links'] = 0
    for link in links:
        href = link['href']
        if any(social_domain in href for social_domain in social_media_domains):
            features['has_social_media_links'] = 1
            break

    # 20. Page Redirects
    # Since we limit to first redirect, we can check meta refresh
    features['has_meta_redirect'] = int(features['has_meta_refresh'])

    # 21. HTTPS Token in URL
    features['https_in_url'] = int('https' in parsed_url.netloc)

    # 22. URL Anchor Tags
    features['num_anchor_tags'] = len(soup.find_all('a'))

    # 23. Shortening Services in URL
    shortening_services = ['bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co']
    features['uses_shortening_service'] = int(any(service in domain for service in shortening_services))

    # Additional features can be added here as needed

    return features

def main():
    # Load the original CSV file
    df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')

    # Create a directory to store fetched HTML pages
    output_dir = 'html_pages'
    os.makedirs(output_dir, exist_ok=True)

    # List to collect feature dictionaries
    features_list = []

    # Iterate over each URL in the DataFrame
    for idx, row in df.iterrows():
        url = row['URL']
        label = row['label']
        print(f"Processing URL {idx + 1}/{len(df)}: {url}")

        # Fetch the HTML content safely
        html_file = fetch_url(url, output_dir)
        if html_file and os.path.exists(html_file):
            try:
                with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
                # Extract features from the HTML content
                features = extract_features(url, html_content)
                # Add the URL and label to the features
                features['URL'] = url
                features['label'] = label
                features_list.append(features)
            except Exception as e:
                print(f"Error processing HTML for URL {url}: {e}")
                continue
        else:
            print(f"Failed to fetch URL {url}")
            continue

    # Create a DataFrame from the features list
    features_df = pd.DataFrame(features_list)

    # Save the features to a new CSV file
    features_df.to_csv('features_dataset.csv', index=False)
    print("Feature extraction complete. Saved to features_dataset.csv")

if __name__ == "__main__":
    main()
