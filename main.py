# ==============================================================================
#                      Local-First Intelligent Scraper (COMPLETE & FINAL)
# ==============================================================================
# --- Standard Library Imports ---
import asyncio
import csv
import hashlib
import json
import logging
import multiprocessing
import os
import re
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from urllib.parse import urlencode, urljoin, urlparse, parse_qs
import collections # Added for collections.deque for potential use in advanced queue management
import random # For random user agents
import io # For image processing bytes
import shutil # For cleaning up temporary directories

# --- Third-party Library Imports ---
import aiohttp # For asynchronous HTTP requests (faster than requests for many small fetches)
import requests # Fallback for synchronous HTTP requests, or specific needs
from bs4 import BeautifulSoup # For parsing HTML content
from flask import Flask, abort, jsonify, request, send_from_directory, Response # Flask web framework components
from flask_cors import CORS # For handling Cross-Origin Resource Sharing
from selenium import webdriver # For browser automation (dynamic content scraping)
from selenium.common.exceptions import (NoSuchElementException, TimeoutException,
                                        WebDriverException) # Specific Selenium exceptions for error handling
from selenium.webdriver.chrome.service import Service as ChromeService # Service object for ChromeDriver
from selenium.webdriver.common.by import By # Locating elements in HTML
from selenium.webdriver.support import expected_conditions as EC # Predefined conditions for WebDriverWait
from selenium.webdriver.support.ui import WebDriverWait # Waiting for elements to appear on dynamic pages
from webdriver_manager.chrome import ChromeDriverManager # Automatically downloads and manages ChromeDriver

# Optional: For image processing (resizing, quality)
try:
    from PIL import Image
    IMAGE_PROCESSING_ENABLED = True
except ImportError:
    # Log a warning if Pillow is not installed, as image features will be disabled.
    print("WARNING: Pillow not found. Image processing features (quality, format) will be disabled.", file=sys.__stderr__, flush=True)
    print("To enable, run: pip install Pillow", file=sys.__stdout__, flush=True)
    IMAGE_PROCESSING_ENABLED = False

# Optional: For system monitoring (CPU, Memory usage)
try:
    import psutil
    SYSTEM_MONITORING_ENABLED = True
except ImportError:
    # Log a warning if psutil is not installed, as system monitoring will be disabled.
    print("WARNING: psutil not found. System resource monitoring will be disabled.", file=sys.__stderr__, flush=True)
    print("To enable, run: pip install psutil", file=sys.__stdout__, flush=True)
    SYSTEM_MONITORING_ENABLED = False

# Optional: For local AI/ML (Summarization)
try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    LOCAL_AI_ENABLED = True
except ImportError:
    # Log a warning if Transformers or PyTorch are not installed, as local AI features will be disabled.
    print("WARNING: Transformers or PyTorch not found. Local AI features will be disabled.", file=sys.__stderr__, flush=True)
    print("To enable, run: pip install transformers torch torchvision", file=sys.__stdout__, flush=True)
    LOCAL_AI_ENABLED = False


# ==============================================================================
#                 SECTION 1: FLASK APP AND ROUTE DEFINITIONS
# ==============================================================================

# --- Flask App Setup ---
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app) # Enable CORS for frontend communication to allow requests from different origins

# These global variables will be initialized inside the main execution block (`if __name__ == '__main__':`)
# and shared across processes using multiprocessing.Manager().
# They are declared here to make their scope clear, but their actual Manager-backed
# objects are assigned in the main block.
scraper_process = None # Holds the multiprocessing.Process object for the scraper
scraper_status = None # multiprocessing.Manager.dict to store real-time scraper status
scraped_products_data = None # multiprocessing.Manager.list to store scraped product data
discovered_categories_data = None # multiprocessing.Manager.list to store discovered categories
scraper_stop_event = None # multiprocessing.Event to signal scraper to stop gracefully
scraper_fatal_error_event = None # multiprocessing.Event to signal a fatal error in scraper
browser_semaphore = None # multiprocessing.Semaphore to control concurrent browser instances
product_link_queue = None # multiprocessing.Manager.Queue for product URLs to be scraped
category_discovery_queue = None # multiprocessing.Manager.Queue for category URLs to be discovered
dynamic_target_concurrency = None # multiprocessing.Manager.dict to hold dynamically adjusted concurrency target
manager = None # The multiprocessing.Manager instance itself


# --- Static File and API Routes ---

@app.route('/')
def serve_index():
    """
    Serves the main index.html file from the root directory.
    This is the entry point for the web application.
    """
    return send_from_directory('.', 'index.html')

@app.route('/strings.json')
def serve_strings():
    """
    Serves the internationalization strings.json file from the root directory.
    This file contains all the UI text in different languages (currently English).
    """
    return send_from_directory('.', 'strings.json')

@app.route('/favicon.ico')
def favicon():
    """
    Serves the favicon.ico file from the static folder.
    Returns a 204 No Content if the favicon is not found,
    preventing unnecessary 404 errors in browser consoles.
    """
    favicon_path = os.path.join(app.root_path, 'static', 'favicon.ico')
    if os.path.exists(favicon_path):
        return send_from_directory(app.static_folder, 'favicon.ico', mimetype='image/vnd.microsoft.icon')
    else:
        app.logger.warning("Favicon.ico not found at %s", favicon_path)
        return '', 204 # No Content, commonly used for missing favicons

@app.route('/api/status', methods=['GET'])
def get_status():
    """
    Returns the current status, logs, scraped products, and discovered categories
    of the scraper process. This endpoint is polled by the frontend to update the UI.
    """
    # Check if the multiprocessing Manager and shared objects are initialized.
    # This prevents errors if the frontend tries to get status before the server is fully ready.
    if scraper_status is None:
        return jsonify({'status': 'error', 'message': 'Server is not fully initialized. Please wait.'}), 503
    
    # Create copies of shared objects for JSON serialization.
    # Manager.dict and Manager.list are proxy objects and not directly JSON serializable.
    status_copy = scraper_status.copy()
    status_copy['log_entries'] = list(status_copy['log_entries']) # Convert manager.list to regular list
    status_copy['products'] = list(scraped_products_data)         # Convert manager.list to regular list
    status_copy['categories'] = list(discovered_categories_data)   # Convert manager.list to regular list
    
    # Ensure 'processed_links_set' (which is a Manager.dict used as a set) is converted to a regular dict.
    if 'processed_links_set' in status_copy:
        status_copy['processed_links_set'] = dict(status_copy['processed_links_set'])

    # Add the current dynamic concurrency target to the status response for UI display.
    if dynamic_target_concurrency is not None:
        status_copy['current_dynamic_concurrency'] = dynamic_target_concurrency.get('value', 0)
    else:
        status_copy['current_dynamic_concurrency'] = 0 # Default if dynamic concurrency is not active

    return jsonify(status_copy)

@app.route('/api/start', methods=['POST'])
def start_scraper():
    """
    Starts a new scraper process with the provided configuration.
    Prevents starting if a scraper is already active to avoid resource conflicts.
    """
    # Declare global variables that will be reassigned within this function.
    # This is crucial to avoid Python interpreting them as new local variables.
    global scraper_process, browser_semaphore, product_link_queue, category_discovery_queue, dynamic_target_concurrency

    # Check if a scraper process is already running.
    if scraper_process and scraper_process.is_alive():
        app.logger.warning("Attempted to start scraper while already active.")
        return jsonify({'status': 'error', 'message': 'Scraper is already active.'}), 400

    # Get configuration from the frontend's JSON request body.
    config = request.json.get('config', {})
    if not config.get('target_url'):
        app.logger.error("Start request failed: target_url is missing.")
        return jsonify({'status': 'error', 'message': 'Target URL is required.'}), 400

    # Generate a unique ID for this scraper run and determine the output directory.
    # The output directory is based on the target site's domain and the current date.
    scraper_id = f"scraper_{time.strftime('%Y%m%d_%H%M%S')}"
    site_name_slug = urlparse(config['target_url']).netloc.replace('.', '_').replace(':', '_') or 'custom_site'
    output_dir = os.path.join(os.getcwd(), f"{site_name_slug}_data_{time.strftime('%Y%m%d')}")
    os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists

    # --- Load previous state if available ---
    # This allows the scraper to resume from where it left off, improving efficiency for long runs.
    app.logger.info(f"Attempting to load previous state from {output_dir}...")
    loaded_state = load_state_from_disk(output_dir, manager) # Pass manager for re-initializing Manager objects

    # Reset shared status and data, then apply loaded state or initialize fresh.
    # Using .clear() and .update()/.extend() on Manager objects preserves their proxy nature.
    scraper_status.clear()
    scraper_status.update({
        'status': 'starting', 'message': 'Preparing scraper...', 'scraped_product_count': 0,
        'total_product_links_found': 0, 'discovered_categories_count': 0, 'error_count': 0,
        'warning_count': 0, 'critical_count': 0, 'last_updated_at': time.time(),
        'log_entries': manager.list(), # Re-initialize log entries as a Manager.list
        'processed_links_set': manager.dict() # Re-initialize processed links set as a Manager.dict
    })
    scraped_products_data[:] = []         # Clear existing products in the shared list
    discovered_categories_data[:] = []     # Clear existing categories in the shared list
    scraper_stop_event.clear()             # Clear the stop signal
    scraper_fatal_error_event.clear()      # Clear the fatal error signal

    if loaded_state:
        app.logger.info("Previous state loaded successfully. Resuming from last progress.")
        # Update scraper_status with loaded values, ensuring Manager.dict handles updates correctly.
        scraper_status.update(loaded_state['status'])
        # Extend Manager.list objects with loaded data.
        scraper_status['log_entries'].extend(loaded_state['status'].get('log_entries', []))
        scraper_status['processed_links_set'].update(loaded_state['status'].get('processed_links_set', {}))
        scraped_products_data.extend(loaded_state['products'])
        discovered_categories_data.extend(loaded_state['categories'])
        
        # Adjust current counts based on loaded data.
        scraper_status['scraped_product_count'] = len(scraped_products_data)
        scraper_status['discovered_categories_count'] = len(discovered_categories_data)
        scraper_status['total_product_links_found'] = len(scraper_status['processed_links_set']) # Assuming all in set were 'found'
        
        # Re-queue pending categories from loaded state to continue discovery.
        for cat in discovered_categories_data:
            if cat['status'] == 'pending' and not cat['no_products_found']:
                category_discovery_queue.put(cat)

    # Initialize or re-initialize the multiprocessing.Semaphore.
    # This semaphore limits the number of concurrent browser instances.
    max_browsers_limit = config.get('max_concurrent_browsers_max', 1)
    browser_semaphore = manager.Semaphore(max_browsers_limit) 

    # Re-initialize shared queues and dynamic concurrency target.
    # These must be Manager objects to be shared across processes.
    product_link_queue = manager.Queue()
    category_discovery_queue = manager.Queue()
    dynamic_target_concurrency = manager.dict({'value': config.get('max_concurrent_browsers_min', 1)})

    app.logger.info(f"Starting new scraper process with config: {config}")
    app.logger.info(f"Output directory: {output_dir}")

    # Create and start the new multiprocessing.Process.
    # All shared Manager objects and Events are passed as arguments to the target function.
    scraper_process = multiprocessing.Process(
        target=run_scraper_process, # The function to be executed in the new process
        args=(config, output_dir, scraper_id, scraper_status, scraped_products_data, discovered_categories_data, 
              scraper_stop_event, scraper_fatal_error_event, browser_semaphore, 
              product_link_queue, category_discovery_queue, dynamic_target_concurrency)
    )
    scraper_process.start() # Start the process
    return jsonify({'status': 'success', 'message': 'Scraper started successfully.'})

@app.route('/api/stop', methods=['POST'])
def stop_scraper():
    """
    Sends a stop signal to the active scraper process and attempts to terminate it.
    This allows for graceful shutdown and state saving.
    """
    global scraper_process # Declare global to modify the process variable

    # Check if a scraper process is currently active.
    if not (scraper_process and scraper_process.is_alive()):
        app.logger.warning("Attempted to stop scraper while not active.")
        return jsonify({'status': 'error', 'message': 'Scraper is not active.'}), 400

    app.logger.info("Stop request received from frontend.")
    scraper_status['status'] = 'stopping' # Update status in shared memory
    scraper_status['message'] = 'Stopping scraper...'
    scraper_stop_event.set() # Set the event to signal the scraper process to stop

    # Wait for the scraper process to terminate gracefully within a timeout.
    scraper_process.join(timeout=30) # Give it 30 seconds to clean up
    if scraper_process.is_alive():
        app.logger.warning("Scraper process did not terminate gracefully. Forcing termination.")
        scraper_process.terminate() # Force termination if it doesn't stop
        scraper_process.join() # Wait for termination to complete

    scraper_status['status'] = 'stopped' # Final status update
    scraper_status['message'] = 'Scraper stopped by user.'
    
    # Save the current state to disk upon graceful shutdown.
    if scraper_status and scraper_status.get('output_dir'): # Ensure output_dir is available in status
        save_state_to_disk(scraper_status['output_dir'], {
            'status': dict(scraper_status), # Convert Manager.dict to dict for saving
            'products': list(scraped_products_data), # Convert Manager.list to list for saving
            'categories': list(discovered_categories_data) # Convert Manager.list to list for saving
        })
        app.logger.info(f"Scraper state saved to disk at {scraper_status['output_dir']}.")

    scraper_process = None # Clear the global process variable as it's no longer running
    app.logger.info("Scraper process stopped successfully.")
    return jsonify({'status': 'success', 'message': 'Scraper stopped.'})

@app.route('/api/summarize', methods=['POST'])
def summarize_endpoint():
    """
    Endpoint to summarize a given description using the local AI model.
    """
    data = request.json
    description = data.get('description')
    
    if not description:
        app.logger.error("Summarization request failed: description is missing.")
        return jsonify({'error': 'Description is required for summarization.'}), 400
    
    app.logger.info("Received summarization request.")
    summary = summarize_description_local(description) # Call the local summarization function

    if "Error" in summary or "disabled" in summary.lower(): # Check for error messages in the summary
        app.logger.error(f"Summarization failed: {summary}")
        return jsonify({'error': summary}), 500 # Return error if summarization failed
    else:
        app.logger.info("Description summarized successfully.")
        return jsonify({'summary': summary}) # Return the successful summary

@app.route('/api/upload_categories', methods=['POST'])
def upload_categories():
    """
    Endpoint to upload a file containing categories (JSON or CSV).
    This allows users to provide a predefined list of categories to scrape.
    """
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file selected.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected.'}), 400
    
    filename = file.filename
    file_extension = os.path.splitext(filename)[1].lower()
    
    try:
        content = file.read().decode('utf-8')
        categories_to_import = []

        if file_extension == '.json':
            data = json.loads(content)
            # Expecting a list of strings (URLs) or a list of objects with a 'url' key.
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        # If it's just a URL string, create a basic category dict.
                        categories_to_import.append({'url': item, 'name': urlparse(item).path.strip('/').split('/')[-1] or 'Unnamed Category', 'no_product_attempts': 0, 'no_products_found': False, 'status': 'pending'})
                    elif isinstance(item, dict) and 'url' in item:
                        # If it's an object, use its properties.
                        categories_to_import.append({
                            'url': item['url'], 
                            'name': item.get('name', urlparse(item['url']).path.strip('/').split('/')[-1] or 'Unnamed Category'),
                            'no_product_attempts': item.get('no_product_attempts', 0), 
                            'no_products_found': item.get('no_products_found', False),
                            'status': item.get('status', 'pending')
                        })
            else:
                raise ValueError("JSON content should be a list of URLs or objects.")
        elif file_extension == '.csv':
            csv_reader = csv.reader(io.StringIO(content))
            header = next(csv_reader, None) # Read header row to find column indices
            url_index = -1
            name_index = -1
            
            if header:
                try:
                    url_index = header.index('url')
                    name_index = header.index('name') if 'name' in header else -1
                except ValueError:
                    raise ValueError("CSV header must contain a 'url' column.")
            
            for row in csv_reader:
                if len(row) > url_index:
                    url = row[url_index].strip()
                    name = row[name_index].strip() if name_index != -1 and len(row) > name_index else urlparse(url).path.strip('/').split('/')[-1] or 'Unnamed Category'
                    categories_to_import.append({'url': url, 'name': name, 'no_product_attempts': 0, 'no_products_found': False, 'status': 'pending'})
        else:
            return jsonify({'status': 'error', 'message': 'Unsupported file format. Please upload JSON or CSV.'}), 400
        
        # Add imported categories to the shared list, avoiding duplicates.
        for cat in categories_to_import:
            if not any(d['url'] == cat['url'] for d in discovered_categories_data):
                discovered_categories_data.append(cat)
                # Also add to the category discovery queue if it's a new, pending category.
                if cat['status'] == 'pending' and not cat['no_products_found']:
                    category_discovery_queue.put(cat)

        scraper_status['discovered_categories_count'] = len(discovered_categories_data)
        app.logger.info(f"Successfully imported {len(categories_to_import)} categories from {filename}.")
        return jsonify({'status': 'success', 'message': f'{len(categories_to_import)} categories imported successfully.'})
    except Exception as e:
        app.logger.error(f"Error processing uploaded file {filename}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Error processing file: {e}'}), 500

@app.route('/api/download_data', methods=['GET'])
def download_data():
    """
    Endpoint to download scraped data in various formats (JSON, CSV).
    Users can select to download products, categories, logs, or all data.
    """
    data_type = request.args.get('type', 'products') # Default to 'products'
    file_format = request.args.get('format', 'json') # Default to 'json'

    output_data = []
    filename_prefix = "scraped_data"

    if data_type == 'products':
        output_data = list(scraped_products_data)
        filename_prefix = "scraped_products"
    elif data_type == 'categories':
        output_data = list(discovered_categories_data)
        filename_prefix = "discovered_categories"
    elif data_type == 'logs':
        output_data = list(scraper_status['log_entries'])
        filename_prefix = "scraper_logs"
    elif data_type == 'all':
        # Combine all data types into a single structured dictionary.
        output_data = {
            'products': list(scraped_products_data),
            'categories': list(discovered_categories_data),
            'logs': list(scraper_status['log_entries']),
            'status': scraper_status.copy() # Include current status snapshot
        }
        # Ensure 'processed_links_set' is a regular dict for JSON serialization.
        if 'processed_links_set' in output_data['status']:
            output_data['status']['processed_links_set'] = dict(output_data['status']['processed_links_set'])

        filename_prefix = "all_scraper_data"
    else:
        return jsonify({'status': 'error', 'message': 'Invalid data type specified.'}), 400

    timestamp_str = time.strftime('%Y%m%d_%H%M%S')
    
    if file_format == 'json':
        response_data = json.dumps(output_data, indent=4, ensure_ascii=False) # Use ensure_ascii=False for Persian characters
        mimetype = 'application/json'
        filename = f"{filename_prefix}_{timestamp_str}.json"
    elif file_format == 'csv':
        if not output_data:
            return jsonify({'status': 'error', 'message': 'No data available for CSV export.'}), 400
        
        # Prepare CSV output in a StringIO buffer.
        output = io.StringIO()
        writer = csv.writer(output)
        
        if data_type == 'products' and output_data:
            # Assume product data has consistent keys for CSV headers.
            headers = list(output_data[0].keys()) if output_data else []
            writer.writerow(headers)
            for row in output_data:
                writer.writerow([row.get(h, '') for h in headers])
        elif data_type == 'categories' and output_data:
            headers = ['url', 'name', 'no_product_attempts', 'no_products_found', 'status']
            writer.writerow(headers)
            for row in output_data:
                writer.writerow([row.get(h, '') for h in headers])
        elif data_type == 'logs' and output_data:
            headers = ['timestamp', 'level', 'message']
            writer.writerow(headers)
            for row in output_data:
                writer.writerow([row.get(h, '') for h in headers])
        elif data_type == 'all':
            # CSV export for 'all' combined data is complex; typically, users download per-type.
            # For simplicity, disallow CSV for 'all' to avoid complex nested CSV structures.
            return jsonify({'status': 'error', 'message': 'CSV export for "all data" is not supported. Please select JSON.'}), 400

        response_data = output.getvalue()
        mimetype = 'text/csv'
        filename = f"{filename_prefix}_{timestamp_str}.csv"
    else:
        return jsonify({'status': 'error', 'message': 'Invalid file format specified.'}), 400

    response = Response(response_data, mimetype=mimetype)
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


# ==============================================================================
#                 SECTION 2: CORE SCRAPING & AI LOGIC
# ==============================================================================

# --- Local LLM Caching & Summarization ---
MODEL_CACHE = None
TOKENIZER_CACHE = None

def summarize_description_local(description: str) -> str:
    """
    Summarizes a given text description using a pre-trained local AI model.
    Caches the model and tokenizer for efficiency to avoid reloading them.
    """
    global MODEL_CACHE, TOKENIZER_CACHE
    
    if not LOCAL_AI_ENABLED:
        return "Local summarization feature is disabled (Install: transformers, torch)."

    model_name = "m3hrdadfi/pegasus-persian-summarization" # A Persian summarization model
    try:
        # Load model and tokenizer only once when the function is first called.
        if MODEL_CACHE is None or TOKENIZER_CACHE is None:
            logging.info(f"Loading local AI model '{model_name}' for the first time... This may take a while.")
            TOKENIZER_CACHE = AutoTokenizer.from_pretrained(model_name)
            # Determine if CUDA (GPU) is available for faster processing, otherwise use CPU.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Using device: {device} for local AI model.")
            MODEL_CACHE = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            logging.info("Local model loaded successfully.")

        # Prepare input for the model: tokenize, convert to PyTorch tensors, and move to device.
        inputs = TOKENIZER_CACHE(description, return_tensors="pt", max_length=1024, truncation=True, padding=True).to(MODEL_CACHE.device)
        
        # Generate summary IDs using beam search for better quality.
        summary_ids = MODEL_CACHE.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
        
        # Decode the generated summary IDs back into human-readable text.
        return TOKENIZER_CACHE.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except Exception as e:
        # Log and return an informative error message if summarization fails.
        logging.error(f"Error during local summarization: {e}", exc_info=True)
        return f"Error in local summarization processing: {e}"

# --- User Agent List for Stealth ---
# A diverse list of common user agents to rotate through to mimic real browser traffic
# and reduce the chances of being blocked by anti-bot measures.
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/109.0.1518.78",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/109.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/109.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Mobile Safari/537.36",
]

def get_random_user_agent():
    """Returns a random user agent string from the predefined list."""
    return random.choice(USER_AGENTS)

# --- Selenium WebDriver Management ---
# Using threading.local() to ensure each thread gets its own WebDriver instance.
# This is crucial when using Selenium within a multi-threaded Flask app (even if processes are used,
# threads within those processes might need their own drivers).
thread_local_drivers = threading.local()
driver_lock = threading.Lock() # A lock to prevent race conditions during driver initialization/cleanup

def _initialize_selenium_driver_for_thread(config):
    """
    Initializes a Selenium WebDriver for the current thread, or returns an existing active one.
    Handles driver setup, options (headless, proxies, user-agent), and temporary user profiles.
    """
    with driver_lock: # Acquire lock to ensure only one thread initializes/checks the driver at a time
        if hasattr(thread_local_drivers, "driver"):
            try:
                # Check if the existing driver is still active by performing a simple operation.
                thread_local_drivers.driver.current_url
                return thread_local_drivers.driver
            except WebDriverException:
                # If the driver is dead (e.g., browser crashed), quit it and remove it.
                try: thread_local_drivers.driver.quit()
                except: pass # Ignore errors during quitting a potentially already dead driver
                del thread_local_drivers.driver # Remove the dead driver from thread-local storage
        
        # Initialize a new driver if none exists or the existing one was dead.
        try:
            service = ChromeService(ChromeDriverManager().install()) # Automatically installs/manages ChromeDriver
            options = webdriver.ChromeOptions()
            
            # Configure browser options based on the scraper's configuration.
            if config.get('headless', True):
                options.add_argument("--headless") # Run browser without a visible UI
            options.add_argument("--no-sandbox") # Required for some Linux environments (e.g., Docker)
            options.add_argument("--disable-dev-shm-usage") # Overcomes limited resource problems in some environments
            options.add_argument("--disable-gpu") # Recommended for headless mode
            options.add_argument("--window-size=1920,1080") # Set a consistent window size to avoid responsive layout issues
            options.add_argument("--ignore-certificate-errors") # Ignore SSL certificate errors
            options.add_argument("--allow-running-insecure-content") # Allow mixed content
            options.add_argument("--disable-blink-features=AutomationControlled") # Attempt to bypass some bot detections

            # Use a random user agent if enabled to further mimic human browsing.
            if config.get('use_random_user_agents', False):
                user_agent = get_random_user_agent()
                options.add_argument(f"user-agent={user_agent}")
                logging.info(f"Using random User-Agent: {user_agent}")

            # Add proxy configuration if specified in the config.
            proxies = config.get('proxies', [])
            if proxies:
                # For simplicity, using the first proxy if multiple are provided.
                # A more advanced implementation would include proxy rotation and health checks.
                options.add_argument(f'--proxy-server={proxies[0]}') 
                logging.info(f"Using proxy: {proxies[0]}")

            # Use a temporary user data directory to simulate a fresh browser profile for each session.
            # This helps prevent persistent cookies/cache from interfering with scraping and aids anti-detection.
            user_data_dir = os.path.join(os.getcwd(), 'chrome_profiles', f'profile_{threading.get_ident()}_{int(time.time())}')
            os.makedirs(user_data_dir, exist_ok=True)
            options.add_argument(f"--user-data-dir={user_data_dir}")
            logging.info(f"Using Chrome user data directory: {user_data_dir}")

            # Initialize the Chrome WebDriver.
            driver = webdriver.Chrome(service=service, options=options)
            thread_local_drivers.driver = driver # Store the active driver in thread-local storage
            logging.info(f"Selenium WebDriver initialized for thread {threading.get_ident()}.")
            return driver
        except Exception as e:
            # If driver initialization fails, log a critical error and signal a fatal error.
            logging.critical(f"Failed to initialize Selenium WebDriver: {e}", exc_info=True)
            scraper_fatal_error_event.set() # Signal a fatal error to the main process
            raise # Re-raise the exception to stop the current scraping task and propagate the error

def _quit_selenium_driver_for_thread():
    """
    Quits the Selenium WebDriver for the current thread and cleans up its temporary user data directory.
    Ensures resources are released properly after each scraping task.
    """
    if hasattr(thread_local_drivers, "driver"):
        try:
            thread_local_drivers.driver.quit()
            logging.info(f"Selenium WebDriver quit for thread {threading.get_ident()}.")
        except Exception as e:
            logging.error(f"Error quitting Selenium WebDriver for thread {threading.get_ident()}: {e}")
        finally:
            # Attempt to clean up the temporary user data directory.
            # This relies on the convention of how the user data directory argument is passed.
            user_data_dir = None
            if hasattr(thread_local_drivers.driver, 'service') and hasattr(thread_local_drivers.driver.service, 'service_args'):
                for arg in thread_local_drivers.driver.service.service_args:
                    if "--user-data-dir=" in arg:
                        user_data_dir = arg.split('=', 1)[1]
                        break
            
            if user_data_dir and os.path.exists(user_data_dir) and "chrome_profiles" in user_data_dir: # Basic sanity check for path
                try:
                    shutil.rmtree(user_data_dir) # Recursively remove the directory
                    logging.info(f"Cleaned up user data directory: {user_data_dir}")
                except Exception as e:
                    logging.warning(f"Failed to remove user data directory {user_data_dir}: {e}")
            
            del thread_local_drivers.driver # Remove the driver from thread-local storage


# --- Page Fetching & Parsing ---
async def fetch_page(url, use_selenium=False, config=None):
    """
    Fetches the content of a given URL, optionally using Selenium for dynamic content.
    Includes a robust retry mechanism with exponential backoff.
    """
    config = config or {} # Use a default empty dict if config is None to prevent errors
    max_retries = config.get('max_retries', 10) # Number of retries for fetching a page
    fetch_timeout = config.get('fetch_timeout', 30) # Timeout for page fetching in seconds

    # Acquire a semaphore permit before starting a browser instance or making a request.
    # This ensures the number of concurrent operations does not exceed the allowed limit.
    async with browser_semaphore:
        for attempt in range(max_retries):
            # Continuously check for stop or fatal error signals from the main process.
            if scraper_stop_event.is_set() or scraper_fatal_error_event.is_set():
                logging.info(f"Stop/Fatal error signal received. Aborting fetch for {url}.")
                return None # Abort and return None if a stop signal is received
            try:
                if use_selenium:
                    # Use Selenium WebDriver for pages requiring JavaScript rendering.
                    driver = _initialize_selenium_driver_for_thread(config)
                    driver.get(url)
                    # Wait for the page body to be present, indicating a basic page load.
                    WebDriverWait(driver, fetch_timeout).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                    return driver.page_source # Return the fully rendered HTML content
                else:
                    # Use aiohttp for faster static content fetching.
                    async with aiohttp.ClientSession(headers={'User-Agent': get_random_user_agent() if config.get('use_random_user_agents', False) else USER_AGENTS[0]}) as session:
                        async with session.get(url, timeout=fetch_timeout) as response:
                            response.raise_for_status() # Raise an exception for HTTP error status codes (4xx or 5xx)
                            return await response.text() # Return the page content as text
            except (aiohttp.ClientError, requests.exceptions.RequestException, TimeoutException, WebDriverException) as e:
                # Handle common network, timeout, or WebDriver-specific errors.
                logging.warning(f"Error fetching {url} (attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                scraper_status['error_count'] += 1 # Increment the shared error count
                await asyncio.sleep(min(60, 2 ** attempt)) # Exponential backoff to avoid hammering the server
            except Exception as e:
                # Catch any other unexpected errors during page fetch.
                logging.critical(f"Unexpected error during page fetch for {url}: {e}", exc_info=True)
                scraper_fatal_error_event.set() # Signal a fatal error to stop the scraper
                return None # Stop on unexpected critical errors
        
        # If all retries fail, log a critical error and give up on this URL.
        logging.error(f"Failed to fetch {url} after {max_retries} retries. Giving up on this URL.")
        scraper_status['critical_count'] += 1 # Increment critical error count
        return None

def parse_html_content(html_content):
    """
    Parses HTML content using BeautifulSoup for easy navigation and data extraction.
    """
    return BeautifulSoup(html_content, 'html.parser')

async def _download_and_save_image(image_url, output_dir, config):
    """
    Downloads an image from a given URL and saves it to the specified directory.
    Includes optional image processing (resizing, quality adjustment) using Pillow.
    """
    if not image_url:
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, timeout=10) as response:
                response.raise_for_status() # Check for HTTP errors
                image_bytes = await response.read() # Read image content as bytes
        
        # Generate a unique filename based on the image's hash to avoid duplicates.
        image_hash = hashlib.md5(image_bytes).hexdigest()
        original_ext = os.path.splitext(urlparse(image_url).path)[1]
        if not original_ext or len(original_ext) > 5: # Fallback if no proper extension is found
            original_ext = '.jpg' # Default to jpg

        # Apply desired output format and quality from configuration.
        output_format = config.get('image_format', 'JPEG').upper()
        output_quality = config.get('image_quality', 85) # Default quality is 85%

        file_name = f"{image_hash}.{output_format.lower()}"
        file_path = os.path.join(output_dir, file_name)

        if IMAGE_PROCESSING_ENABLED:
            try:
                # Open image from bytes using Pillow.
                img = Image.open(io.BytesIO(image_bytes))
                # Convert to RGB if not already (essential for JPEG saving).
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save the processed image with specified format and quality.
                img.save(file_path, format=output_format, quality=output_quality)
                logging.info(f"Saved processed image: {file_path}")
                return file_name # Return the filename for storage in product data
            except Exception as e:
                # If image processing fails, log a warning and save the original bytes as a fallback.
                logging.warning(f"Image processing failed for {image_url}: {e}. Saving original bytes.")
                with open(file_path, 'wb') as f:
                    f.write(image_bytes)
                return file_name
        else:
            # If image processing is disabled, just save the raw bytes directly.
            with open(file_path, 'wb') as f:
                f.write(image_bytes)
            logging.info(f"Saved raw image: {file_path}")
            return file_name

    except Exception as e:
        # Handle any errors during image download or saving.
        logging.warning(f"Failed to download or save image {image_url}: {e}")
        scraper_status['warning_count'] += 1 # Increment shared warning count
        return None


# --- Masterpiece Mode: Intelligent Data Extraction ---
def scrape_product_details_masterpiece_mode(soup: BeautifulSoup, url: str) -> dict:
    """
    Intelligently scrapes product details using a comprehensive set of heuristics
    (Masterpiece Mode). This mode attempts to find common product data points
    without explicit CSS selectors, making it adaptable to various website structures.
    """
    logging.info(f"Running Masterpiece Mode for {url}...")
    # Initialize product data with default 'Unknown' values.
    product_data = {'product_url': url, 'name': 'Unknown Name', 'price': 'Unknown Price', 'description': '', 'brand': 'Unknown Brand', 'image_url': ''}
    
    # --- Advanced Heuristics for Product Name ---
    # Prioritize specific itemprop, then common class names, then generic tags, and finally meta tags.
    name_selectors = [
        'h1[itemprop="name"]', 'h1.product-title', 'h1.product__title', 'h1',
        'h2[itemprop="name"]', 'h2.product-title', 'h2.product__title', 'h2',
        '.product-name', '.title-product', '.dkp-product-title', '.product-heading',
        '[data-product-name]', '[data-title]',
        '.c-product__title', # Specific to Digikala (Persian e-commerce)
        '.product-item-info .product-item-name', # Common e-commerce pattern (e.g., Magento)
        'meta[property="og:title"]', 'meta[name="twitter:title"]', 'meta[itemprop="name"]' # Open Graph/Twitter Card/Schema.org metadata
    ]
    for selector in name_selectors:
        if elem := soup.select_one(selector): # Use walrus operator for concise assignment and check
            if elem.name == 'meta':
                product_data['name'] = elem.get('content', '').strip()
            else:
                product_data['name'] = elem.get_text(strip=True)
            # If a meaningful name is found, break the loop.
            if product_data['name'] and product_data['name'] != 'Unknown Name' and len(product_data['name']) > 5:
                break
            
    # --- Advanced Heuristics for Product Price ---
    # Look for itemprop, common price classes, data attributes, and then use regex for robust extraction.
    price_selectors = [
        '[itemprop="price"]', '.product-price .price-value', '.price-current',
        '.c-price__value', '.js-price-value', '.price', '.product-price',
        '[data-price]', '[data-snt-price]', '[data-price-amount]',
        '.c-price__value-wrapper', # Specific to Digikala
        '.price-box .price', # Generic e-commerce
        '.money', '.amount', '.product-info-price' # More general price indicators
    ]
    # Regex to capture numerical prices, optionally with thousands separators and common currencies.
    price_regex = re.compile(r'(\d{1,3}(?:[.,]\d{3})*(?:\.\d+)?)\s*(?:تومان|ریال|ت|ری|IRT|IRR|\$|€|£)?', re.IGNORECASE)
    
    for selector in price_selectors:
        if price_elem := soup.select_one(selector):
            text = price_elem.get_text(strip=True).replace(',', '') # Remove commas for numerical parsing
            if match := price_regex.search(text):
                product_data['price'] = match.group(0) # Keep the full matched string including currency
                break
            elif re.match(r'^\d[\d.]*$', text): # If it's just a number, take it as is
                product_data['price'] = text
                break
    
    # --- Advanced Heuristics for Product Description ---
    # Prioritize itemprop, then common IDs/classes, and finally meta tags.
    description_selectors = [
        '[itemprop="description"]', '.product-description', '#description',
        '.c-params--product-details p', '.product-single__description',
        '.description-content', '.review-description', '.product-long-description',
        '.c-params__headline', # Specific to Digikala (for brief description)
        '.product.attribute.description', # Common e-commerce pattern
        'meta[property="og:description"]', 'meta[name="twitter:description"]', 'meta[itemprop="description"]' # Metadata
    ]
    for selector in description_selectors:
        if desc_elem := soup.select_one(selector):
            if desc_elem.name == 'meta':
                product_data['description'] = desc_elem.get('content', '').strip()
            else:
                product_data['description'] = desc_elem.get_text(strip=True)
            product_data['description'] = product_data['description'][:2000] # Limit description length to prevent excessive data
            if product_data['description'] and len(product_data['description']) > 20: # Ensure it's a meaningful description
                break
            
    # --- Advanced Heuristics for Product Brand ---
    # Look for itemprop, common brand classes/links, and then product metadata.
    brand_selectors = [
        '[itemprop="brand"] span', '[itemprop="brand"] a', '.product-brand a', '.brand-name',
        '.c-product__brand-link', '.brand-label', '.product-manufacturer',
        '.c-product__seller-name', # Digikala specific (sometimes seller acts as brand)
        '.product-brand-name', # Generic e-commerce
        'meta[property="product:brand"]', 'meta[name="product:brand"]' # Product metadata
    ]
    for selector in brand_selectors:
        if brand_elem := soup.select_one(selector):
            if brand_elem.name == 'meta':
                product_data['brand'] = brand_elem.get('content', '').strip()
            else:
                product_data['brand'] = brand_elem.get_text(strip=True)
            if product_data['brand'] and product_data['brand'] != 'Unknown Brand' and len(product_data['brand']) > 2:
                break

    # --- Advanced Heuristics for Product Image URL ---
    # Prioritize itemprop, then common image gallery selectors, and finally meta tags.
    image_selectors = [
        'img[itemprop="image"]', '.product-gallery img', '.main-product-image img',
        'img.c-product__img', '[data-gallery-img]', '[data-main-image]',
        '.c-gallery__img', # Digikala specific
        '.product-image-container img', # Generic e-commerce
        'meta[property="og:image"]', 'meta[name="twitter:image"]', 'link[rel="image_src"]' # Metadata
    ]
    for selector in image_selectors:
        if img_elem := soup.select_one(selector):
            # Check common attributes for image source.
            src = img_elem.get('src') or img_elem.get('data-src') or img_elem.get('data-original') or img_elem.get('content') # 'content' for meta tags
            if src:
                product_data['image_url'] = urljoin(url, src) # Resolve relative URLs
                break

    return product_data

# --- Main Scraping Logic per Product ---
async def scrape_product_details(product_url, config, output_dir, scraper_id):
    """
    Scrapes details for a single product URL based on the configured mode (Masterpiece or Manual).
    Prioritizes manual selectors if provided, falls back to Masterpiece Mode for missing fields or failures.
    """
    # Initialize product data with default values, including a placeholder for local image path.
    product_data = {
        'product_url': product_url, 
        'name': 'Unknown Name', 
        'price': 'Unknown Price', 
        'description': '', 
        'brand': 'Unknown Brand', 
        'image_url': '', 
        'local_image_path': None
    }

    # Fetch the product page content using Selenium, as product pages are often dynamic.
    html_content = await fetch_page(product_url, use_selenium=True, config=config)
    if not html_content:
        logging.warning(f"Skipping product {product_url} due to failed content fetch.")
        return None

    soup = parse_html_content(html_content)
    
    extracted_data = {}
    manual_selectors_successful = True

    # If in manual configuration mode, attempt to use provided selectors first.
    if config.get('mode') == 'manual_config' or config.get('mode') == 'single_product':
        for key, selector in config.get('product_detail_selectors', {}).items():
            if selector: # Only process if a selector is actually provided
                if (element := soup.select_one(selector)):
                    if key == 'image_url':
                        # For image URLs, check common attributes for the source.
                        src = element.get('src') or element.get('data-src') or element.get('data-original')
                        extracted_data[key] = urljoin(product_url, src) if src else ''
                    else:
                        # For other fields, extract text content.
                        extracted_data[key] = element.get_text(strip=True)
                else:
                    # If any manual selector fails, mark as unsuccessful and break to fall back to Masterpiece Mode.
                    logging.warning(f"Manual selector for {key} failed on {product_url}. Attempting fallback to Masterpiece for this field.")
                    manual_selectors_successful = False
                    break 
        
        # If manual selectors were not fully successful, or if in full_auto mode, use Masterpiece Mode.
        if not manual_selectors_successful:
            logging.info(f"Falling back to Masterpiece Mode for {product_url} due to failed manual selectors.")
            # Merge Masterpiece extracted data with any partially successful manual data.
            masterpiece_data = scrape_product_details_masterpiece_mode(soup, product_url)
            # Prioritize manual data if it exists, otherwise use masterpiece data.
            for k, v in masterpiece_data.items():
                if k not in extracted_data or not extracted_data[k] or extracted_data[k] == 'Unknown Name' or extracted_data[k] == 'Unknown Price':
                    extracted_data[k] = v
        else:
            # If manual selectors were fully successful, ensure description is limited.
            if 'description' in extracted_data:
                extracted_data['description'] = extracted_data['description'][:2000]
    else: # Default to full_auto mode if not explicitly manual or single product
        extracted_data = scrape_product_details_masterpiece_mode(soup, product_url)


    # Merge extracted data with initial product_data, overwriting defaults.
    product_data.update(extracted_data)
    
    # Download and save image if configured and an image URL was found.
    if config.get('save_images', False) and product_data.get('image_url'):
        image_filename = await _download_and_save_image(product_data['image_url'], output_dir, config)
        if image_filename:
            # Store a relative path or filename for UI display and future reference.
            product_data['local_image_path'] = image_filename 
            logging.info(f"Image for {product_data['name']} saved to {image_filename}")
        else:
            logging.warning(f"Failed to save image for {product_data['name']}")

    # Check if a meaningful product name was extracted.
    if product_data.get('name') and product_data.get('name') != 'Unknown Name' and len(product_data.get('name')) > 5:
        scraped_products_data.append(product_data) # Add to the shared list of scraped products
        scraper_status['scraped_product_count'] = len(scraped_products_data) # Update shared count
        logging.info(f"Successfully scraped: {product_data['name']} from {product_url}")
        return True # Indicate that a product was successfully found and scraped
    else:
        logging.warning(f"Could not extract meaningful product data from {product_url}. Skipping.")
        scraper_status['warning_count'] += 1 # Increment warning for unidentifiable products
        return False # Indicate that no meaningful product was found


# --- Link & Category Discovery Logic ---
async def discover_links_from_page(current_url, config, soup):
    """
    Discovers product and category links from a given BeautifulSoup object (parsed HTML).
    Puts discovered product links into `product_link_queue` and new categories into `category_discovery_queue`.
    Returns True if any product links were found on this page, False otherwise.
    """
    found_product_links_on_page = False
    
    for a_tag in soup.find_all('a', href=True): # Find all anchor tags with an href attribute
        href = a_tag['href']
        full_url = urljoin(current_url, href) # Resolve relative URLs to absolute URLs
        
        # Normalize URL to remove fragments and irrelevant query parameters for consistent unique identification.
        parsed_full_url = urlparse(full_url)
        # Keep only scheme, netloc, and path for normalization for uniqueness comparison.
        normalized_url = parsed_full_url._replace(query='', fragment='').geturl()
        
        # Apply domain filter: skip external links if configured to do so.
        if config.get('domain_filter', True) and urlparse(normalized_url).netloc != urlparse(config['target_url']).netloc:
            logging.debug(f"Skipping external link: {normalized_url}")
            continue

        # Heuristics to identify product links based on URL patterns and CSS classes.
        product_patterns = ['/product/', '/dkp-', '/p/', '/buy/', '/item/', '/shop/', '/detail/', '/product-']
        product_classes = ['product-link', 'c-product-box__img', 'c-listing__item-link', 'product-item-link', 'thumbnail', 'js-product-url', 'product-box-img', 'product-card']
        
        is_product_link = any(p in normalized_url for p in product_patterns) or \
                          re.search(r'/\d{7,}/', normalized_url) or \
                          (a_tag.get('class') and any(cls in a_tag['class'] for cls in product_classes))
        
        if is_product_link:
            # Use a Manager.dict as a set to efficiently track processed links and avoid duplicates.
            if normalized_url not in scraper_status['processed_links_set']: 
                scraper_status['processed_links_set'][normalized_url] = True # Add to shared "set"
                product_link_queue.put(normalized_url) # Add the product URL to the shared queue for scraping
                scraper_status['total_product_links_found'] += 1 # Increment shared count
                found_product_links_on_page = True
                logging.debug(f"Discovered product link: {normalized_url}")
            continue # If it's a product link, no need to check it as a category.

        # Heuristics to identify category links.
        category_patterns = ['/category/', '/categories/', '/cat/', '/group/', '/brand/', '/list/', '/collection/', '/c/'] # Added /c/
        is_category_link = any(p in normalized_url for p in category_patterns)
        
        if is_category_link:
            # Check if this category URL has already been discovered to avoid re-processing.
            existing_category_urls = {c['url'] for c in discovered_categories_data} # Convert to set for efficient lookup
            if normalized_url not in existing_category_urls:
                # Create a dictionary for the new category's information.
                category_info = {
                    'url': normalized_url, 
                    'name': a_tag.get_text(strip=True) or urlparse(normalized_url).path.strip('/').split('/')[-1] or 'Unnamed Category',
                    'no_product_attempts': 0, # Counter for attempts with no products found on this category
                    'no_products_found': False, # Flag to mark if a category consistently yields no products
                    'status': 'pending' # Initial status for a newly discovered category
                }
                discovered_categories_data.append(category_info) # Add to the shared list of discovered categories
                category_discovery_queue.put(category_info) # Add to the shared queue for category processing
                scraper_status['discovered_categories_count'] = len(discovered_categories_data) # Update shared count
                logging.info(f"Discovered new category: {normalized_url}")
    
    return found_product_links_on_page


async def process_category_page(category_info, config):
    """
    Fetches a category page, discovers product and category links within it,
    and handles pagination for that category.
    Returns True if products were found on this category page, False otherwise.
    """
    category_url = category_info['url']
    logging.info(f"Processing category page: {category_url}")
    
    # Fetch the category page using Selenium, as category pages often load dynamically.
    html_content = await fetch_page(category_url, use_selenium=True, config=config)
    if not html_content:
        logging.warning(f"Failed to fetch category page {category_url}.")
        return False

    soup = parse_html_content(html_content)
    products_found_on_page = await discover_links_from_page(category_url, config, soup)
    
    # --- Advanced Pagination Handling ---
    # This section needs significant enhancement for a truly robust scraper.
    # Current implementation relies on `discover_links_from_page` to find next page links,
    # but a dedicated pagination logic would be more effective.

    # Example of a more advanced pagination strategy (conceptual, requires implementation):
    # 1. Identify pagination type:
    #    - Next button: Look for common "Next Page" selectors (e.g., '.next-page', '#pagination a.next').
    #    - Page numbers: Extract page number links and iterate.
    #    - Infinite scroll: Requires scrolling down with Selenium and waiting for new content.
    #    - URL parameter based: Modify a 'page' or 'offset' query parameter.

    # For now, `discover_links_from_page` will add any new category/product links it finds,
    # including potential pagination links if they match category/product patterns.
    # A dedicated pagination system would look like this:
    
    # if config.get('pagination_type') == 'next_button':
    #     next_button_selector = config.get('next_button_selector') or '.pagination-next'
    #     if next_button := soup.select_one(next_button_selector):
    #         next_page_url = urljoin(category_url, next_button.get('href'))
    #         # Add next_page_url to category_discovery_queue if not already processed
    #         # and mark it as a 'pagination_page' type of category to avoid re-processing as a main category.
    # elif config.get('pagination_type') == 'url_parameter':
    #     page_param_name = config.get('page_param_name', 'page')
    #     current_page_num = int(parse_qs(urlparse(category_url).query).get(page_param_name, ['1'])[0])
    #     next_page_num = current_page_num + 1
    #     # Construct next page URL and add to queue.
    #     # This requires more sophisticated URL manipulation.

    return products_found_on_page

async def dynamic_concurrency_monitor(config):
    """
    Monitors system resources (CPU, Memory) and scraper performance (errors, products scraped)
    to dynamically adjust the target concurrency for product scraping workers.
    This helps optimize performance and prevent system overload.
    """
    global dynamic_target_concurrency
    
    base_concurrency = config.get('max_concurrent_browsers_min', 1) # Minimum allowed concurrent browsers
    max_concurrency = config.get('max_concurrent_browsers_max', 5) # Maximum allowed concurrent browsers

    # Initialize the dynamic target concurrency to the base value.
    dynamic_target_concurrency['value'] = base_concurrency
    logging.info(f"Dynamic concurrency monitor started. Initial target: {base_concurrency}")

    # Variables to track performance metrics over time.
    last_error_count = scraper_status['error_count']
    last_warning_count = scraper_status['warning_count']
    last_scraped_count = scraper_status['scraped_product_count']
    last_check_time = time.time()

    while not scraper_stop_event.is_set() and not scraper_fatal_error_event.is_set():
        await asyncio.sleep(5) # Check metrics every 5 seconds

        current_time = time.time()
        time_diff = current_time - last_check_time
        if time_diff == 0: # Avoid division by zero if time hasn't advanced
            continue
        
        # Calculate rates of errors, warnings, and products scraped per second.
        errors_per_second = (scraper_status['error_count'] - last_error_count) / time_diff
        warnings_per_second = (scraper_status['warning_count'] - last_warning_count) / time_diff
        products_per_second = (scraper_status['scraped_product_count'] - last_scraped_count) / time_diff

        logging.debug(f"Monitor: Errors/s: {errors_per_second:.2f}, Warnings/s: {warnings_per_second:.2f}, Products/s: {products_per_second:.2f}")

        # Get system CPU and memory usage if system monitoring is enabled.
        cpu_percent = 0
        memory_percent = 0
        if SYSTEM_MONITORING_ENABLED:
            try:
                cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking CPU usage
                memory_percent = psutil.virtual_memory().percent # System-wide memory usage
                logging.debug(f"System: CPU: {cpu_percent}%, Memory: {memory_percent}%")
            except Exception as e:
                logging.warning(f"Could not get system metrics: {e}. Disabling system monitoring.")
                SYSTEM_MONITORING_ENABLED = False # Disable if it keeps failing
        
        current_target = dynamic_target_concurrency['value']
        new_target = current_target

        # --- Dynamic Adjustment Logic (Heuristics-based) ---
        # This logic prioritizes stability and resource conservation over raw speed.

        # Decrease concurrency if high error/warning rates are detected, suggesting issues with the target site.
        if errors_per_second > 0.1 or warnings_per_second > 0.5:
            new_target = max(base_concurrency, current_target - 1)
            logging.warning(f"High error/warning rate detected. Decreasing concurrency to {new_target}.")
        # Decrease concurrency if system resources are heavily utilized.
        elif (SYSTEM_MONITORING_ENABLED and (cpu_percent > 85 or memory_percent > 90)):
            new_target = max(base_concurrency, current_target - 1)
            logging.warning(f"High system resource usage detected. Decreasing concurrency to {new_target}.")
        # Decrease concurrency if throughput is low but not at base, suggesting over-provisioning.
        elif products_per_second < 0.1 and current_target > base_concurrency:
            new_target = max(base_concurrency, current_target - 1)
            logging.info(f"Low product scraping rate. Decreasing concurrency to {new_target}.")
        # Increase concurrency if performance is good, errors are low, and resources are available.
        elif products_per_second > 0.5 and errors_per_second < 0.05 and warnings_per_second < 0.1 and (not SYSTEM_MONITORING_ENABLED or (cpu_percent < 70 and memory_percent < 80)):
            new_target = min(max_concurrency, current_target + 1)
            logging.info(f"Good performance. Increasing concurrency to {new_target}.")
        
        # Clamp the new target concurrency within the defined min/max bounds.
        new_target = max(config.get('max_concurrent_browsers_min', 1), min(max_concurrency, new_target))

        # Update the shared dynamic target concurrency if it has changed.
        if new_target != current_target:
            dynamic_target_concurrency['value'] = new_target
            logging.info(f"Dynamic concurrency adjusted to: {new_target}")
        
        # Update metrics for the next check interval.
        last_error_count = scraper_status['error_count']
        last_warning_count = scraper_status['warning_count']
        last_scraped_count = scraper_status['scraped_product_count']
        last_check_time = current_time


async def save_state_to_disk(output_dir, state_data):
    """
    Saves the current scraper's operational state (status, products, categories) to JSON files on disk.
    This allows for resuming interrupted scraping sessions.
    """
    try:
        os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists
        # Convert Manager.dict and Manager.list objects to standard Python dicts/lists for JSON serialization.
        status_copy = dict(state_data['status'])
        if 'processed_links_set' in status_copy:
            status_copy['processed_links_set'] = dict(status_copy['processed_links_set'])
        
        # Save scraper status.
        with open(os.path.join(output_dir, 'scraper_status.json'), 'w', encoding='utf-8') as f:
            json.dump(status_copy, f, indent=4, ensure_ascii=False) # Pretty print JSON, handle non-ASCII chars
        
        # Save scraped products data.
        with open(os.path.join(output_dir, 'scraped_products.json'), 'w', encoding='utf-8') as f:
            json.dump(state_data['products'], f, indent=4, ensure_ascii=False)
        
        # Save discovered categories data.
        with open(os.path.join(output_dir, 'discovered_categories.json'), 'w', encoding='utf-8') as f:
            json.dump(state_data['categories'], f, indent=4, ensure_ascii=False)
        
        logging.info(f"Scraper state saved to {output_dir}.")
    except Exception as e:
        logging.error(f"Error saving scraper state to disk: {e}", exc_info=True)


def load_state_from_disk(output_dir, manager_instance):
    """
    Loads a previous scraper's operational state from JSON files on disk.
    Re-initializes Manager objects with loaded data.
    """
    loaded_data = {'status': {}, 'products': [], 'categories': []}
    try:
        # Load scraper status.
        status_path = os.path.join(output_dir, 'scraper_status.json')
        if os.path.exists(status_path):
            with open(status_path, 'r', encoding='utf-8') as f:
                loaded_status = json.load(f)
                loaded_data['status'] = loaded_status
                # Convert loaded 'processed_links_set' back to Manager.dict for shared use.
                if 'processed_links_set' in loaded_data['status']:
                    loaded_data['status']['processed_links_set'] = manager_instance.dict(loaded_data['status']['processed_links_set'])
                else:
                    loaded_data['status']['processed_links_set'] = manager_instance.dict() # Ensure it's always a Manager.dict

                # Restore log_entries as Manager.list if it was part of the status.
                if 'log_entries' in loaded_data['status']:
                    loaded_data['status']['log_entries'] = manager_instance.list(loaded_data['status']['log_entries'])
                else:
                    loaded_data['status']['log_entries'] = manager_instance.list()

        # Load scraped products data.
        products_path = os.path.join(output_dir, 'scraped_products.json')
        if os.path.exists(products_path):
            with open(products_path, 'r', encoding='utf-8') as f:
                loaded_data['products'] = json.load(f)
        
        # Load discovered categories data.
        categories_path = os.path.join(output_dir, 'discovered_categories.json')
        if os.path.exists(categories_path):
            with open(categories_path, 'r', encoding='utf-8') as f:
                loaded_data['categories'] = json.load(f)
        
        logging.info(f"Successfully loaded state from {output_dir}.")
        return loaded_data
    except Exception as e:
        # Log a warning if state loading fails, indicating a fresh start.
        logging.warning(f"Could not load previous scraper state from disk: {e}. Starting fresh.", exc_info=True)
        return None


async def _scraper_main_logic(config, output_dir, scraper_id):
    """
    The main asynchronous orchestrator for the scraping process.
    Manages category discovery, product scraping, concurrency, and graceful shutdown.
    """
    target_url = config.get('target_url')
    max_products_to_scrape = config.get('max_products_to_scrape') # Can be None for no limit
    max_empty_category_attempts = config.get('max_empty_category_attempts', 10) # Max retries for categories with no products

    # Ensure output_dir is stored in shared status for the periodic save function.
    scraper_status['output_dir'] = output_dir

    # If in single product mode, directly add the target URL to the product queue.
    if config.get('mode') == 'single_product':
        if target_url not in scraper_status['processed_links_set']:
            product_link_queue.put(target_url)
            scraper_status['total_product_links_found'] += 1
            scraper_status['processed_links_set'][target_url] = True
        logging.info(f"Single product mode: {target_url} added to product queue.")
    else:
        # For full-auto or manual config modes, add the initial target URL as a category to start discovery.
        initial_category_info = {
            'url': target_url, 
            'name': 'Initial Target', 
            'no_product_attempts': 0, 
            'no_products_found': False,
            'status': 'pending' # Initial status for a new category
        }
        # Only add to discovered_categories_data if not already present (e.g., from a loaded state or upload).
        if not any(c['url'] == initial_category_info['url'] for c in discovered_categories_data):
            discovered_categories_data.append(initial_category_info)
            category_discovery_queue.put(initial_category_info)
            scraper_status['discovered_categories_count'] = len(discovered_categories_data)
            logging.info(f"Initial target {target_url} added to category discovery queue.")
        else:
            # If loaded from state, ensure any pending categories (including the initial one) are re-queued.
            for cat in discovered_categories_data:
                if cat['status'] == 'pending' and not cat['no_products_found'] and cat['url'] == target_url:
                    category_discovery_queue.put(cat)
                    logging.info(f"Re-queued initial target {target_url} from loaded state.")


    # Set to keep track of active category processing tasks.
    active_category_tasks = set()
    
    # Start product scraping worker tasks.
    product_scraper_workers = []
    # Create workers up to the maximum allowed by the semaphore, ready to consume from the product queue.
    num_workers_to_start = config.get('max_concurrent_browsers_max', 5)
    for i in range(num_workers_to_start): 
        worker_task = asyncio.create_task(product_scraper_worker(config, output_dir, scraper_id, i))
        product_scraper_workers.append(worker_task)

    # Start the dynamic concurrency monitor if enabled in full-auto mode.
    concurrency_monitor_task = None
    if config.get('auto_adjust_concurrency') and config.get('auto_adjust_mode') == 'full_auto':
        concurrency_monitor_task = asyncio.create_task(dynamic_concurrency_monitor(config))
        logging.info("Dynamic concurrency monitor task started.")
    
    # Start the periodic state saving task to ensure progress is regularly saved.
    state_saving_task = asyncio.create_task(
        periodic_save_state(output_dir, scraper_status, scraped_products_data, discovered_categories_data, save_interval=60)
    )

    # --- Main Orchestration Loop ---
    # This loop continuously manages category discovery and product scraping tasks.
    while True:
        # Check for explicit stop signals or fatal errors.
        if scraper_stop_event.is_set():
            logging.info("Scraper stop event detected. Terminating main loop.")
            scraper_status['status'] = 'stopped'
            scraper_status['message'] = 'Scraping stopped by user.'
            break
        if scraper_fatal_error_event.is_set():
            logging.critical("Fatal error event detected. Terminating main loop.")
            scraper_status['status'] = 'fatal_error'
            scraper_status['message'] = 'A fatal error occurred in the scraper.'
            break
        
        # Check if the maximum products limit has been reached (if set).
        if max_products_to_scrape is not None and scraper_status['scraped_product_count'] >= max_products_to_scrape:
            logging.info(f"Max products limit ({max_products_to_scrape}) reached. Terminating main loop.")
            scraper_status['status'] = 'completed'
            scraper_status['message'] = f'{max_products_to_scrape} products scraped. Finishing.'
            break

        # Process new categories from the queue.
        while not category_discovery_queue.empty():
            category_info = category_discovery_queue.get_nowait()
            
            # Find the actual category object in `discovered_categories_data` to update its status.
            current_category_obj = None
            for i, cat_obj in enumerate(discovered_categories_data):
                if cat_obj['url'] == category_info['url']:
                    current_category_obj = cat_obj
                    break
            
            # Skip categories that have been marked as consistently yielding no products.
            if current_category_obj and current_category_obj['no_products_found']:
                logging.info(f"Skipping category {category_info['url']} as it was marked as no products found.")
                continue

            # Update the status of the category in the shared list to 'processing'.
            if current_category_obj:
                idx = discovered_categories_data.index(current_category_obj)
                discovered_categories_data[idx] = {**current_category_obj, 'status': 'processing'}
            
            # Create an asyncio task to process the category page.
            task = asyncio.create_task(process_category_page(category_info, config))
            task._category_url = category_info['url'] # Attach URL for easy retrieval on completion
            active_category_tasks.add(task) # Add to the set of active tasks
            logging.info(f"Started processing category: {category_info['url']}")

        # Process completed category tasks.
        done_category_tasks = {task for task in active_category_tasks if task.done()}
        for task in done_category_tasks:
            active_category_tasks.remove(task)
            category_url = task._category_url # Retrieve the category URL from the task
            products_found_on_page = task.result() # Get the result (True/False) from the task

            # Update the status of the processed category in the shared list.
            for i, cat_obj in enumerate(discovered_categories_data):
                if cat_obj['url'] == category_url:
                    if products_found_on_page:
                        # Reset attempts if products were found.
                        discovered_categories_data[i] = {**cat_obj, 'no_product_attempts': 0, 'no_products_found': False, 'status': 'processed'}
                        logging.info(f"Category {category_url} processed, products found.")
                    else:
                        # Increment no-product attempts; if threshold reached, mark as 'no_products_found'.
                        cat_obj['no_product_attempts'] += 1
                        if cat_obj['no_product_attempts'] >= max_empty_category_attempts:
                            discovered_categories_data[i] = {**cat_obj, 'no_products_found': True, 'status': 'no_products_found'}
                            logging.warning(f"Category {category_url} marked as NO PRODUCTS FOUND after {cat_obj['no_product_attempts']} attempts.")
                        else:
                            # Re-add to queue for retry if not exhausted.
                            discovered_categories_data[i] = {**cat_obj, 'status': 'retrying'}
                            category_discovery_queue.put(cat_obj) # Re-add for another attempt
                            logging.info(f"Category {category_url} processed, no products found. Attempt {cat_obj['no_product_attempts']}/{max_empty_category_attempts}. Re-adding to queue.")
                    break

        # Check for natural completion: all queues empty and no active tasks.
        # This condition is typically met when there are no more links to discover or products to scrape.
        if config.get('mode') != 'single_product' and \
           category_discovery_queue.empty() and not active_category_tasks and \
           product_link_queue.empty() and all(w.done() for w in product_scraper_workers):
            # Give a small grace period to ensure no late discoveries are missed.
            await asyncio.sleep(2) 
            if category_discovery_queue.empty() and not active_category_tasks and \
               product_link_queue.empty() and all(w.done() for w in product_scraper_workers):
                logging.info("All category discovery and product scraping tasks appear to be complete (natural end).")
                scraper_status['status'] = 'completed'
                scraper_status['message'] = 'Scraping completed successfully.'
                break
        
        # Prevent busy waiting by yielding control to the event loop.
        await asyncio.sleep(0.5) 

    # Signal product workers to stop by putting None (sentinel) into their queue.
    for _ in product_scraper_workers:
        product_link_queue.put(None) 
    await asyncio.gather(*product_scraper_workers) # Wait for all workers to finish processing sentinels

    # Cancel and await completion of background monitoring and saving tasks.
    if concurrency_monitor_task:
        concurrency_monitor_task.cancel()
        try:
            await concurrency_monitor_task 
        except asyncio.CancelledError:
            logging.info("Dynamic concurrency monitor task cancelled.")
            
    if state_saving_task:
        state_saving_task.cancel()
        try:
            await state_saving_task 
        except asyncio.CancelledError:
            logging.info("Periodic state saving task cancelled.")

    logging.info("All scraping tasks completed or aborted.")
    # Perform a final save of the state, regardless of how the scraper stopped.
    save_state_to_disk(output_dir, {
        'status': dict(scraper_status),
        'products': list(scraped_products_data),
        'categories': list(discovered_categories_data)
    })
    logging.info(f"Final scraper state saved to disk at {output_dir}.")

async def periodic_save_state(output_dir, status_shared, products_shared, categories_shared, save_interval=60):
    """
    Periodically saves the scraper's state to disk at a specified interval.
    This ensures data persistence even if the scraper crashes unexpectedly.
    """
    while not scraper_stop_event.is_set() and not scraper_fatal_error_event.is_set():
        try:
            await asyncio.sleep(save_interval) # Wait for the specified interval
            await save_state_to_disk(output_dir, {
                'status': status_shared,
                'products': products_shared,
                'categories': categories_shared
            })
        except asyncio.CancelledError:
            logging.info("Periodic state saving task cancelled.")
            break
        except Exception as e:
            logging.error(f"Error in periodic state saving: {e}", exc_info=True)


async def product_scraper_worker(config, output_dir, scraper_id, worker_id):
    """
    Worker function executed by each concurrent scraping process.
    It fetches product URLs from the shared queue and scrapes their details.
    """
    global dynamic_target_concurrency # Access the shared dynamic concurrency value
    
    while not scraper_stop_event.is_set() and not scraper_fatal_error_event.is_set():
        try:
            # Check if this worker should be active based on the dynamically adjusted concurrency target.
            if config.get('auto_adjust_concurrency') and config.get('auto_adjust_mode') == 'full_auto':
                current_active_target = dynamic_target_concurrency.get('value', config.get('max_concurrent_browsers_min', 1))
                if worker_id >= current_active_target:
                    await asyncio.sleep(1) # This worker is "paused" if its ID is above the current target
                    continue

            # Check max products limit before attempting to get from queue.
            max_products_to_scrape = config.get('max_products_to_scrape')
            if max_products_to_scrape is not None and \
               scraper_status['scraped_product_count'] >= max_products_to_scrape:
                logging.info(f"Worker {worker_id}: Max products limit ({max_products_to_scrape}) reached. Exiting.")
                product_link_queue.put(None) # Put sentinel back for other workers to stop gracefully
                break # Exit worker loop

            # Get a product URL from the queue with a timeout.
            # A timeout prevents workers from blocking indefinitely if the queue becomes empty.
            product_url = await asyncio.wait_for(product_link_queue.get(), timeout=10) # Increased timeout for robustness
            if product_url is None: # Check for the sentinel value to gracefully stop the worker
                product_link_queue.task_done() # Mark the sentinel as done
                break
            
            logging.info(f"Worker {worker_id} scraping product: {product_url}")
            await scrape_product_details(product_url, config, output_dir, scraper_id)
            product_link_queue.task_done() # Mark the task as done in the queue

        except asyncio.TimeoutError:
            # If the queue is empty for the timeout duration, continue checking.
            logging.debug(f"Worker {worker_id}: Product queue empty for a while, waiting...")
            continue
        except Exception as e:
            # Handle any unexpected errors within the worker.
            logging.error(f"Error in product scraper worker {worker_id}: {e}", exc_info=True)
            scraper_status['critical_count'] += 1 # Increment critical error count
            scraper_fatal_error_event.set() # Signal a fatal error to the main process
            break # Stop worker on critical error


def run_scraper_process(config, output_dir, scraper_id, status_shared_ext, products_shared_ext, categories_shared_ext, stop_event_ext, fatal_error_event_ext, browser_semaphore_ext, product_link_queue_ext, category_discovery_queue_ext, dynamic_target_concurrency_ext):
    """
    The target function for the new multiprocessing.Process.
    This function is executed in a separate process and sets up the environment
    (logging, global shared objects) for the asynchronous scraper logic.
    """
    # Assign shared memory objects (passed as arguments) to global variables within this new child process.
    # This allows the child process to access and modify the shared state.
    global scraper_status, scraped_products_data, discovered_categories_data, scraper_stop_event, scraper_fatal_error_event, browser_semaphore, product_link_queue, category_discovery_queue, dynamic_target_concurrency
    scraper_status = status_shared_ext
    scraped_products_data = products_shared_ext
    discovered_categories_data = categories_shared_ext
    scraper_stop_event = stop_event_ext
    scraper_fatal_error_event = fatal_error_event_ext
    browser_semaphore = browser_semaphore_ext
    product_link_queue = product_link_queue_ext
    category_discovery_queue = category_discovery_queue_ext
    dynamic_target_concurrency = dynamic_target_concurrency_ext

    # Configure logging specifically for this child process.
    # This ensures logs from the scraper process are distinguishable and properly handled.
    # IMPORTANT: Clear existing handlers to prevent issues with inherited handlers
    # that might not be pickleable or could cause duplicate log entries.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()

    logging.basicConfig(
        level=getattr(logging, config.get('log_level', 'INFO').upper()), # Use log level from config (e.g., INFO, DEBUG)
        format=f'%(asctime)s - %(levelname)s - [Scraper Process {os.getpid()}] - %(message)s', # Format includes process ID
        handlers=[
            logging.StreamHandler(sys.stdout), # Log to console (stdout)
        ]
    )
    # Add the custom shared list handler separately to send logs to the main process's shared list.
    logging.root.addHandler(SharedListHandler(scraper_status['log_entries']))


    logging.info(f"Scraper process {os.getpid()} started with config: {config.get('mode')}.")
    
    try:
        # Run the main asynchronous scraping logic within this process.
        asyncio.run(_scraper_main_logic(config, output_dir, scraper_id))
        
        # Update final status based on how the scraper process completed.
        if not scraper_stop_event.is_set() and not scraper_fatal_error_event.is_set():
            scraper_status['status'] = 'completed'
            scraper_status['message'] = 'Scraping completed successfully.'
            logging.info("Scraping completed successfully.")
        elif scraper_stop_event.is_set():
            scraper_status['status'] = 'stopped'
            scraper_status['message'] = 'Scraping stopped by user.'
            logging.info("Scraping stopped by user.")
        else: # Fatal error occurred
            logging.critical("Scraper process terminated due to a fatal error.")
    
    except Exception as e:
        # Catch any unexpected exceptions that might occur during the scraping process.
        logging.critical(f"A fatal error occurred in scraper process {os.getpid()}: {e}", exc_info=True)
        scraper_status['status'] = 'fatal_error'
        scraper_status['message'] = f"Fatal Error: {e}"
        scraper_fatal_error_event.set() # Signal fatal error to the main process
    finally:
        # Ensure the Selenium WebDriver is quit when the process finishes, regardless of success or failure.
        _quit_selenium_driver_for_thread()
        logging.info(f"Scraper process {os.getpid()} finished.")


# --- Custom Log Handler for Shared List ---
class SharedListHandler(logging.Handler):
    """
    A custom logging handler that appends formatted log records to a shared
    multiprocessing.Manager.list(). This allows logs from child processes
    to be visible in the main Flask process's UI dashboard.
    """
    def __init__(self, shared_list):
        super().__init__()
        self.shared_list = shared_list
        # Set a formatter specifically for this handler.
        self.setFormatter(logging.Formatter('%(levelname)s - %(message)s')) 

    def emit(self, record):
        """
        Emits a log record by formatting it and appending it to the shared list.
        Limits the size of the log list to prevent excessive memory consumption.
        """
        try:
            log_entry = {
                'timestamp': time.time(),
                'level': record.levelname,
                'message': self.format(record) # Format the record using the handler's formatter
            }
            # Limit the number of logs to prevent excessive memory usage in the shared list.
            if len(self.shared_list) >= 500: # Keep only the last 500 log entries
                del self.shared_list[0] # Remove the oldest log entry
            self.shared_list.append(log_entry) # Add the new log entry
        except Exception:
            self.handleError(record) # Call default error handler if emission fails


# ==============================================================================
#                 SECTION 4: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    # This setup is crucial for multiprocessing, especially on Windows.
    # It ensures that when a new process is spawned, it correctly imports
    # the main module without re-executing the top-level code.
    multiprocessing.freeze_support()

    # --- Initialize Shared State Objects here, ONLY in the main process ---
    # These objects allow communication and data sharing between the main Flask process
    # and the child scraper process. They must be created by a Manager.
    manager = multiprocessing.Manager() # Initialize the multiprocessing Manager

    # Shared dictionary for scraper status and metrics.
    scraper_status = manager.dict({
        'status': 'idle', 'message': 'Scraper is ready.', 'scraped_product_count': 0,
        'total_product_links_found': 0, 'discovered_categories_count': 0, 'error_count': 0,
        'warning_count': 0, 'critical_count': 0, 'last_updated_at': time.time(),
        'log_entries': manager.list(), # Shared list to store logs from all processes
        'processed_links_set': manager.dict() # Shared dict used as a set to track processed URLs
    })
    # Shared list to store scraped product data.
    scraped_products_data = manager.list() 
    # Shared list to store discovered categories. Each category is a dict.
    discovered_categories_data = manager.list() 
    # Events for signaling between processes.
    scraper_stop_event = multiprocessing.Event() # Signals the scraper to stop
    scraper_fatal_error_event = multiprocessing.Event() # Signals a fatal error occurred

    # Shared queues for inter-process communication.
    product_link_queue = manager.Queue() # Queue for product URLs to be scraped
    category_discovery_queue = manager.Queue() # Queue for category URLs to be discovered
    # Semaphore to limit concurrent browser instances.
    browser_semaphore = manager.Semaphore(1) # Default initial value, will be updated by config
    # Shared dictionary for dynamic concurrency adjustment.
    dynamic_target_concurrency = manager.dict({'value': 1}) # Default initial value

    # Configure logging for the main Flask process.
    # This logger will output to the console (stdout).
    logging.basicConfig(
        level=logging.INFO, # Default log level for the main process
        format='%(asctime)s - %(levelname)s - [Flask Main] - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    logging.info("Starting Flask server on http://127.0.0.1:5000")
    # use_reloader=False is crucial for stability when using multiprocessing in debug mode.
    # The reloader can cause child processes to be spawned multiple times, leading to issues.
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)

