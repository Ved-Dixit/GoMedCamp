import os
import psycopg2
import psycopg2.extras # Added for DictCursor
from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from dotenv import load_dotenv
import json # Standard library for json.loads
import pandas as pd
import geopandas as gpd
import logging # For configuring logging
import numpy as np # Import numpy for type checking if needed, or just cast
from math import radians, sin, cos, sqrt, atan2 # For Haversine distance (optional future use)
from datetime import datetime # Added for timestamping
import zipfile 
import requests # <--- ADDED for downloading files from URLs
import io       # <--- ADDED for io.BytesIO

# --- Import for local Hugging Face models (chatbot and translation) ---
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configuration
app.config['BCRYPT_LOG_ROUNDS'] = 12 # Configuration for the Bcrypt extension

# Initialize extensions
bcrypt = Bcrypt(app)
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Enable CORS for all /api routes

# Database connection details from environment variables
DB_NAME=os.getenv("DB_NAME", ) # Default to railway, but allow override
DB_USER=os.getenv("DB_USER", )
DB_PASSWORD=os.getenv("DB_PASSWORD", )
DB_HOST=os.getenv("DB_HOST", )
DB_PORT=os.getenv("DB_PORT", )

# --- Configuration for Heatmap Data ---
# These are now URLs by default as per your app.py
BASE_JSON_DIR = os.getenv('APP_BASE_JSON_DIR', )
CSV_POINTS_PATH = os.getenv('APP_CSV_POINTS_PATH', )
# --- ADDED CSV Column Environment Variables ---
ENV_CSV_STATE_COL = os.getenv('APP_CSV_STATE_COL')
ENV_CSV_DISTRICT_COL = os.getenv('APP_CSV_DISTRICT_COL')
ENV_CSV_LAT_COL = os.getenv('APP_CSV_LAT_COL')
ENV_CSV_LON_COL = os.getenv('APP_CSV_LON_COL')


# --- Configuration for AI Chatbot and Translation ---
HF_API_TOKEN = os.getenv('HF_API_TOKEN') 
HF_CHATBOT_MODEL_ID = os.getenv('HF_CHATBOT_MODEL_ID', "gpt2") 
HF_TRANSLATION_MODEL_ID = os.getenv('HF_TRANSLATION_MODEL_ID', "facebook/nllb-200-distilled-600M")

# --- Global variables for local CHATBOT model ---
local_chatbot_pipeline = None
local_chatbot_tokenizer = None 
LOCAL_CHATBOT_MODEL_INIT_STATUS = "pending"

# --- Global variables for local TRANSLATION model ---
local_translation_pipeline = None
LOCAL_TRANSLATION_MODEL_INIT_STATUS = "pending"

# --- NLLB Language Code Mapping ---
LANGUAGE_CODE_MAP_NLLB = {
    "en": "eng_Latn", "hi": "hin_Deva", "es": "spa_Latn", "fr": "fra_Latn",
    "de": "deu_Latn", "ar": "ara_Arab", "bn": "ben_Beng", "gu": "guj_Gujr",
    "kn": "kan_Knda", "ml": "mal_Mlym", "mr": "mar_Deva", "pa": "pan_Guru",
    "ta": "tam_Taml", "te": "tel_Telu", "ur": "urd_Arab",
}

INTERNAL_BOT_ERROR_MESSAGES = {
    "Chatbot is currently unavailable (local model issue).",
    "The input message is too long for the chatbot to process.",
    "Chatbot received an unexpected response from the local model.",
    "An error occurred while communicating with the local chatbot model."
}

def initialize_local_chatbot_model():
    global local_chatbot_pipeline, local_chatbot_tokenizer, LOCAL_CHATBOT_MODEL_INIT_STATUS
    if LOCAL_CHATBOT_MODEL_INIT_STATUS != "pending": return
    if not HF_CHATBOT_MODEL_ID:
        app.logger.error("HF_CHATBOT_MODEL_ID not configured. Cannot initialize local chatbot.")
        LOCAL_CHATBOT_MODEL_INIT_STATUS = "failed"; return
    try:
        app.logger.info(f"Attempting to initialize local CHATBOT pipeline for model: {HF_CHATBOT_MODEL_ID}...")
        local_chatbot_pipeline = pipeline('text-generation', model=HF_CHATBOT_MODEL_ID)
        local_chatbot_tokenizer = AutoTokenizer.from_pretrained(HF_CHATBOT_MODEL_ID)
        if local_chatbot_pipeline.tokenizer.pad_token_id is None:
            local_chatbot_pipeline.tokenizer.pad_token_id = local_chatbot_pipeline.tokenizer.eos_token_id
        if local_chatbot_tokenizer.pad_token_id is None:
            local_chatbot_tokenizer.pad_token_id = local_chatbot_tokenizer.eos_token_id
        LOCAL_CHATBOT_MODEL_INIT_STATUS = "success"
        app.logger.info(f"Local CHATBOT pipeline for {HF_CHATBOT_MODEL_ID} initialized successfully.")
    except Exception as e:
        app.logger.error(f"Failed to initialize local CHATBOT pipeline for {HF_CHATBOT_MODEL_ID}: {e}", exc_info=True)
        LOCAL_CHATBOT_MODEL_INIT_STATUS = "failed"

def initialize_local_translation_model():
    global local_translation_pipeline, LOCAL_TRANSLATION_MODEL_INIT_STATUS
    if LOCAL_TRANSLATION_MODEL_INIT_STATUS != "pending": return
    if not HF_TRANSLATION_MODEL_ID:
        app.logger.error("HF_TRANSLATION_MODEL_ID not configured. Cannot initialize local translation model.")
        LOCAL_TRANSLATION_MODEL_INIT_STATUS = "failed"; return
    try:
        app.logger.info(f"Attempting to initialize local TRANSLATION pipeline for model: {HF_TRANSLATION_MODEL_ID}...")
        local_translation_pipeline = pipeline("translation", model=HF_TRANSLATION_MODEL_ID)
        LOCAL_TRANSLATION_MODEL_INIT_STATUS = "success"
        app.logger.info(f"Local TRANSLATION pipeline for {HF_TRANSLATION_MODEL_ID} initialized successfully.")
    except Exception as e:
        app.logger.error(f"Failed to initialize local TRANSLATION pipeline for {HF_TRANSLATION_MODEL_ID}: {e}", exc_info=True)
        LOCAL_TRANSLATION_MODEL_INIT_STATUS = "failed"

def check_db_env_vars():
    required_db_vars = {'DB_NAME': DB_NAME, 'DB_USER': DB_USER, 'DB_HOST': DB_HOST, 'DB_PORT': DB_PORT}
    missing_vars = [var_name for var_name, var_value in required_db_vars.items() if not var_value]
    if missing_vars:
        app.logger.critical(f"Missing critical database environment variables: {', '.join(missing_vars)}. Database operations will likely fail.")
        return False
    app.logger.info("All critical database environment variables appear to be set.")
    return True

def get_db_connection():
    if not all([DB_NAME, DB_USER, DB_HOST, DB_PORT]):
        app.logger.error("Cannot attempt database connection due to missing DB configuration variables.")
        return None
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        return conn
    except psycopg2.Error as e:
        app.logger.error(f"Error connecting to PostgreSQL database: {e}", exc_info=True)
        return None
    except Exception as e: 
        app.logger.error(f"Unexpected error connecting to PostgreSQL database: {e}", exc_info=True)
        return None

def create_tables():
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                # Users Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY, username VARCHAR(80) UNIQUE NOT NULL, email VARCHAR(120) UNIQUE NOT NULL,
                        phone_number VARCHAR(20) UNIQUE NOT NULL, password_hash VARCHAR(128) NOT NULL,
                        user_type VARCHAR(50) NOT NULL, address TEXT, latitude DECIMAL(10, 8), longitude DECIMAL(11, 8),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);""")
                # Camps Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS camps (
                        id SERIAL PRIMARY KEY, name VARCHAR(255) NOT NULL, location_latitude DECIMAL(10, 8),
                        location_longitude DECIMAL(11, 8), location_address TEXT, start_date DATE NOT NULL, end_date DATE NOT NULL,
                        organizer_id INTEGER REFERENCES users(id) ON DELETE SET NULL, description TEXT,
                        status VARCHAR(50) DEFAULT 'planned', target_patients INTEGER DEFAULT 0,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);""")
                # Patients Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS patients (
                        id SERIAL PRIMARY KEY, camp_id INTEGER REFERENCES camps(id) ON DELETE CASCADE, 
                        user_id INTEGER REFERENCES users(id) ON DELETE SET NULL, name VARCHAR(150) NOT NULL,
                        email VARCHAR(150) NOT NULL, phone_number VARCHAR(20), disease_detected TEXT, area_location VARCHAR(255),
                        organizer_notes TEXT, created_by_organizer_id INTEGER REFERENCES users(id) ON DELETE SET NULL, 
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);""")
                try: cur.execute("ALTER TABLE patients ALTER COLUMN camp_id DROP NOT NULL;")
                except psycopg2.Error as _: pass 
                try: cur.execute("ALTER TABLE patients ALTER COLUMN created_by_organizer_id DROP NOT NULL;")
                except psycopg2.Error as _: pass 
                cur.execute("CREATE INDEX IF NOT EXISTS idx_patients_email ON patients (email);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_patients_user_id ON patients (user_id);")
                # Other tables
                cur.execute("CREATE TABLE IF NOT EXISTS camp_registrations (id SERIAL PRIMARY KEY, camp_id INTEGER REFERENCES camps(id) ON DELETE CASCADE NOT NULL, user_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL, registration_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, status VARCHAR(50) DEFAULT 'pending', notes TEXT, UNIQUE (camp_id, user_id));")
                cur.execute("CREATE TABLE IF NOT EXISTS connection_requests (id SERIAL PRIMARY KEY, camp_id INTEGER REFERENCES camps(id) ON DELETE CASCADE NOT NULL, organizer_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL, local_org_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL, status VARCHAR(50) DEFAULT 'pending', requested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, responded_at TIMESTAMP WITH TIME ZONE, UNIQUE (camp_id, organizer_id, local_org_id));")
                cur.execute("CREATE TABLE IF NOT EXISTS chat_messages (id SERIAL PRIMARY KEY, connection_request_id INTEGER REFERENCES connection_requests(id) ON DELETE CASCADE NOT NULL, sender_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL, message_text TEXT NOT NULL, sent_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, read_at TIMESTAMP WITH TIME ZONE);")
                cur.execute("CREATE TABLE IF NOT EXISTS camp_staff (id SERIAL PRIMARY KEY, camp_id INTEGER REFERENCES camps(id) ON DELETE CASCADE NOT NULL, name VARCHAR(255) NOT NULL, role VARCHAR(255), origin TEXT, contact VARCHAR(100), notes TEXT);")
                cur.execute("CREATE TABLE IF NOT EXISTS camp_medicines (id SERIAL PRIMARY KEY, camp_id INTEGER REFERENCES camps(id) ON DELETE CASCADE NOT NULL, name VARCHAR(255) NOT NULL, unit VARCHAR(50), quantity_per_patient DECIMAL(10,2), notes TEXT);")
                cur.execute("CREATE TABLE IF NOT EXISTS camp_equipment (id SERIAL PRIMARY KEY, camp_id INTEGER REFERENCES camps(id) ON DELETE CASCADE NOT NULL, name VARCHAR(255) NOT NULL, quantity INTEGER, notes TEXT);")
                cur.execute("CREATE TABLE IF NOT EXISTS patient_feedback (id SERIAL PRIMARY KEY, patient_user_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL, patient_record_id INTEGER REFERENCES patients(id) ON DELETE SET NULL, feedback_text TEXT NOT NULL, rating INTEGER CHECK (rating >= 1 AND rating <= 5), language VARCHAR(10), created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);")
                cur.execute("CREATE TABLE IF NOT EXISTS patient_chat_messages (id SERIAL PRIMARY KEY, patient_user_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL, patient_record_id INTEGER REFERENCES patients(id) ON DELETE SET NULL, message_text TEXT NOT NULL, sender_type VARCHAR(10) NOT NULL, language VARCHAR(10), timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);")
                cur.execute("CREATE TABLE IF NOT EXISTS camp_reviews (id SERIAL PRIMARY KEY, camp_id INTEGER REFERENCES camps(id) ON DELETE CASCADE NOT NULL, patient_user_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL, rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5), comment TEXT, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);")
                cur.execute("CREATE TABLE IF NOT EXISTS camp_follow_ups (id SERIAL PRIMARY KEY, camp_id INTEGER REFERENCES camps(id) ON DELETE CASCADE NOT NULL, patient_identifier TEXT NOT NULL, notes TEXT, added_by_organizer_id INTEGER REFERENCES users(id) ON DELETE SET NULL, linked_patient_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);")
                conn.commit()
                app.logger.info("All tables checked/created and alterations attempted successfully.")
                return True 
        except psycopg2.Error as e:
            app.logger.error(f"Error during table creation/alteration: {e}", exc_info=True) 
            if conn: conn.rollback()
            return False 
        except Exception as e: 
            app.logger.error(f"Unexpected error during table creation/alteration: {e}", exc_info=True)
            if conn: conn.rollback()
            return False 
        finally:
            if conn and not conn.closed: 
                try: conn.close()
                except Exception as e_close: app.logger.error(f"[create_tables] Error closing connection: {e_close}", exc_info=True)
    else:
        app.logger.error("Could not create/alter tables due to failed database connection.")
        return False 

def standardize_name(name):
    if not isinstance(name, str):
        try: name = str(name)
        except Exception: return "" 
    return ' '.join(name.lower().replace('_', ' ').replace('-', ' ').split())

def load_indicator_data_for_state(state_name_url_case, indicator_id_req):
    all_district_data = []
    full_indicator_name_text = f"Indicator ID {indicator_id_req}"

    # Check if BASE_JSON_DIR is a URL to a ZIP file
    if BASE_JSON_DIR.lower().startswith(('http://', 'https://')) and BASE_JSON_DIR.lower().endswith('.zip'):
        app.logger.info(f"Attempting to load indicator data from ZIP URL: {BASE_JSON_DIR}")
        try:
            response = requests.get(BASE_JSON_DIR, timeout=60) 
            response.raise_for_status()  
            zip_content = io.BytesIO(response.content)
            with zipfile.ZipFile(zip_content, 'r') as zf:
                state_path_prefix_in_zip = state_name_url_case.replace(os.path.sep, '/') + '/'
                candidate_files = [
                    name for name in zf.namelist()
                    if name.startswith(state_path_prefix_in_zip) and \
                       name.lower().endswith('.json') and \
                       name.count('/') == state_path_prefix_in_zip.count('/')
                ]
                if not candidate_files:
                    app.logger.warning(f"No JSON files found for state '{state_name_url_case}' (path prefix '{state_path_prefix_in_zip}') in ZIP from URL {BASE_JSON_DIR}")
                    return None, full_indicator_name_text
                for filepath_in_zip in candidate_files:
                    filename_part = filepath_in_zip.split('/')[-1]
                    district_name_from_file = standardize_name(filename_part.replace('.json', ''))
                    if not district_name_from_file: continue
                    try:
                        json_content_bytes = zf.read(filepath_in_zip)
                        data = json.loads(json_content_bytes.decode('utf-8'))
                        indicator_info = data.get('indicators', {}).get(indicator_id_req)
                        if indicator_info:
                            value = indicator_info.get('value')
                            current_indicator_text = indicator_info.get('indicator', full_indicator_name_text)
                            if indicator_id_req in current_indicator_text:
                                name_part = current_indicator_text.split(indicator_id_req, 1)[-1].strip()
                                if name_part.startswith((".", ")", ":")): name_part = name_part[1:].strip()
                                full_indicator_name_text = name_part if name_part else current_indicator_text
                            else: full_indicator_name_text = current_indicator_text
                            all_district_data.append({
                                'district_standardized': district_name_from_file,
                                'value': pd.to_numeric(value, errors='coerce'),
                                'indicator_name_text': full_indicator_name_text
                            })
                    except json.JSONDecodeError as jde:
                        app.logger.error(f"Error decoding JSON from {filepath_in_zip} in ZIP from URL {BASE_JSON_DIR}: {jde}", exc_info=True)
                    except Exception as e_file:
                        app.logger.error(f"Error processing file {filepath_in_zip} from ZIP URL {BASE_JSON_DIR}: {e_file}", exc_info=True)
        except requests.exceptions.RequestException as req_e:
            app.logger.error(f"Error downloading ZIP file from {BASE_JSON_DIR}: {req_e}", exc_info=True)
            return None, full_indicator_name_text
        except zipfile.BadZipFile:
            app.logger.error(f"Bad ZIP file from URL: {BASE_JSON_DIR}", exc_info=True)
            return None, full_indicator_name_text
        except Exception as e_zip_url:
            app.logger.error(f"Error processing ZIP from URL {BASE_JSON_DIR}: {e_zip_url}", exc_info=True)
            return None, full_indicator_name_text
    
    # Check if BASE_JSON_DIR points to a LOCAL ZIP file
    elif os.path.isfile(BASE_JSON_DIR) and BASE_JSON_DIR.lower().endswith('.zip'):
        app.logger.info(f"Attempting to load indicator data from LOCAL ZIP archive: {BASE_JSON_DIR}")
        try:
            with zipfile.ZipFile(BASE_JSON_DIR, 'r') as zf:
                state_path_prefix_in_zip = state_name_url_case.replace(os.path.sep, '/') + '/'
                candidate_files = [
                    name for name in zf.namelist() 
                    if name.startswith(state_path_prefix_in_zip) and \
                       name.lower().endswith('.json') and \
                       name.count('/') == state_path_prefix_in_zip.count('/') 
                ]
                if not candidate_files:
                    app.logger.warning(f"No JSON files for state '{state_name_url_case}' in LOCAL ZIP {BASE_JSON_DIR}")
                    return None, full_indicator_name_text
                for filepath_in_zip in candidate_files:
                    filename_part = filepath_in_zip.split('/')[-1]
                    district_name_from_file = standardize_name(filename_part.replace('.json', ''))
                    if not district_name_from_file: continue
                    try:
                        json_content_bytes = zf.read(filepath_in_zip)
                        data = json.loads(json_content_bytes.decode('utf-8'))
                        indicator_info = data.get('indicators', {}).get(indicator_id_req)
                        if indicator_info: 
                            value = indicator_info.get('value')
                            current_indicator_text = indicator_info.get('indicator', full_indicator_name_text)
                            if indicator_id_req in current_indicator_text:
                                name_part = current_indicator_text.split(indicator_id_req, 1)[-1].strip()
                                if name_part.startswith((".", ")", ":")): name_part = name_part[1:].strip()
                                full_indicator_name_text = name_part if name_part else current_indicator_text
                            else: full_indicator_name_text = current_indicator_text
                            all_district_data.append({
                                'district_standardized': district_name_from_file,
                                'value': pd.to_numeric(value, errors='coerce'),
                                'indicator_name_text': full_indicator_name_text 
                            })
                    except json.JSONDecodeError as jde:
                        app.logger.error(f"Error decoding JSON from {filepath_in_zip} in {BASE_JSON_DIR}: {jde}", exc_info=True)
                    except Exception as e_file:
                        app.logger.error(f"Error processing file {filepath_in_zip} from ZIP {BASE_JSON_DIR}: {e_file}", exc_info=True)
        except zipfile.BadZipFile: 
            app.logger.error(f"Bad ZIP file: {BASE_JSON_DIR}", exc_info=True)
            return None, full_indicator_name_text
        except FileNotFoundError: 
            app.logger.error(f"ZIP file not found: {BASE_JSON_DIR}", exc_info=True)
            return None, full_indicator_name_text
        except Exception as e_zip: 
            app.logger.error(f"Error reading ZIP file {BASE_JSON_DIR}: {e_zip}", exc_info=True)
            return None, full_indicator_name_text

    # Original logic for LOCAL directory-based JSONs
    elif os.path.isdir(BASE_JSON_DIR):
        state_json_path = os.path.join(BASE_JSON_DIR, state_name_url_case)
        if not os.path.isdir(state_json_path):
            app.logger.warning(f"State JSON directory not found: {state_json_path}")
            return None, f"Indicator ID {indicator_id_req}"
        app.logger.info(f"Loading indicator data from directory: {state_json_path}")
        for filename in os.listdir(state_json_path):
            if filename.endswith('.json'):
                district_name_from_file = standardize_name(filename.replace('.json', ''))
                if not district_name_from_file: continue
                filepath = os.path.join(state_json_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f: 
                        data = json.load(f)
                    indicator_info = data.get('indicators', {}).get(indicator_id_req)
                    if indicator_info: 
                        value = indicator_info.get('value')
                        current_indicator_text = indicator_info.get('indicator', full_indicator_name_text)
                        if indicator_id_req in current_indicator_text:
                            name_part = current_indicator_text.split(indicator_id_req, 1)[-1].strip()
                            if name_part.startswith((".", ")", ":")): name_part = name_part[1:].strip()
                            full_indicator_name_text = name_part if name_part else current_indicator_text
                        else: full_indicator_name_text = current_indicator_text
                        all_district_data.append({
                            'district_standardized': district_name_from_file,
                            'value': pd.to_numeric(value, errors='coerce'),
                            'indicator_name_text': full_indicator_name_text 
                        })
                except Exception as e: 
                    app.logger.error(f"Error processing file {filepath} from directory: {e}", exc_info=True)
    else:
        app.logger.error(f"BASE_JSON_DIR ('{BASE_JSON_DIR}') is not a valid URL, local zip file, or local directory.")
        return None, f"Indicator ID {indicator_id_req}"

    # Common processing part
    if not all_district_data:
        app.logger.info(f"No district data loaded for state '{state_name_url_case}', indicator '{indicator_id_req}'.")
        return None, full_indicator_name_text
    df_indicators = pd.DataFrame(all_district_data)
    if df_indicators.empty: return None, full_indicator_name_text
    df_indicators.dropna(subset=['value'], inplace=True)
    if df_indicators.empty: return None, full_indicator_name_text
    if 'indicator_name_text' in df_indicators.columns and not df_indicators.empty:
        common_name = df_indicators['indicator_name_text'].mode()
        if not common_name.empty: full_indicator_name_text = common_name[0]
        elif not df_indicators['indicator_name_text'].empty: full_indicator_name_text = df_indicators['indicator_name_text'].iloc[0]
        df_indicators['indicator_name_text'] = full_indicator_name_text
    return df_indicators, full_indicator_name_text

def _get_column_name(df_columns, env_var_value, possible_names_list, column_type_name, csv_path_for_logging):
    env_var_key_name = f'APP_CSV_{column_type_name.upper().replace(" ", "_")}_COL'
    if env_var_value and env_var_value in df_columns:
        app.logger.info(f"Using specified {column_type_name} column '{env_var_value}' (from env var {env_var_key_name}) for {csv_path_for_logging}.")
        return env_var_value
    elif env_var_value: app.logger.warning(f"Specified {column_type_name} column '{env_var_value}' (from env var {env_var_key_name}) not found in {csv_path_for_logging}. Available: {df_columns.tolist()}. Auto-detecting...")
    for col in possible_names_list:
        if col in df_columns:
            app.logger.info(f"Auto-detected {column_type_name} column: '{col}' in {csv_path_for_logging}.")
            return col
    app.logger.error(f"Could not auto-identify {column_type_name} column in {csv_path_for_logging}. Tried: {possible_names_list}. Available: {df_columns.tolist()}. Set {env_var_key_name}.")
    return None

def load_geographic_data_from_csv(state_name_standardized_filter):
    df_all_geo_points = None
    is_url = CSV_POINTS_PATH.lower().startswith(('http://', 'https://'))

    if not is_url and not os.path.isfile(CSV_POINTS_PATH):
        app.logger.error(f"Geographic CSV file not found (local path specified): {os.path.abspath(CSV_POINTS_PATH)}")
        return None, None
    
    app.logger.info(f"Attempting to load geographic CSV from: {CSV_POINTS_PATH}")
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'utf-16'] # Added utf-8 as first try

    for enc in encodings_to_try:
        try:
            # For URLs, pandas handles the download. For local files, it reads directly.
            df_all_geo_points = pd.read_csv(CSV_POINTS_PATH, encoding=enc)
            app.logger.info(f"Successfully loaded geographic CSV: {CSV_POINTS_PATH} using '{enc}' encoding.")
            break 
        except UnicodeDecodeError:
            app.logger.warning(f"UnicodeDecodeError with '{enc}' for {CSV_POINTS_PATH}. Trying next...")
        except pd.errors.EmptyDataError:
            app.logger.error(f"EmptyDataError: CSV file {CSV_POINTS_PATH} is empty or contains no data with encoding '{enc}'.")
            df_all_geo_points = pd.DataFrame() 
            break 
        except Exception as e: 
            app.logger.error(f"Error loading geographic CSV {CSV_POINTS_PATH} with '{enc}': {e}", exc_info=True)
            df_all_geo_points = None 
    
    if df_all_geo_points is None:
        app.logger.error(f"Failed to load geographic CSV {CSV_POINTS_PATH} after trying all encodings or due to other error.")
        return None, None
    
    current_csv_state_col = _get_column_name(df_all_geo_points.columns, ENV_CSV_STATE_COL, ['State_Name', 'state_name', 'State', 'state', 'NAME_1', 'ADM1_EN', 'ST_NM'], "state name", CSV_POINTS_PATH)
    if not current_csv_state_col: return None, None
    df_all_geo_points['state_standardized_csv'] = df_all_geo_points[current_csv_state_col].astype(str).apply(standardize_name)
    df_state_geo_points = df_all_geo_points[df_all_geo_points['state_standardized_csv'] == state_name_standardized_filter].copy()
    if df_state_geo_points.empty: app.logger.warning(f"No geographic data for state '{state_name_standardized_filter}' in {CSV_POINTS_PATH}."); return None, None

    current_csv_district_col = _get_column_name(df_state_geo_points.columns, ENV_CSV_DISTRICT_COL, ['District_Name', 'district_name', 'District', 'district', 'NAME_2', 'ADM2_EN', 'dt_name', 'Dist_Name'], "district name", CSV_POINTS_PATH)
    if not current_csv_district_col: return None, None
    current_csv_lat_col = _get_column_name(df_state_geo_points.columns, ENV_CSV_LAT_COL, ['Latitude', 'latitude', 'Lat', 'lat', 'Y', 'y_coord'], "latitude", CSV_POINTS_PATH)
    if not current_csv_lat_col: return None, None
    current_csv_lon_col = _get_column_name(df_state_geo_points.columns, ENV_CSV_LON_COL, ['Longitude', 'longitude', 'Lon', 'lon', 'X', 'x_coord'], "longitude", CSV_POINTS_PATH)
    if not current_csv_lon_col: return None, None

    df_state_geo_points[current_csv_lat_col] = pd.to_numeric(df_state_geo_points[current_csv_lat_col], errors='coerce')
    df_state_geo_points[current_csv_lon_col] = pd.to_numeric(df_state_geo_points[current_csv_lon_col], errors='coerce')
    df_state_geo_points.dropna(subset=[current_csv_lat_col, current_csv_lon_col], inplace=True)
    if df_state_geo_points.empty: app.logger.warning(f"No valid lat/lon data for state '{state_name_standardized_filter}'."); return None, None
    
    try:
        geometry = gpd.points_from_xy(df_state_geo_points[current_csv_lon_col], df_state_geo_points[current_csv_lat_col])
        cols_to_keep = [col for col in [current_csv_district_col, current_csv_state_col] if col in df_state_geo_points.columns]
        gdf_districts = gpd.GeoDataFrame(df_state_geo_points[cols_to_keep], geometry=geometry, crs="EPSG:4326")
    except Exception as e: app.logger.error(f"Error creating GeoDataFrame for state '{state_name_standardized_filter}': {e}", exc_info=True); return None, current_csv_district_col
    gdf_districts['district_standardized_geo'] = gdf_districts[current_csv_district_col].astype(str).apply(standardize_name)
    gdf_districts = gdf_districts[gdf_districts['district_standardized_geo'] != ""]
    if gdf_districts.empty: app.logger.warning(f"GeoDataFrame for state '{state_name_standardized_filter}' empty after removing empty standardized district names."); return None, current_csv_district_col
    return gdf_districts, current_csv_district_col

def haversine(lat1, lon1, lat2, lon2):
    R = 6371; lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2]); dlon = lon2 - lon1; dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2; c = 2 * atan2(sqrt(a), sqrt(1 - a)); return R * c

def row_to_dict(row_raw):
    if not row_raw: return None; row = dict(row_raw)
    for k, v in row.items():
        if isinstance(v, datetime): row[k] = v.isoformat()
        elif isinstance(v, pd.Timestamp): row[k] = v.isoformat()
        elif not isinstance(v, (float, int, str, bool)) and v is not None:
            try: row[k] = str(v)
            except: row[k] = None
    return row

# --- API Endpoints ---
@app.route('/api/signup', methods=['POST'])
def signup():
    if not request.is_json: return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    username = data.get('username'); email = data.get('email'); phone_number = data.get('phone_number')
    password = data.get('password'); user_type = data.get('userType'); address = data.get('address')
    if not all([username, email, phone_number, password, user_type]): return jsonify({"error": "Missing required fields."}), 400
    valid_user_types = ['organizer', 'requester', 'local_organisation']
    if user_type not in valid_user_types: return jsonify({"error": f"Invalid user type. Must be one of: {', '.join(valid_user_types)}"}), 400
    if user_type == 'local_organisation' and not address: return jsonify({"error": "Address is required for Local Organisation user type."}), 400
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "Database connection failed."}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT id FROM users WHERE username = %s OR email = %s OR phone_number = %s", (username, email, phone_number))
            if cur.fetchone(): return jsonify({"error": "User with this username, email, or phone number already exists."}), 409
            sql_user_insert = "INSERT INTO users (username, email, phone_number, password_hash, user_type, address) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id, username, email, user_type, address, created_at;"
            cur.execute(sql_user_insert, (username, email, phone_number, hashed_password, user_type, address if user_type == 'local_organisation' else None))
            new_user_raw = cur.fetchone()
            if new_user_raw:
                new_user_id = new_user_raw['id']
                if user_type == 'requester':
                    cur.execute("INSERT INTO patients (user_id, name, email, phone_number, camp_id, created_by_organizer_id) VALUES (%s, %s, %s, %s, NULL, NULL)", (new_user_id, username, email, phone_number))
                conn.commit() 
                user_data_to_return = row_to_dict(new_user_raw)
                if 'user_type' in user_data_to_return: user_data_to_return['userType'] = user_data_to_return.pop('user_type')
                return jsonify({"message": "User created successfully!", "user": user_data_to_return}), 201
            else: 
                if conn: conn.rollback()
                return jsonify({"error": "User creation failed unexpectedly."}), 500
    except psycopg2.Error as e:
        if conn: conn.rollback()
        app.logger.error(f"[signup] Database error: {e}", exc_info=True)
        if hasattr(e, 'pgcode') and e.pgcode == '23505': return jsonify({"error": "A user with this username, email, or phone number already exists."}), 409
        return jsonify({"error": "An error occurred during registration."}), 500
    except Exception as e:
        if conn: conn.rollback()
        app.logger.error(f"[signup] Unexpected error: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred."}), 500
    finally:
        if conn and not conn.closed:
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[signup] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/login', methods=['POST'])
def login():
    if not request.is_json: return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json(); email = data.get('email'); password = data.get('password')
    if not email or not password: return jsonify({"error": "Email and password are required."}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "Database connection failed."}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT id, username, email, password_hash, user_type, address, created_at FROM users WHERE email = %s", (email,))
            user_raw = cur.fetchone()
            if user_raw and bcrypt.check_password_hash(user_raw['password_hash'], password):
                user_info = row_to_dict(user_raw)
                if 'user_type' in user_info: user_info['userType'] = user_info.pop('user_type')
                if 'password_hash' in user_info: del user_info['password_hash']
                return jsonify({"message": "Login successful!", "user": user_info}), 200
            else: return jsonify({"error": "Invalid email or password."}), 401
    except psycopg2.Error as e: app.logger.error(f"Database error during login: {e}", exc_info=True); return jsonify({"error": "An error occurred during login."}), 500
    except Exception as e: app.logger.error(f"[login] An unexpected error occurred: {e}", exc_info=True); return jsonify({"error": "An unexpected server error occurred during login."}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[login] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/heatmap_data', methods=['GET'])
def get_heatmap_data():
    state_name_req = request.args.get('state'); indicator_id_req = request.args.get('indicator_id')
    if not state_name_req or not indicator_id_req: return jsonify({"error": "Missing 'state' or 'indicator_id' query parameter"}), 400
    state_name_for_json_path = standardize_name(state_name_req).replace(' ', '_')
    state_name_standardized_filter = standardize_name(state_name_req)
    try:
        df_indicators, full_indicator_name = load_indicator_data_for_state(state_name_for_json_path, indicator_id_req)
        indicator_data_summary = {"count": int(len(df_indicators)) if df_indicators is not None else 0, "available": df_indicators is not None and not df_indicators.empty}
        if not indicator_data_summary["available"]:
            return jsonify({"type": "FeatureCollection", "features": [], "metadata": {"query_state": state_name_req, "query_indicator_id": indicator_id_req, "full_indicator_name": full_indicator_name, "message": f"No indicator data for state '{state_name_req}', ID '{indicator_id_req}'.", "indicator_data_summary": indicator_data_summary, "geographic_data_summary": {"available": False, "count": 0, "message": "Geographic data not loaded."}}}), 200
        
        gdf_state_districts, csv_district_col_name = load_geographic_data_from_csv(state_name_standardized_filter)
        geographic_data_summary = {"count": int(len(gdf_state_districts)) if gdf_state_districts is not None else 0, "available": gdf_state_districts is not None and not gdf_state_districts.empty, "message": ""}
        if not geographic_data_summary["available"]:
            geographic_data_summary["message"] = f"Geographic point data not found for state '{state_name_req}'."
            return jsonify({"type": "FeatureCollection", "features": [], "metadata": {"query_state": state_name_req, "query_indicator_id": indicator_id_req, "full_indicator_name": full_indicator_name, "message": geographic_data_summary["message"], "indicator_data_summary": indicator_data_summary, "geographic_data_summary": geographic_data_summary}}), 200
        
        merged_gdf = gdf_state_districts.merge(df_indicators, left_on='district_standardized_geo', right_on='district_standardized', how='left')
        merged_gdf['value'] = pd.to_numeric(merged_gdf['value'], errors='coerce')
        matched_count = int(merged_gdf['value'].notna().sum().item()) if isinstance(merged_gdf['value'].notna().sum(), np.generic) else int(merged_gdf['value'].notna().sum())
        unmatched_geo_districts = merged_gdf[merged_gdf['value'].isna()]['district_standardized_geo'].tolist()
        unmatched_indicator_districts = list(set(df_indicators['district_standardized']) - set(gdf_state_districts['district_standardized_geo']))
        features_list = []
        for _, row in merged_gdf.iterrows():
            properties = {'original_csv_district_name': row.get(csv_district_col_name, "N/A") if pd.notna(row.get(csv_district_col_name)) else "N/A", 'district_standardized_geo': row['district_standardized_geo'], 'value': None if pd.isna(row['value']) else float(row['value']), 'indicator_id': indicator_id_req, 'indicator_name': full_indicator_name}
            if row['geometry'] and not row['geometry'].is_empty: features_list.append({"type": "Feature", "properties": properties, "geometry": gpd.GeoSeries([row['geometry']]).__geo_interface__['features'][0]['geometry']})
        
        final_message = f"Retrieved data for {len(features_list)} points." if features_list else "No points found/matched."
        response_geojson = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}, "features": features_list, "metadata": {"query_state": state_name_req, "query_indicator_id": indicator_id_req, "full_indicator_name": full_indicator_name, "message": final_message, "indicator_data_summary": indicator_data_summary, "geographic_data_summary": geographic_data_summary, "merge_summary": {"geo_districts_count": geographic_data_summary["count"], "indicator_districts_count": indicator_data_summary["count"], "matched_districts_count": matched_count, "unmatched_geo_districts_sample": unmatched_geo_districts[:5], "unmatched_indicator_districts_sample": unmatched_indicator_districts[:5]}}}
        return jsonify(response_geojson), 200
    except Exception as e:
        app.logger.error(f"Error in get_heatmap_data for state {state_name_req}, indicator {indicator_id_req}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred while fetching heatmap data."}), 500

@app.route('/api/organizer/camps', methods=['POST'])
def create_camp_endpoint():
    organizer_user_id_str = request.headers.get('X-User-Id')
    if not organizer_user_id_str: return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try: organizer_user_id = int(organizer_user_id_str)
    except ValueError: return jsonify({"error": "Invalid user identifier format."}), 400
    if not request.is_json: return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    required_fields = ['name', 'location_latitude', 'location_longitude', 'start_date', 'end_date']
    if not all(field in data and data[field] is not None for field in required_fields):
        missing = [field for field in required_fields if field not in data or data[field] is None]
        return jsonify({"error": f"Missing required camp data for fields: {', '.join(missing)}"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "Database connection failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (organizer_user_id,))
            user = cur.fetchone()
            if not user or user['user_type'] != 'organizer': return jsonify({"error": "Forbidden: Only organizers can create camps."}), 403
            cur.execute(
                "INSERT INTO camps (name, description, location_latitude, location_longitude, location_address, start_date, end_date, organizer_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id, name, description, location_latitude, location_longitude, location_address, start_date, end_date, organizer_id, created_at, status, target_patients;",
                (data['name'], data.get('description'), data['location_latitude'], data['location_longitude'], data.get('location_address'), data['start_date'], data['end_date'], organizer_user_id)
            )
            new_camp_raw = cur.fetchone()
            conn.commit()
            if new_camp_raw:
                new_camp = row_to_dict(new_camp_raw)
                for key in ['location_latitude', 'location_longitude']:
                    if new_camp.get(key) is not None: new_camp[key] = float(new_camp[key])
                return jsonify({"message": "Camp created successfully", "camp": new_camp}), 201
            else: 
                if conn: conn.rollback()
                return jsonify({"error": "Failed to create camp, no data returned."}), 500
    except psycopg2.Error as e:
        if conn: conn.rollback()
        app.logger.error(f"Error creating camp: {e}", exc_info=True)
        return jsonify({"error": "Failed to create camp"}), 500
    except Exception as e:
        if conn: conn.rollback()
        app.logger.error(f"Unexpected error creating camp: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred while creating camp."}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[create_camp_endpoint] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/organizer/camps', methods=['GET'])
def get_organizer_camps_endpoint():
    organizer_user_id_str = request.headers.get('X-User-Id')
    if not organizer_user_id_str: return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try: organizer_user_id = int(organizer_user_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID format."}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "Database connection failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (organizer_user_id,))
            user = cur.fetchone()
            if not user or user['user_type'] != 'organizer': return jsonify({"error": "Forbidden"}), 403
            cur.execute("SELECT id, name, description, location_latitude, location_longitude, location_address, start_date, end_date, organizer_id, status, target_patients FROM camps WHERE organizer_id = %s ORDER BY start_date DESC", (organizer_user_id,))
            camps_raw = cur.fetchall()
            camps = []
            for row_raw in camps_raw:
                camp = row_to_dict(row_raw)
                camp['lat'] = float(camp.pop('location_latitude')) if camp.get('location_latitude') is not None else None
                camp['lng'] = float(camp.pop('location_longitude')) if camp.get('location_longitude') is not None else None
                camps.append(camp)
            return jsonify(camps), 200
    except psycopg2.Error as e: app.logger.error(f"DB error get_organizer_camps: {e}", exc_info=True); return jsonify({"error": "Failed to fetch camps"}), 500
    except Exception as e: app.logger.error(f"Unexpected error get_organizer_camps: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[get_organizer_camps_endpoint] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/organizer/camps/<int:camp_id>', methods=['GET'])
def get_camp_details_endpoint(camp_id):
    requesting_user_id_str = request.headers.get('X-User-Id')
    if not requesting_user_id_str: return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try: requesting_user_id = int(requesting_user_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID format."}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "Database connection failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (requesting_user_id,))
            user = cur.fetchone()
            if not user or user['user_type'] != 'organizer': return jsonify({"error": "Forbidden"}), 403
            cur.execute("SELECT id, name, description, location_latitude, location_longitude, location_address, start_date, end_date, organizer_id, status, target_patients, created_at, updated_at FROM camps WHERE id = %s", (camp_id,))
            camp_raw = cur.fetchone()
            if not camp_raw: return jsonify({"message": "Camp not found."}), 404
            if camp_raw['organizer_id'] != requesting_user_id: return jsonify({"error": "Forbidden"}), 403
            camp = row_to_dict(camp_raw)
            if camp.get('location_latitude') is not None: camp['location_latitude'] = float(camp['location_latitude'])
            if camp.get('location_longitude') is not None: camp['location_longitude'] = float(camp['location_longitude'])
            return jsonify(camp), 200
    except psycopg2.Error as e: app.logger.error(f"DB error get_camp_details: {e}", exc_info=True); return jsonify({"error": "Failed to fetch details"}), 500
    except Exception as e: app.logger.error(f"Unexpected error get_camp_details: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[get_camp_details_endpoint] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/organizer/camps/<int:camp_id>', methods=['DELETE'])
def delete_camp_endpoint(camp_id):
    organizer_user_id_str = request.headers.get('X-User-Id')
    if not organizer_user_id_str: return jsonify({"error": "Unauthorized"}), 401
    try: requesting_organizer_id = int(organizer_user_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (requesting_organizer_id,))
            user_check = cur.fetchone()
            if not user_check or user_check['user_type'] != 'organizer': return jsonify({"error": "Forbidden"}), 403
            cur.execute("SELECT organizer_id FROM camps WHERE id = %s", (camp_id,))
            camp = cur.fetchone()
            if not camp: return jsonify({"error": "Camp not found."}), 404
            if camp['organizer_id'] != requesting_organizer_id: return jsonify({"error": "Forbidden"}), 403
            cur.execute("DELETE FROM camps WHERE id = %s", (camp_id,))
            if cur.rowcount == 0: 
                if conn: conn.rollback()
                return jsonify({"error": "Camp not found or failed to delete."}), 404 
            conn.commit()
            return jsonify({"message": f"Camp {camp_id} deleted."}), 200
    except psycopg2.Error as e:
        if conn: conn.rollback()
        app.logger.error(f"DB error delete_camp: {e}", exc_info=True); return jsonify({"error": "Failed to delete"}), 500
    except Exception as e:
        if conn: conn.rollback()
        app.logger.error(f"Unexpected error delete_camp: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[delete_camp_endpoint] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/organizer/camp/<int:camp_id>/resources', methods=['GET'])
def get_camp_resources(camp_id):
    requesting_user_id_str = request.headers.get('X-User-Id')
    if not requesting_user_id_str: return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try: requesting_user_id = int(requesting_user_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID format."}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "Database connection failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (requesting_user_id,))
            user_check = cur.fetchone()
            if not user_check or user_check['user_type'] != 'organizer': return jsonify({"error": "Forbidden"}), 403
            cur.execute("SELECT target_patients, organizer_id FROM camps WHERE id = %s", (camp_id,))
            camp_info = cur.fetchone()
            if not camp_info: return jsonify({"error": "Camp not found"}), 404
            if camp_info['organizer_id'] != requesting_user_id: return jsonify({"error": "Forbidden"}), 403
            target_patients = camp_info['target_patients']
            cur.execute("SELECT id, name, role, origin, contact, notes FROM camp_staff WHERE camp_id = %s", (camp_id,))
            staff_list = [row_to_dict(row) for row in cur.fetchall()]
            cur.execute("SELECT id, name, unit, quantity_per_patient, notes FROM camp_medicines WHERE camp_id = %s", (camp_id,))
            medicine_list_raw = cur.fetchall()
            medicine_list = []
            for med_raw in medicine_list_raw:
                med = row_to_dict(med_raw)
                if med.get('quantity_per_patient') is not None: med['quantity_per_patient'] = float(med['quantity_per_patient'])
                medicine_list.append(med)
            cur.execute("SELECT id, name, quantity, notes FROM camp_equipment WHERE camp_id = %s", (camp_id,))
            equipment_list = [row_to_dict(row) for row in cur.fetchall()]
            return jsonify({"targetPatients": target_patients, "staffList": staff_list, "medicineList": medicine_list, "equipmentList": equipment_list}), 200
    except psycopg2.Error as e: app.logger.error(f"DB error get_camp_resources: {e}", exc_info=True); return jsonify({"error": "Failed to fetch resources"}), 500
    except Exception as e: app.logger.error(f"Unexpected error get_camp_resources: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[get_camp_resources] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/organizer/camp/<int:camp_id>/resources', methods=['POST'])
def save_camp_resources(camp_id):
    organizer_user_id_str = request.headers.get('X-User-Id')
    if not organizer_user_id_str: return jsonify({"error": "Unauthorized"}), 401
    try: requesting_organizer_id = int(organizer_user_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    if not request.is_json: return jsonify({"error": "Missing JSON"}), 400
    data = request.get_json()
    target_patients = data.get('targetPatients')
    staff_list = data.get('staffList', [])
    medicine_list = data.get('medicineList', [])
    equipment_list = data.get('equipmentList', [])
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (requesting_organizer_id,))
            user_check = cur.fetchone()
            if not user_check or user_check['user_type'] != 'organizer': return jsonify({"error": "Forbidden"}), 403
            cur.execute("SELECT organizer_id FROM camps WHERE id = %s", (camp_id,))
            camp_owner = cur.fetchone()
            if not camp_owner: return jsonify({"error": "Camp not found"}), 404
            if camp_owner['organizer_id'] != requesting_organizer_id: return jsonify({"error": "Forbidden"}), 403
            if target_patients is not None: cur.execute("UPDATE camps SET target_patients = %s WHERE id = %s", (target_patients, camp_id))
            cur.execute("DELETE FROM camp_staff WHERE camp_id = %s", (camp_id,))
            for staff in staff_list: cur.execute("INSERT INTO camp_staff (camp_id, name, role, origin, contact, notes) VALUES (%s, %s, %s, %s, %s, %s)", (camp_id, staff.get('name'), staff.get('role'), staff.get('origin'), staff.get('contact'), staff.get('notes')))
            cur.execute("DELETE FROM camp_medicines WHERE camp_id = %s", (camp_id,))
            for med in medicine_list: cur.execute("INSERT INTO camp_medicines (camp_id, name, unit, quantity_per_patient, notes) VALUES (%s, %s, %s, %s, %s)", (camp_id, med.get('name'), med.get('unit'), med.get('quantityPerPatient'), med.get('notes')))
            cur.execute("DELETE FROM camp_equipment WHERE camp_id = %s", (camp_id,))
            for equip in equipment_list: cur.execute("INSERT INTO camp_equipment (camp_id, name, quantity, notes) VALUES (%s, %s, %s, %s)", (camp_id, equip.get('name'), equip.get('quantity'), equip.get('notes')))
            conn.commit()
            return jsonify({"message": "Resources saved"}), 200
    except psycopg2.Error as e:
        if conn: conn.rollback()
        app.logger.error(f"DB error save_camp_resources: {e}", exc_info=True); return jsonify({"error": "Failed to save"}), 500
    except Exception as e:
        if conn: conn.rollback()
        app.logger.error(f"Unexpected error save_camp_resources: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[save_camp_resources] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/organizer/camp/<int:camp_id>/patients', methods=['POST'])
def add_patient_to_camp(camp_id):
    organizer_user_id_str = request.headers.get('X-User-Id')
    if not organizer_user_id_str: return jsonify({"error": "Unauthorized"}), 401
    try: current_organizer_id = int(organizer_user_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    if not request.is_json: return jsonify({"error": "Missing JSON"}), 400
    data = request.get_json()
    patient_name = data.get('name'); patient_email = data.get('email')
    if not patient_name or not patient_email: return jsonify({"error": "Name and email required"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (current_organizer_id,))
            user_check = cur.fetchone()
            if not user_check or user_check['user_type'] != 'organizer': return jsonify({"error": "Forbidden"}), 403
            cur.execute("SELECT organizer_id FROM camps WHERE id = %s", (camp_id,))
            camp = cur.fetchone()
            if not camp: return jsonify({"error": "Camp not found"}), 404
            if camp['organizer_id'] != current_organizer_id: return jsonify({"error": "Forbidden"}), 403
            cur.execute("SELECT id FROM patients WHERE email = %s AND camp_id = %s", (patient_email, camp_id))
            if cur.fetchone(): return jsonify({"error": f"Patient {patient_email} exists in camp."}), 409
            cur.execute("SELECT id, user_type FROM users WHERE email = %s", (patient_email,))
            existing_user = cur.fetchone()
            patient_user_id_to_link = existing_user['id'] if existing_user and existing_user['user_type'] == 'requester' else None
            sql_insert = "INSERT INTO patients (camp_id, user_id, name, email, phone_number, disease_detected, area_location, organizer_notes, created_by_organizer_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id, camp_id, user_id, name, email, phone_number, disease_detected, area_location, organizer_notes, created_by_organizer_id, created_at;"
            params = (camp_id, patient_user_id_to_link, patient_name, patient_email, data.get('phone_number'), data.get('disease_detected'), data.get('area_location'), data.get('organizer_notes'), current_organizer_id)
            cur.execute(sql_insert, params)
            new_patient_raw = cur.fetchone()
            if not new_patient_raw:
                if conn: conn.rollback(); return jsonify({"error": "Failed to add patient."}), 500
            conn.commit()
            patient_dict = row_to_dict(new_patient_raw)
            cur.execute("SELECT name FROM camps WHERE id = %s", (patient_dict['camp_id'],))
            camp_details = cur.fetchone()
            patient_dict['camp_name'] = camp_details['name'] if camp_details else None
            patient_dict['is_registered_user'] = patient_dict['user_id'] is not None
            return jsonify({"message": "Patient added", "patient": patient_dict}), 201
    except psycopg2.Error as e:
        if conn: conn.rollback()
        app.logger.error(f"DB error add_patient_to_camp: {e}", exc_info=True)
        if hasattr(e, 'pgcode') and e.pgcode == '23505': return jsonify({"error": "Patient might already exist."}), 409
        return jsonify({"error": "DB error adding patient."}), 500
    except Exception as e:
        if conn: conn.rollback()
        app.logger.error(f"Unexpected error add_patient_to_camp: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[add_patient_to_camp] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/organizer/camp/<int:camp_id>/patients', methods=['GET'])
def get_camp_patients(camp_id):
    organizer_user_id_str = request.headers.get('X-User-Id')
    if not organizer_user_id_str: return jsonify({"error": "Unauthorized"}), 401
    try: current_organizer_id = int(organizer_user_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (current_organizer_id,))
            user_check = cur.fetchone()
            if not user_check or user_check['user_type'] != 'organizer': return jsonify({"error": "Forbidden"}), 403
            cur.execute("SELECT organizer_id FROM camps WHERE id = %s", (camp_id,))
            camp = cur.fetchone()
            if not camp: return jsonify({"error": "Camp not found"}), 404
            if camp['organizer_id'] != current_organizer_id: return jsonify({"error": "Forbidden"}), 403
            sql = "SELECT p.id, p.camp_id, c.name as camp_name, p.user_id, p.name, p.email, p.phone_number, p.disease_detected, p.area_location, p.organizer_notes, p.created_by_organizer_id, p.created_at FROM patients p JOIN camps c ON p.camp_id = c.id WHERE p.camp_id = %s ORDER BY p.name;"
            cur.execute(sql, (camp_id,))
            patients_raw = cur.fetchall()
            patients_list = [row_to_dict(p_raw) for p_raw in patients_raw]
            for p_dict in patients_list: p_dict['is_registered_user'] = p_dict['user_id'] is not None
            return jsonify(patients_list), 200
    except psycopg2.Error as e: app.logger.error(f"DB error get_camp_patients: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e: app.logger.error(f"Unexpected error get_camp_patients: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[get_camp_patients] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/patient/my-details', methods=['GET'])
def get_my_patient_details():
    current_user_id_str = request.headers.get('X-User-Id')
    if not current_user_id_str: return jsonify({"error": "Unauthorized"}), 401
    try: current_user_id = int(current_user_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT email, user_type FROM users WHERE id = %s", (current_user_id,))
            user = cur.fetchone()
            if not user: return jsonify({"error": "User not found."}), 404
            user_email = user['email']
            cur.execute("UPDATE patients SET user_id = %s WHERE email = %s AND user_id IS NULL AND camp_id IS NOT NULL", (current_user_id, user_email))
            conn.commit()
            sql_user = "SELECT p.id, p.camp_id, c.name as camp_name, p.user_id, p.name, p.email, p.phone_number, p.disease_detected, p.area_location, p.organizer_notes, p.created_by_organizer_id, p.created_at FROM patients p LEFT JOIN camps c ON p.camp_id = c.id WHERE p.user_id = %s ORDER BY p.created_at DESC;"
            cur.execute(sql_user, (current_user_id,))
            profiles_raw = cur.fetchall()
            if not profiles_raw:
                sql_email = "SELECT p.id, p.camp_id, c.name as camp_name, p.user_id, p.name, p.email, p.phone_number, p.disease_detected, p.area_location, p.organizer_notes, p.created_by_organizer_id, p.created_at FROM patients p JOIN camps c ON p.camp_id = c.id WHERE p.email = %s AND p.user_id IS NULL ORDER BY p.created_at DESC;"
                cur.execute(sql_email, (user_email,))
                profiles_raw = cur.fetchall()
            if not profiles_raw: return jsonify({"message": "No patient records found."}), 404
            return jsonify([row_to_dict(p_raw) for p_raw in profiles_raw]), 200
    except psycopg2.Error as e:
        if conn: conn.rollback()
        app.logger.error(f"DB error get_my_patient_details: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e:
        if conn: conn.rollback()
        app.logger.error(f"Unexpected error get_my_patient_details: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[get_my_patient_details] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/local-organisations', methods=['GET'])
def get_local_organisations():
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT id, username, email, address, phone_number FROM users WHERE user_type = 'local_organisation'")
            orgs_raw = cur.fetchall()
            orgs = [row_to_dict(row_raw) for row_raw in orgs_raw]
            for org in orgs: org['name'] = org.pop('username')
            return jsonify(orgs), 200
    except psycopg2.Error as e: app.logger.error(f"DB error get_local_organisations: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e: app.logger.error(f"Unexpected error get_local_organisations: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[get_local_organisations] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/chat/request', methods=['POST'])
def send_connection_request():
    organizer_id_str = request.headers.get('X-User-Id')
    if not organizer_id_str: return jsonify({"error": "Unauthorized"}), 401
    if not request.is_json: return jsonify({"error": "Missing JSON"}), 400
    data = request.get_json()
    camp_id_str = data.get('campId'); local_org_id_str = data.get('localOrgId')
    try:
        organizer_id = int(organizer_id_str)
        if camp_id_str is None: return jsonify({"error": "campId required"}), 400
        camp_id = int(camp_id_str)
        if local_org_id_str is None: return jsonify({"error": "localOrgId required"}), 400
        local_org_id = int(local_org_id_str)
    except (ValueError, TypeError) as e: app.logger.error(f"Invalid ID format: {e}", exc_info=True); return jsonify({"error": "Invalid ID format"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (organizer_id,))
            user_check = cur.fetchone()
            if not user_check or user_check['user_type'] != 'organizer': return jsonify({"error": "Forbidden"}), 403
            cur.execute("SELECT id FROM camps WHERE id = %s AND organizer_id = %s", (camp_id, organizer_id))
            if not cur.fetchone(): return jsonify({"error": "Camp not found or not owned"}), 404
            cur.execute("SELECT id FROM users WHERE id = %s AND user_type = 'local_organisation'", (local_org_id,))
            if not cur.fetchone(): return jsonify({"error": "Local org not found"}), 404
            cur.execute("INSERT INTO connection_requests (camp_id, organizer_id, local_org_id) VALUES (%s, %s, %s) RETURNING id, status, requested_at;", (camp_id, organizer_id, local_org_id))
            new_req_raw = cur.fetchone()
            conn.commit()
            if new_req_raw: return jsonify({"message": "Request sent", "request": row_to_dict(new_req_raw)}), 201
            else: if conn: conn.rollback(); return jsonify({"error": "Failed to create request"}), 500
    except psycopg2.IntegrityError as e:
        if conn: conn.rollback()
        app.logger.warning(f"Integrity error send_connection_request: {e}", exc_info=True); return jsonify({"error": "Request already exists or invalid IDs"}), 409
    except psycopg2.Error as e:
        if conn: conn.rollback()
        app.logger.error(f"DB error send_connection_request: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e:
        if conn: conn.rollback()
        app.logger.error(f"Unexpected error send_connection_request: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[send_connection_request] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/local-organisation/<int:user_id>/requests', methods=['GET'])
def get_local_org_requests(user_id):
    requesting_user_id_str = request.headers.get('X-User-Id')
    if not requesting_user_id_str: return jsonify({"error": "Unauthorized"}), 401
    try:
        requesting_user_id = int(requesting_user_id_str)
        if requesting_user_id != user_id: return jsonify({"error": "Forbidden"}), 403
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (user_id,))
            user_details = cur.fetchone()
            if not user_details or user_details['user_type'] != 'local_organisation': return jsonify({"error": "Forbidden"}), 403
            sql = "SELECT cr.id as request_id, cr.status, cr.requested_at, c.id as camp_id, c.name as camp_name, c.start_date as camp_start_date, u.id as organizer_id, u.username as organizer_name FROM connection_requests cr JOIN camps c ON cr.camp_id = c.id JOIN users u ON cr.organizer_id = u.id WHERE cr.local_org_id = %s AND cr.status = 'pending' ORDER BY cr.requested_at DESC;"
            cur.execute(sql, (user_id,))
            reqs_raw = cur.fetchall()
            return jsonify({"pendingRequests": [row_to_dict(req_raw) for req_raw in reqs_raw]}), 200
    except psycopg2.Error as e: app.logger.error(f"DB error get_local_org_requests: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e: app.logger.error(f"Unexpected error get_local_org_requests: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[get_local_org_requests] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/local-organisation/<int:user_id>/connections', methods=['GET'])
def get_local_org_connections(user_id):
    requesting_user_id_str = request.headers.get('X-User-Id')
    if not requesting_user_id_str: return jsonify({"error": "Unauthorized"}), 401
    try:
        requesting_user_id = int(requesting_user_id_str)
        if requesting_user_id != user_id: return jsonify({"error": "Forbidden"}), 403
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    status_filter = request.args.get('status')
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (user_id,))
            user_details = cur.fetchone()
            if not user_details or user_details['user_type'] != 'local_organisation': return jsonify({"error": "Forbidden"}), 403
            sql = "SELECT cr.id as connection_id, cr.camp_id, c.name as camp_name, cr.organizer_id, u_org.username as organizer_name, cr.status, cr.requested_at, cr.responded_at FROM connection_requests cr JOIN camps c ON cr.camp_id = c.id JOIN users u_org ON cr.organizer_id = u_org.id WHERE cr.local_org_id = %s"
            params = [user_id]
            if status_filter: sql += " AND cr.status = %s"; params.append(status_filter)
            sql += " ORDER BY cr.responded_at DESC, cr.requested_at DESC;"
            cur.execute(sql, tuple(params))
            conns_raw = cur.fetchall()
            return jsonify([row_to_dict(row_raw) for row_raw in conns_raw]), 200
    except psycopg2.Error as e: app.logger.error(f"DB error get_local_org_connections: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e: app.logger.error(f"Unexpected error get_local_org_connections: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[get_local_org_connections] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/chat/request/<int:request_id>/respond', methods=['PUT'])
def respond_to_connection_request(request_id):
    local_org_user_id_str = request.headers.get('X-User-Id')
    if not local_org_user_id_str: return jsonify({"error": "Unauthorized"}), 401
    try: local_org_user_id = int(local_org_user_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    if not request.is_json: return jsonify({"error": "Missing JSON"}), 400
    data = request.get_json(); new_status = data.get('status')
    if new_status not in ['accepted', 'declined']: return jsonify({"error": "Invalid status"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT id, status, local_org_id FROM connection_requests WHERE id = %s", (request_id,))
            req = cur.fetchone()
            if not req: return jsonify({"error": "Request not found"}), 404
            if req['local_org_id'] != local_org_user_id: return jsonify({"error": "Forbidden"}), 403
            if req['status'] != 'pending': return jsonify({"error": f"Request already responded ({req['status']})"}), 400
            cur.execute("UPDATE connection_requests SET status = %s, responded_at = CURRENT_TIMESTAMP WHERE id = %s RETURNING id, status, responded_at;", (new_status, request_id))
            updated_req_raw = cur.fetchone()
            conn.commit()
            if updated_req_raw: return jsonify({"message": f"Request {new_status}", "request": row_to_dict(updated_req_raw)}), 200
            else: if conn: conn.rollback(); return jsonify({"error": "Failed to update"}), 500
    except psycopg2.Error as e:
        if conn: conn.rollback()
        app.logger.error(f"DB error respond_to_connection_request: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e:
        if conn: conn.rollback()
        app.logger.error(f"Unexpected error respond_to_connection_request: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[respond_to_connection_request] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/organizer/camp/<int:camp_id>/connections', methods=['GET'])
def get_organizer_camp_connections(camp_id):
    organizer_id_str = request.headers.get('X-User-Id')
    if not organizer_id_str: return jsonify({"error": "Unauthorized"}), 401
    try: organizer_id = int(organizer_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (organizer_id,))
            user_check = cur.fetchone()
            if not user_check or user_check['user_type'] != 'organizer': return jsonify({"error": "Forbidden"}), 403
            cur.execute("SELECT id FROM camps WHERE id = %s AND organizer_id = %s", (camp_id, organizer_id))
            if not cur.fetchone(): return jsonify({"error": "Camp not found or not owned"}), 404
            sql = "SELECT cr.id as connection_id, cr.local_org_id, u_local_org.username as local_org_name, cr.status, cr.requested_at, cr.responded_at FROM connection_requests cr JOIN users u_local_org ON cr.local_org_id = u_local_org.id WHERE cr.camp_id = %s AND cr.organizer_id = %s;"
            cur.execute(sql, (camp_id, organizer_id))
            conns_raw = cur.fetchall()
            return jsonify([row_to_dict(conn_req) for conn_req in conns_raw]), 200
    except psycopg2.Error as e: app.logger.error(f"DB error get_organizer_camp_connections: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e: app.logger.error(f"Unexpected error get_organizer_camp_connections: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[get_organizer_camp_connections] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/chat/conversation/<int:connection_id>/messages', methods=['GET'])
def get_chat_messages(connection_id):
    user_id_str = request.headers.get('X-User-Id')
    if not user_id_str: return jsonify({"error": "Unauthorized"}), 401
    try: user_id = int(user_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT organizer_id, local_org_id, status FROM connection_requests WHERE id = %s", (connection_id,))
            conn_req = cur.fetchone()
            if not conn_req: return jsonify({"error": "Connection not found"}), 404
            if conn_req['status'] != 'accepted': return jsonify({"error": "Chat not active"}), 403
            if user_id not in [conn_req['organizer_id'], conn_req['local_org_id']]: return jsonify({"error": "Forbidden"}), 403
            sql = "SELECT cm.id, cm.sender_id, u.username as sender_name, cm.message_text, cm.sent_at FROM chat_messages cm JOIN users u ON cm.sender_id = u.id WHERE cm.connection_request_id = %s ORDER BY cm.sent_at ASC;"
            cur.execute(sql, (connection_id,))
            msgs_raw = cur.fetchall()
            return jsonify([row_to_dict(msg_raw) for msg_raw in msgs_raw]), 200
    except psycopg2.Error as e: app.logger.error(f"DB error get_chat_messages: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e: app.logger.error(f"Unexpected error get_chat_messages: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[get_chat_messages] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/chat/conversation/<int:connection_id>/message', methods=['POST'])
def send_chat_message(connection_id):
    sender_id_str = request.headers.get('X-User-Id')
    if not sender_id_str: return jsonify({"error": "Unauthorized"}), 401
    try: sender_id = int(sender_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    if not request.is_json: return jsonify({"error": "Missing JSON"}), 400
    data = request.get_json(); message_text = data.get('text')
    if not message_text or not message_text.strip(): return jsonify({"error": "Message empty"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT cr.organizer_id, cr.local_org_id, cr.status FROM connection_requests cr WHERE cr.id = %s", (connection_id,))
            conn_req = cur.fetchone()
            if not conn_req: return jsonify({"error": "Connection not found"}), 404
            if conn_req['status'] != 'accepted': return jsonify({"error": "Chat not active"}), 403
            if sender_id not in [conn_req['organizer_id'], conn_req['local_org_id']]: return jsonify({"error": "Forbidden"}), 403
            cur.execute("INSERT INTO chat_messages (connection_request_id, sender_id, message_text) VALUES (%s, %s, %s) RETURNING id, sender_id, message_text, sent_at;", (connection_id, sender_id, message_text))
            new_msg_raw = cur.fetchone()
            conn.commit()
            if new_msg_raw:
                new_msg = row_to_dict(new_msg_raw)
                cur.execute("SELECT username FROM users WHERE id = %s", (new_msg['sender_id'],))
                sender_details = cur.fetchone()
                if sender_details: new_msg['sender_name'] = sender_details['username']
                return jsonify({"message": "Message sent", "chatMessage": new_msg}), 201
            else: if conn: conn.rollback(); return jsonify({"error": "Failed to send"}), 500
    except psycopg2.Error as e:
        if conn: conn.rollback()
        app.logger.error(f"DB error send_chat_message: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e:
        if conn: conn.rollback()
        app.logger.error(f"Unexpected error send_chat_message: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[send_chat_message] Error closing connection: {e_close}", exc_info=True)

def translate_text_local_hf(text, target_lang_simple, source_lang_simple="auto"):
    global local_translation_pipeline, LOCAL_TRANSLATION_MODEL_INIT_STATUS, LANGUAGE_CODE_MAP_NLLB
    if LOCAL_TRANSLATION_MODEL_INIT_STATUS == "pending": initialize_local_translation_model()
    if LOCAL_TRANSLATION_MODEL_INIT_STATUS == "failed" or local_translation_pipeline is None:
        app.logger.error(f"Local translation model {HF_TRANSLATION_MODEL_ID} unavailable."); return text
    if not text or not text.strip(): return text
    nllb_target = LANGUAGE_CODE_MAP_NLLB.get(target_lang_simple)
    nllb_source = LANGUAGE_CODE_MAP_NLLB.get("en") if source_lang_simple == "auto" and target_lang_simple != "en" else LANGUAGE_CODE_MAP_NLLB.get(source_lang_simple)
    if source_lang_simple == "auto" and target_lang_simple == "en":
        app.logger.error("Ambiguous auto source to 'en' for NLLB."); return text
    if not nllb_target or not nllb_source: app.logger.error("Unsupported lang for NLLB."); return text
    if nllb_source == nllb_target: return text
    try:
        result = local_translation_pipeline(text, src_lang=nllb_source, tgt_lang=nllb_target)
        if result and isinstance(result, list) and result[0] and "translation_text" in result[0]:
            return result[0]["translation_text"]
        app.logger.error(f"Unexpected NLLB translation format: {result}"); return text
    except Exception as e: app.logger.error(f"NLLB translation error: {e}", exc_info=True); return text

def query_huggingface_model_local(prompt_text):
    global local_chatbot_pipeline, local_chatbot_tokenizer, LOCAL_CHATBOT_MODEL_INIT_STATUS
    if LOCAL_CHATBOT_MODEL_INIT_STATUS == "pending": initialize_local_chatbot_model()
    if LOCAL_CHATBOT_MODEL_INIT_STATUS == "failed" or not all([local_chatbot_pipeline, local_chatbot_tokenizer]):
        app.logger.error(f"Local chatbot model {HF_CHATBOT_MODEL_ID} unavailable."); return INTERNAL_BOT_ERROR_MESSAGES[0]
    try:
        tokens = local_chatbot_tokenizer.encode(prompt_text, return_tensors='pt')
        prompt_len = tokens.shape[1]; max_new = 150
        max_len_cfg = getattr(local_chatbot_pipeline.model.config, 'max_position_embeddings', getattr(local_chatbot_pipeline.model.config, 'n_positions', 512))
        calc_max_len = min(prompt_len + max_new, max_len_cfg)
        if prompt_len >= calc_max_len: return INTERNAL_BOT_ERROR_MESSAGES[1]
        results = local_chatbot_pipeline(prompt_text, max_length=calc_max_len, num_return_sequences=1)
        if results and isinstance(results, list) and results[0] and "generated_text" in results[0]:
            full_text = results[0]["generated_text"]
            response = full_text[len(prompt_text):].strip() if full_text.startswith(prompt_text) else full_text.split("Assistant:", 1)[-1].strip() if "Assistant:" in full_text else full_text
            return response
        app.logger.error(f"Unexpected local model format: {results}"); return INTERNAL_BOT_ERROR_MESSAGES[2]
    except Exception as e: app.logger.error(f"Local HF model query error: {e}", exc_info=True); return INTERNAL_BOT_ERROR_MESSAGES[3]

@app.route('/api/translate', methods=['POST'])
def translate_api_endpoint():
    if not request.is_json: return jsonify({"error": "Missing JSON"}), 400
    data = request.get_json()
    text = data.get('text'); target_lang = data.get('target_lang'); source_lang = data.get('source_lang', 'auto')
    if not text or not target_lang: return jsonify({"error": "Missing 'text' or 'target_lang'"}), 400
    try:
        translated = translate_text_local_hf(text, target_lang, source_lang)
        detected_src = "en (assumed)" if source_lang == 'auto' and target_lang != "en" else "auto (NLLB needs explicit source for 'en' target)" if source_lang == 'auto' else source_lang
        return jsonify({"translated_text": translated, "source_lang_detected": detected_src}), 200
    except Exception as e: app.logger.error(f"Translate API error: {e}", exc_info=True); return jsonify({"error": "Translation error"}), 500

@app.route('/api/patient/chatbot', methods=['POST'])
def patient_chatbot():
    user_id_str = request.headers.get('X-User-Id')
    if not user_id_str: return jsonify({"error": "Unauthorized"}), 401
    try: user_id = int(user_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    if not request.is_json: return jsonify({"error": "Missing JSON"}), 400
    data = request.get_json()
    user_msg = data.get('message'); target_lang = data.get('language', 'en'); patient_rec_id = data.get('patient_record_id')
    if not user_msg: return jsonify({"error": "Message required"}), 400
    
    disease, location, name = "not specified", "not specified", "Patient"
    conn_context, conn_store = None, None
    try:
        conn_context = get_db_connection()
        if conn_context:
            with conn_context.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                query = "SELECT name, disease_detected, area_location FROM patients WHERE "
                params = []
                if patient_rec_id: query += "id = %s AND user_id = %s"; params.extend([patient_rec_id, user_id])
                else: query += "user_id = %s ORDER BY created_at DESC LIMIT 1"; params.append(user_id)
                cur.execute(query, tuple(params))
                ctx = cur.fetchone()
                if ctx: name, disease, location = ctx['name'], ctx['disease_detected'] or disease, ctx['area_location'] or location
                cur.execute("INSERT INTO patient_chat_messages (patient_user_id, patient_record_id, message_text, sender_type, language) VALUES (%s, %s, %s, 'user', %s)", (user_id, patient_rec_id, user_msg, target_lang))
                conn_context.commit()
    except psycopg2.Error as e: app.logger.error(f"DB error chatbot context: {e}", exc_info=True); if conn_context: conn_context.rollback()
    finally:
        if conn_context and not conn_context.closed: 
            try: conn_context.close()
            except Exception as e_close: app.logger.error(f"[patient_chatbot] Error closing conn_context: {e_close}", exc_info=True)

    msg_for_bot = translate_text_local_hf(user_msg, "en", target_lang) if target_lang != 'en' else user_msg
    prompt = f"You are a helpful medical information assistant for GoMedCamp.\nA patient, {name}, is asking for information.\nPatient's detected condition: {disease}.\nPatient's location: {location}.\nThe patient says (translated to English for you, if originally not in English): \"{msg_for_bot}\"\n\nPlease provide helpful, general information. \nDo NOT give specific medical diagnoses or treatment plans.\nAlways advise the patient to consult with a qualified healthcare professional for any medical concerns or before making any health decisions.\nIf asked about where to go, suggest looking for local clinics, hospitals, or specialists in their area ({location}) and consulting the camp organizers for referrals if applicable.\nKeep your response concise and easy to understand. Respond in English.\n\nAssistant: "
    bot_reply_en = query_huggingface_model_local(prompt)
    final_reply = translate_text_local_hf(bot_reply_en, target_lang, "en") if target_lang != 'en' and bot_reply_en not in INTERNAL_BOT_ERROR_MESSAGES else bot_reply_en
    
    try:
        conn_store = get_db_connection()
        if conn_store:
            with conn_store.cursor() as cur_store:
                cur_store.execute("INSERT INTO patient_chat_messages (patient_user_id, patient_record_id, message_text, sender_type, language) VALUES (%s, %s, %s, 'bot', %s)", (user_id, patient_rec_id, final_reply, target_lang))
                conn_store.commit()
    except psycopg2.Error as e: app.logger.error(f"DB error storing bot msg: {e}", exc_info=True); if conn_store: conn_store.rollback()
    except Exception as e_gen: app.logger.error(f"Unexpected error storing bot msg: {e_gen}", exc_info=True); if conn_store: conn_store.rollback()
    finally:
        if conn_store and not conn_store.closed: 
            try: conn_store.close()
            except Exception as e_close: app.logger.error(f"[patient_chatbot] Error closing conn_store: {e_close}", exc_info=True)
            
    return jsonify({"reply": final_reply, "language": target_lang}), 200

@app.route('/api/patient/feedback', methods=['POST'])
def patient_feedback():
    user_id_str = request.headers.get('X-User-Id')
    if not user_id_str: return jsonify({"error": "Unauthorized"}), 401
    try: user_id = int(user_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    if not request.is_json: return jsonify({"error": "Missing JSON"}), 400
    data = request.get_json()
    text = data.get('feedback_text'); rating_val = data.get('rating'); rec_id = data.get('patient_record_id'); lang = data.get('language', 'en')
    if not text: return jsonify({"error": "Feedback text required"}), 400
    if rating_val is not None:
        try:
            rating_val = int(rating_val)
            if not (1 <= rating_val <= 5): return jsonify({"error": "Rating 1-5"}), 400
        except ValueError: return jsonify({"error": "Invalid rating"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor() as cur:
            cur.execute("INSERT INTO patient_feedback (patient_user_id, patient_record_id, feedback_text, rating, language) VALUES (%s, %s, %s, %s, %s)", (user_id, rec_id, text, rating_val, lang))
            conn.commit()
        return jsonify({"message": "Feedback submitted"}), 201
    except psycopg2.Error as e:
        if conn: conn.rollback()
        app.logger.error(f"DB error patient_feedback: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e:
        if conn: conn.rollback()
        app.logger.error(f"Unexpected error patient_feedback: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[patient_feedback] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/camps', methods=['GET'])
def get_all_camps_for_review():
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT id, name FROM camps WHERE status IN ('active', 'completed', 'planned') ORDER BY name ASC")
            camps_raw = cur.fetchall()
            return jsonify([row_to_dict(camp) for camp in camps_raw]), 200
    except psycopg2.Error as e: app.logger.error(f"DB error get_all_camps_for_review: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e: app.logger.error(f"Unexpected error get_all_camps_for_review: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[get_all_camps_for_review] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/reviews', methods=['POST'])
def submit_camp_review():
    user_id_str = request.headers.get('X-User-Id')
    if not user_id_str: return jsonify({"error": "Unauthorized"}), 401
    try: user_id = int(user_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    if not request.is_json: return jsonify({"error": "Missing JSON"}), 400
    data = request.get_json()
    camp_id_val = data.get('campId'); rating_val = data.get('rating'); comment_val = data.get('comment')
    if not camp_id_val or rating_val is None: return jsonify({"error": "campId, rating required"}), 400
    try:
        rating_val = int(rating_val)
        if not (1 <= rating_val <= 5): return jsonify({"error": "Rating 1-5"}), 400
    except ValueError: return jsonify({"error": "Invalid rating"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (user_id,))
            user = cur.fetchone()
            if not user or user['user_type'] != 'requester': return jsonify({"error": "Forbidden"}), 403
            cur.execute("SELECT id FROM camps WHERE id = %s", (camp_id_val,))
            if not cur.fetchone(): return jsonify({"error": "Camp not found"}), 404
            cur.execute("SELECT id FROM camp_reviews WHERE camp_id = %s AND patient_user_id = %s", (camp_id_val, user_id))
            if cur.fetchone(): return jsonify({"error": "Already reviewed"}), 409
            cur.execute("INSERT INTO camp_reviews (camp_id, patient_user_id, rating, comment) VALUES (%s, %s, %s, %s) RETURNING id;", (camp_id_val, user_id, rating_val, comment_val))
            review_id = cur.fetchone()['id']
            conn.commit()
            return jsonify({"message": "Review submitted", "review_id": review_id}), 201
    except psycopg2.Error as e:
        if conn: conn.rollback()
        app.logger.error(f"DB error submit_camp_review: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e:
        if conn: conn.rollback()
        app.logger.error(f"Unexpected error submit_camp_review: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[submit_camp_review] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/camps/<int:camp_id>/reviews', methods=['GET'])
def get_camp_reviews_for_organizer(camp_id):
    organizer_id_str = request.headers.get('X-User-Id')
    if not organizer_id_str: return jsonify({"error": "Unauthorized"}), 401
    try: organizer_id = int(organizer_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (organizer_id,))
            user = cur.fetchone()
            if not user or user['user_type'] != 'organizer': return jsonify({"error": "Forbidden"}), 403
            cur.execute("SELECT organizer_id FROM camps WHERE id = %s", (camp_id,))
            camp = cur.fetchone()
            if not camp: return jsonify({"error": "Camp not found"}), 404
            if camp['organizer_id'] != organizer_id: return jsonify({"error": "Forbidden"}), 403
            sql = "SELECT cr.id, cr.patient_user_id, u.username as patient_name, cr.rating, cr.comment, cr.created_at FROM camp_reviews cr JOIN users u ON cr.patient_user_id = u.id WHERE cr.camp_id = %s ORDER BY cr.created_at DESC;"
            cur.execute(sql, (camp_id,))
            reviews_raw = cur.fetchall()
            return jsonify([row_to_dict(review) for review in reviews_raw]), 200
    except psycopg2.Error as e: app.logger.error(f"DB error get_camp_reviews_for_organizer: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e: app.logger.error(f"Unexpected error get_camp_reviews_for_organizer: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[get_camp_reviews_for_organizer] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/camps/<int:camp_id>/patients/followup', methods=['POST'])
def add_patient_for_followup(camp_id):
    organizer_id_str = request.headers.get('X-User-Id')
    if not organizer_id_str: return jsonify({"error": "Unauthorized"}), 401
    try: organizer_id = int(organizer_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    if not request.is_json: return jsonify({"error": "Missing JSON"}), 400
    data = request.get_json()
    identifier = data.get('patientIdentifier'); notes_val = data.get('notes')
    if not identifier: return jsonify({"error": "Identifier required"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (organizer_id,))
            user = cur.fetchone()
            if not user or user['user_type'] != 'organizer': return jsonify({"error": "Forbidden"}), 403
            cur.execute("SELECT organizer_id FROM camps WHERE id = %s", (camp_id,))
            camp = cur.fetchone()
            if not camp: return jsonify({"error": "Camp not found"}), 404
            if camp['organizer_id'] != organizer_id: return jsonify({"error": "Forbidden"}), 403
            linked_user_id = None
            cur.execute("SELECT id FROM users WHERE (email = %s OR phone_number = %s) AND user_type = 'requester'", (identifier, identifier))
            matched = cur.fetchone()
            if matched: linked_user_id = matched['id']
            cur.execute("INSERT INTO camp_follow_ups (camp_id, patient_identifier, notes, added_by_organizer_id, linked_patient_user_id) VALUES (%s, %s, %s, %s, %s) RETURNING id, patient_identifier, notes, created_at;", (camp_id, identifier, notes_val, organizer_id, linked_user_id))
            new_fu_raw = cur.fetchone()
            conn.commit()
            return jsonify({"message": "Patient added for followup", "follow_up": row_to_dict(new_fu_raw)}), 201
    except psycopg2.Error as e:
        if conn: conn.rollback()
        app.logger.error(f"DB error add_patient_for_followup: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e:
        if conn: conn.rollback()
        app.logger.error(f"Unexpected error add_patient_for_followup: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[add_patient_for_followup] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/camps/<int:camp_id>/patients/followup', methods=['GET'])
def get_camp_followup_patients(camp_id):
    organizer_id_str = request.headers.get('X-User-Id')
    if not organizer_id_str: return jsonify({"error": "Unauthorized"}), 401
    try: organizer_id = int(organizer_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (organizer_id,))
            user = cur.fetchone()
            if not user or user['user_type'] != 'organizer': return jsonify({"error": "Forbidden"}), 403
            cur.execute("SELECT organizer_id FROM camps WHERE id = %s", (camp_id,))
            camp = cur.fetchone()
            if not camp: return jsonify({"error": "Camp not found"}), 404
            if camp['organizer_id'] != organizer_id: return jsonify({"error": "Forbidden"}), 403
            cur.execute("SELECT id, patient_identifier, notes, created_at, linked_patient_user_id FROM camp_follow_ups WHERE camp_id = %s ORDER BY created_at DESC;", (camp_id,))
            fus_raw = cur.fetchall()
            return jsonify([row_to_dict(fu) for fu in fus_raw]), 200
    except psycopg2.Error as e: app.logger.error(f"DB error get_camp_followup_patients: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e: app.logger.error(f"Unexpected error get_camp_followup_patients: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[get_camp_followup_patients] Error closing connection: {e_close}", exc_info=True)

@app.route('/api/patient/followup-eligibility', methods=['GET'])
def check_patient_followup_eligibility():
    user_id_str = request.headers.get('X-User-Id')
    if not user_id_str: return jsonify({"error": "Unauthorized"}), 401
    try: user_id = int(user_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID"}), 400
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "DB failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT email, phone_number, user_type FROM users WHERE id = %s", (user_id,))
            user = cur.fetchone()
            if not user or user['user_type'] != 'requester': return jsonify({"error": "Forbidden"}), 403
            email_val, phone_val = user['email'], user['phone_number']
            sql = "SELECT cf.id, cf.notes, c.name as camp_name FROM camp_follow_ups cf JOIN camps c ON cf.camp_id = c.id WHERE cf.linked_patient_user_id = %s OR cf.patient_identifier = %s OR (%s IS NOT NULL AND cf.patient_identifier = %s) ORDER BY cf.created_at DESC LIMIT 1;"
            cur.execute(sql, (user_id, email_val, phone_val, phone_val))
            eligible_fu = cur.fetchone()
            if eligible_fu:
                msg = f"Followup for camp '{eligible_fu['camp_name']}'."
                if eligible_fu['notes']: msg += f" Notes: {eligible_fu['notes']}"
                return jsonify({"eligible": True, "message": msg, "follow_up_details": row_to_dict(eligible_fu)}), 200
            else: return jsonify({"eligible": False, "message": "No followups scheduled."}), 200
    except psycopg2.Error as e: app.logger.error(f"DB error check_patient_followup_eligibility: {e}", exc_info=True); return jsonify({"error": "DB error"}), 500
    except Exception as e: app.logger.error(f"Unexpected error check_patient_followup_eligibility: {e}", exc_info=True); return jsonify({"error": "Unexpected error"}), 500
    finally:
        if conn and not conn.closed: 
            try: conn.close()
            except Exception as e_close: app.logger.error(f"[check_patient_followup_eligibility] Error closing connection: {e_close}", exc_info=True)

@app.route('/')
def index():
    return "GoMedCamp Backend is running!"

with app.app_context():
    if not app.logger.handlers and not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(name)s: %(message)s')
        app.logger.setLevel(logging.INFO)
    app.logger.info("Attempting to initialize database tables on application startup...")
    if not create_tables():
        app.logger.critical("############################################################")
        app.logger.critical("!! DATABASE TABLES FAILED TO CREATE OR INITIALIZE !!")
        app.logger.critical("The application will start, but WILL LIKELY NOT FUNCTION until database issues are resolved.")
        app.logger.critical("Please check logs for specific database errors.")
        app.logger.critical("############################################################")
    else:
        app.logger.info("Database tables initialization completed successfully on application startup.")

if __name__ == '__main__':
    log_level = logging.DEBUG if os.getenv('FLASK_DEBUG') == '1' or app.debug else logging.INFO
    if not logging.getLogger().hasHandlers() and not app.logger.handlers: 
        logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s: %(name)s: %(message)s [in %(pathname)s:%(lineno)d]')
    app.logger.setLevel(log_level)
    app.logger.info("Application starting up (direct execution)...")
    if not check_db_env_vars():
        app.logger.critical("Application may not start properly due to missing DB env vars.")
    
    app.logger.info(f"Expecting indicator JSONs from: {BASE_JSON_DIR}")
    if BASE_JSON_DIR.lower().startswith(('http://', 'https://')):
        app.logger.info(f"Indicator JSONs source is a URL. Will be downloaded on demand.")
    elif os.path.isfile(BASE_JSON_DIR) and BASE_JSON_DIR.lower().endswith('.zip'):
        if not os.path.exists(BASE_JSON_DIR):
             app.logger.warning(f"APP_BASE_JSON_DIR (ZIP file '{os.path.abspath(BASE_JSON_DIR)}') not found.")
    elif os.path.isdir(BASE_JSON_DIR):
        if not os.path.exists(BASE_JSON_DIR):
            app.logger.warning(f"APP_BASE_JSON_DIR (directory '{os.path.abspath(BASE_JSON_DIR)}') not found.")
    else:
        app.logger.warning(f"APP_BASE_JSON_DIR ('{BASE_JSON_DIR}') is not a recognized local path or URL type.")

    app.logger.info(f"Expecting geographic points CSV from: {CSV_POINTS_PATH}")
    if CSV_POINTS_PATH.lower().startswith(('http://', 'https://')):
        app.logger.info(f"Geographic points CSV source is a URL. Pandas will attempt to read it directly.")
    elif not os.path.isfile(CSV_POINTS_PATH): # Check only if it's not a URL
        app.logger.warning(f"APP_CSV_POINTS_PATH (local file '{os.path.abspath(CSV_POINTS_PATH)}') not found.")

    app.logger.info(f"Hugging Face Chatbot Model ID (Local): {HF_CHATBOT_MODEL_ID}")
    initialize_local_chatbot_model() 
    if LOCAL_CHATBOT_MODEL_INIT_STATUS == "success": app.logger.info(f"Local chatbot model '{HF_CHATBOT_MODEL_ID}' ready.")
    else: app.logger.error(f"Local chatbot model '{HF_CHATBOT_MODEL_ID}' FAILED to initialize.")

    app.logger.info(f"Hugging Face Translation Model ID (Local): {HF_TRANSLATION_MODEL_ID}")
    initialize_local_translation_model() 
    if LOCAL_TRANSLATION_MODEL_INIT_STATUS == "success": app.logger.info(f"Local translation model '{HF_TRANSLATION_MODEL_ID}' ready.")
    else: app.logger.error(f"Local translation model '{HF_TRANSLATION_MODEL_ID}' FAILED to initialize.")
    
    port = int(os.environ.get("PORT", 5001)) 
    app.logger.info(f"Starting Flask server on host 0.0.0.0 port {port}. Debug mode: {app.debug}")
    app.run(host='0.0.0.0', port=port, debug=app.debug)
