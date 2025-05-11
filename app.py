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
# import requests # No longer needed if Lingva is fully replaced

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
DB_NAME='railway'
DB_USER='postgres'
DB_PASSWORD='YugfsjGfbcGvNRVpGtMFlJvsUtCwocba'
DB_HOST='shuttle.proxy.rlwy.net'
DB_PORT='21426'

# --- Configuration for Heatmap Data ---
BASE_JSON_DIR = os.getenv('APP_BASE_JSON_DIR', 'json/')
CSV_POINTS_PATH = "https://github.com/Ved-Dixit/GoMedCamp/releases/download/coordinates/Ind_adm2_Points.csv"

# --- Configuration for AI Chatbot and Translation ---
HF_API_TOKEN = os.getenv('HF_API_TOKEN') # Kept for completeness, not used by local models
HF_CHATBOT_MODEL_ID = os.getenv('HF_CHATBOT_MODEL_ID', "gpt2") # Model for local chatbot
HF_TRANSLATION_MODEL_ID = os.getenv('HF_TRANSLATION_MODEL_ID', "facebook/nllb-200-distilled-600M") # Model for local translation
# LINGVA_API_URL = os.getenv('LINGVA_API_URL', "https://lingva.thedaviddelta.com") # Removed

# --- Global variables for local CHATBOT model ---
local_chatbot_pipeline = None
local_chatbot_tokenizer = None # Renamed from local_tokenizer to be specific
LOCAL_CHATBOT_MODEL_INIT_STATUS = "pending"

# --- Global variables for local TRANSLATION model ---
local_translation_pipeline = None
# local_translation_tokenizer = None # Pipeline usually includes tokenizer
LOCAL_TRANSLATION_MODEL_INIT_STATUS = "pending"

# --- NLLB Language Code Mapping (extend as needed) ---
# Find codes here: https://huggingface.co/facebook/nllb-200-distilled-600M#languages-covered
LANGUAGE_CODE_MAP_NLLB = {
    "en": "eng_Latn",  # English
    "hi": "hin_Deva",  # Hindi
    "es": "spa_Latn",  # Spanish
    "fr": "fra_Latn",  # French
    "de": "deu_Latn",  # German
    "ar": "ara_Arab",  # Arabic
    "bn": "ben_Beng",  # Bengali
    "gu": "guj_Gujr",  # Gujarati
    "kn": "kan_Knda",  # Kannada
    "ml": "mal_Mlym",  # Malayalam
    "mr": "mar_Deva",  # Marathi
    "pa": "pan_Guru",  # Punjabi
    "ta": "tam_Taml",  # Tamil
    "te": "tel_Telu",  # Telugu
    "ur": "urd_Arab",  # Urdu
    # Add more languages as needed
}

# --- Set of internal error messages from the bot that should not be translated ---
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
        LOCAL_CHATBOT_MODEL_INIT_STATUS = "failed"
        return

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
        LOCAL_TRANSLATION_MODEL_INIT_STATUS = "failed"
        return
    
    try:
        app.logger.info(f"Attempting to initialize local TRANSLATION pipeline for model: {HF_TRANSLATION_MODEL_ID}...")
        # For NLLB, the task is "translation" but we specify src_lang and tgt_lang during the call
        # The pipeline can be initialized with the model name directly.
        # The tokenizer is loaded automatically by the pipeline.
        local_translation_pipeline = pipeline("translation", model=HF_TRANSLATION_MODEL_ID)
        LOCAL_TRANSLATION_MODEL_INIT_STATUS = "success"
        app.logger.info(f"Local TRANSLATION pipeline for {HF_TRANSLATION_MODEL_ID} initialized successfully.")
    except Exception as e:
        app.logger.error(f"Failed to initialize local TRANSLATION pipeline for {HF_TRANSLATION_MODEL_ID}: {e}", exc_info=True)
        LOCAL_TRANSLATION_MODEL_INIT_STATUS = "failed"


def check_db_env_vars():
    """Checks if essential database environment variables are set."""
    required_db_vars = {'DB_NAME': DB_NAME, 'DB_USER': DB_USER, 'DB_HOST': DB_HOST, 'DB_PORT': DB_PORT}
    missing_vars = [var_name for var_name, var_value in required_db_vars.items() if not var_value]
    if missing_vars:
        app.logger.critical(f"Missing critical database environment variables: {', '.join(missing_vars)}. Database operations will likely fail.")
        return False
    app.logger.info("All critical database environment variables appear to be set.")
    return True

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    if not all([DB_NAME, DB_USER, DB_HOST, DB_PORT]):
        app.logger.error("Cannot attempt database connection due to missing DB configuration variables.")
        return None
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except psycopg2.Error as e:
        app.logger.error(f"Error connecting to PostgreSQL database: {e}")
        return None

def create_tables():
    """Creates tables if they don't exist and ensures specific columns are nullable."""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                # Users Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(80) UNIQUE NOT NULL,
                        email VARCHAR(120) UNIQUE NOT NULL,
                        phone_number VARCHAR(20) UNIQUE NOT NULL,
                        password_hash VARCHAR(128) NOT NULL,
                        user_type VARCHAR(50) NOT NULL, 
                        address TEXT,
                        latitude DECIMAL(10, 8),
                        longitude DECIMAL(11, 8),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                app.logger.info("Users table checked/created successfully.")

                # Camps Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS camps (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        location_latitude DECIMAL(10, 8),
                        location_longitude DECIMAL(11, 8),
                        location_address TEXT,
                        start_date DATE NOT NULL,
                        end_date DATE NOT NULL,
                        organizer_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                        description TEXT,
                        status VARCHAR(50) DEFAULT 'planned',
                        target_patients INTEGER DEFAULT 0,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                app.logger.info("Camps table checked/created successfully.")

                # Patients Table - Definition
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS patients (
                        id SERIAL PRIMARY KEY,
                        camp_id INTEGER REFERENCES camps(id) ON DELETE CASCADE, 
                        user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                        name VARCHAR(150) NOT NULL,
                        email VARCHAR(150) NOT NULL,
                        phone_number VARCHAR(20),
                        disease_detected TEXT,
                        area_location VARCHAR(255),
                        organizer_notes TEXT,
                        created_by_organizer_id INTEGER REFERENCES users(id) ON DELETE SET NULL, 
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Explicitly ALTER TABLE to ensure columns are nullable if table already existed with NOT NULL constraints
                try:
                    cur.execute("ALTER TABLE patients ALTER COLUMN camp_id DROP NOT NULL;")
                    app.logger.info("Ensured 'camp_id' column in 'patients' table is nullable.")
                except psycopg2.Error as alter_err_camp_id:
                    app.logger.warning(f"Could not alter 'camp_id' to DROP NOT NULL (may be already nullable or other issue): {alter_err_camp_id}")

                try:
                    cur.execute("ALTER TABLE patients ALTER COLUMN created_by_organizer_id DROP NOT NULL;")
                    app.logger.info("Ensured 'created_by_organizer_id' column in 'patients' table is nullable.")
                except psycopg2.Error as alter_err_organizer_id:
                    app.logger.warning(f"Could not alter 'created_by_organizer_id' to DROP NOT NULL (may be already nullable or other issue): {alter_err_organizer_id}")

                cur.execute("CREATE INDEX IF NOT EXISTS idx_patients_email ON patients (email);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_patients_user_id ON patients (user_id);")
                app.logger.info("Patients table checked/created/altered successfully.")


                # Camp Registrations Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS camp_registrations (
                        id SERIAL PRIMARY KEY,
                        camp_id INTEGER REFERENCES camps(id) ON DELETE CASCADE NOT NULL,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL,
                        registration_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        status VARCHAR(50) DEFAULT 'pending',
                        notes TEXT,
                        UNIQUE (camp_id, user_id)
                    );
                """)
                app.logger.info("Camp registrations table checked/created successfully.")

                # Connection Requests Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS connection_requests (
                        id SERIAL PRIMARY KEY,
                        camp_id INTEGER REFERENCES camps(id) ON DELETE CASCADE NOT NULL,
                        organizer_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL,
                        local_org_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL,
                        status VARCHAR(50) DEFAULT 'pending',
                        requested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        responded_at TIMESTAMP WITH TIME ZONE,
                        UNIQUE (camp_id, organizer_id, local_org_id)
                    );
                """)
                app.logger.info("Connection requests table checked/created successfully.")

                # Chat Messages Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id SERIAL PRIMARY KEY,
                        connection_request_id INTEGER REFERENCES connection_requests(id) ON DELETE CASCADE NOT NULL,
                        sender_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL,
                        message_text TEXT NOT NULL,
                        sent_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        read_at TIMESTAMP WITH TIME ZONE
                    );
                """)
                app.logger.info("Chat messages table checked/created successfully.")

                # Camp Staff Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS camp_staff (
                        id SERIAL PRIMARY KEY,
                        camp_id INTEGER REFERENCES camps(id) ON DELETE CASCADE NOT NULL,
                        name VARCHAR(255) NOT NULL,
                        role VARCHAR(255),
                        origin TEXT,
                        contact VARCHAR(100),
                        notes TEXT
                    );
                """)
                app.logger.info("Camp staff table checked/created successfully.")

                # Camp Medicines Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS camp_medicines (
                        id SERIAL PRIMARY KEY,
                        camp_id INTEGER REFERENCES camps(id) ON DELETE CASCADE NOT NULL,
                        name VARCHAR(255) NOT NULL,
                        unit VARCHAR(50),
                        quantity_per_patient DECIMAL(10,2),
                        notes TEXT
                    );
                """)
                app.logger.info("Camp medicines table checked/created successfully.")

                # Camp Equipment Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS camp_equipment (
                        id SERIAL PRIMARY KEY,
                        camp_id INTEGER REFERENCES camps(id) ON DELETE CASCADE NOT NULL,
                        name VARCHAR(255) NOT NULL,
                        quantity INTEGER,
                        notes TEXT
                    );
                """)
                app.logger.info("Camp equipment table checked/created successfully.")

                # Patient Feedback Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS patient_feedback (
                        id SERIAL PRIMARY KEY,
                        patient_user_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL,
                        patient_record_id INTEGER REFERENCES patients(id) ON DELETE SET NULL,
                        feedback_text TEXT NOT NULL,
                        rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                        language VARCHAR(10),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                app.logger.info("Patient feedback table checked/created successfully.")

                # Patient Chat Messages Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS patient_chat_messages (
                        id SERIAL PRIMARY KEY,
                        patient_user_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL,
                        patient_record_id INTEGER REFERENCES patients(id) ON DELETE SET NULL,
                        message_text TEXT NOT NULL,
                        sender_type VARCHAR(10) NOT NULL, -- 'user' or 'bot'
                        language VARCHAR(10),
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                app.logger.info("Patient chat messages table checked/created successfully.")

                # Camp Reviews Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS camp_reviews (
                        id SERIAL PRIMARY KEY,
                        camp_id INTEGER REFERENCES camps(id) ON DELETE CASCADE NOT NULL,
                        patient_user_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL,
                        rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                        comment TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                app.logger.info("Camp reviews table checked/created successfully.")

                # Camp Follow-ups Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS camp_follow_ups (
                        id SERIAL PRIMARY KEY,
                        camp_id INTEGER REFERENCES camps(id) ON DELETE CASCADE NOT NULL,
                        patient_identifier TEXT NOT NULL,
                        notes TEXT,
                        added_by_organizer_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                        linked_patient_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                app.logger.info("Camp follow-ups table checked/created successfully.")
                
                conn.commit()
        except psycopg2.Error as e:
            app.logger.error(f"Error during table creation/alteration: {e}")
            conn.rollback()
        finally:
            conn.close()
    else:
        app.logger.error("Could not create/alter tables due to failed database connection.")

# --- Heatmap Helper Functions (Existing - Unchanged) ---
def standardize_name(name):
    if not isinstance(name, str):
        try: name = str(name)
        except Exception as e:
            app.logger.warning(f"Could not convert value '{name}' of type {type(name)} to string in standardize_name: {e}")
            return "" 
    processed_name = name.lower().replace('_', ' ').replace('-', ' ')
    return ' '.join(processed_name.split())

def load_indicator_data_for_state(state_name_url_case, indicator_id_req):
    state_json_path = os.path.join(BASE_JSON_DIR, state_name_url_case)
    if not os.path.isdir(state_json_path):
        app.logger.warning(f"State JSON directory not found: {state_json_path}")
        return None, f"Indicator ID {indicator_id_req}" 
    all_district_data = []
    full_indicator_name_text = f"Indicator ID {indicator_id_req}" 
    for filename in os.listdir(state_json_path):
        if filename.endswith('.json'):
            district_name_from_file = standardize_name(filename.replace('.json', ''))
            if not district_name_from_file: continue
            filepath = os.path.join(state_json_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
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
            except Exception as e: app.logger.error(f"Error processing {filename}: {e}")
    if not all_district_data: return None, full_indicator_name_text
    df_indicators = pd.DataFrame(all_district_data)
    if df_indicators.empty: return None, full_indicator_name_text
    df_indicators.dropna(subset=['value'], inplace=True)
    if df_indicators.empty: return None, full_indicator_name_text
    if 'indicator_name_text' in df_indicators.columns and not df_indicators.empty:
        common_name = df_indicators['indicator_name_text'].mode()
        full_indicator_name_text = common_name[0] if not common_name.empty else df_indicators['indicator_name_text'].iloc[0]
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
    if not os.path.isfile(CSV_POINTS_PATH):
        app.logger.error(f"Geographic CSV file not found: {os.path.abspath(CSV_POINTS_PATH)}")
        return None, None
    df_all_geo_points = None
    encodings_to_try = ['utf-16', 'utf-8-sig', 'latin1']
    for enc in encodings_to_try:
        try:
            df_all_geo_points = pd.read_csv(CSV_POINTS_PATH, encoding=enc)
            app.logger.info(f"Successfully loaded geographic CSV: {CSV_POINTS_PATH} using '{enc}' encoding.")
            break
        except UnicodeDecodeError: app.logger.warning(f"UnicodeDecodeError with '{enc}' for {CSV_POINTS_PATH}. Trying next...")
        except Exception as e: app.logger.error(f"Error loading geographic CSV {CSV_POINTS_PATH} with '{enc}': {e}"); return None, None
    if df_all_geo_points is None: app.logger.error(f"Failed to load geographic CSV {CSV_POINTS_PATH} with tried encodings."); return None, None
    
    CSV_STATE_NAME_COL = _get_column_name(df_all_geo_points.columns, ENV_CSV_STATE_COL, ['State_Name', 'state_name', 'State', 'state', 'NAME_1', 'ADM1_EN', 'ST_NM'], "state name", CSV_POINTS_PATH)
    if not CSV_STATE_NAME_COL: return None, None
    df_all_geo_points['state_standardized_csv'] = df_all_geo_points[CSV_STATE_NAME_COL].astype(str).apply(standardize_name)
    df_state_geo_points = df_all_geo_points[df_all_geo_points['state_standardized_csv'] == state_name_standardized_filter].copy()
    if df_state_geo_points.empty: app.logger.warning(f"No geographic data for state '{state_name_standardized_filter}' in {CSV_POINTS_PATH}."); return None, None

    CSV_DISTRICT_NAME_COL = _get_column_name(df_state_geo_points.columns, ENV_CSV_DISTRICT_COL, ['District_Name', 'district_name', 'District', 'district', 'NAME_2', 'ADM2_EN', 'dt_name', 'Dist_Name'], "district name", CSV_POINTS_PATH)
    if not CSV_DISTRICT_NAME_COL: return None, None
    CSV_LAT_COL = _get_column_name(df_state_geo_points.columns, ENV_CSV_LAT_COL, ['Latitude', 'latitude', 'Lat', 'lat', 'Y', 'y_coord'], "latitude", CSV_POINTS_PATH)
    if not CSV_LAT_COL: return None, None
    CSV_LON_COL = _get_column_name(df_state_geo_points.columns, ENV_CSV_LON_COL, ['Longitude', 'longitude', 'Lon', 'lon', 'X', 'x_coord'], "longitude", CSV_POINTS_PATH)
    if not CSV_LON_COL: return None, None

    df_state_geo_points[CSV_LAT_COL] = pd.to_numeric(df_state_geo_points[CSV_LAT_COL], errors='coerce')
    df_state_geo_points[CSV_LON_COL] = pd.to_numeric(df_state_geo_points[CSV_LON_COL], errors='coerce')
    df_state_geo_points.dropna(subset=[CSV_LAT_COL, CSV_LON_COL], inplace=True)
    if df_state_geo_points.empty: app.logger.warning(f"No valid lat/lon data for state '{state_name_standardized_filter}'."); return None, None
    
    try:
        geometry = gpd.points_from_xy(df_state_geo_points[CSV_LON_COL], df_state_geo_points[CSV_LAT_COL])
        cols_to_keep = [col for col in [CSV_DISTRICT_NAME_COL, CSV_STATE_NAME_COL] if col in df_state_geo_points.columns]
        gdf_districts = gpd.GeoDataFrame(df_state_geo_points[cols_to_keep], geometry=geometry, crs="EPSG:4326")
    except Exception as e: app.logger.error(f"Error creating GeoDataFrame for state '{state_name_standardized_filter}': {e}"); return None, CSV_DISTRICT_NAME_COL
    gdf_districts['district_standardized_geo'] = gdf_districts[CSV_DISTRICT_NAME_COL].astype(str).apply(standardize_name)
    gdf_districts = gdf_districts[gdf_districts['district_standardized_geo'] != ""]
    if gdf_districts.empty: app.logger.warning(f"GeoDataFrame for state '{state_name_standardized_filter}' empty after removing empty standardized district names."); return None, CSV_DISTRICT_NAME_COL
    return gdf_districts, CSV_DISTRICT_NAME_COL

# --- Haversine Distance (Existing - Unchanged) ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# --- Helper to convert DB row to dict (Existing - Unchanged) ---
def row_to_dict(row_raw):
    if not row_raw:
        return None
    row = dict(row_raw)
    for key, value in row.items():
        if isinstance(value, datetime):
            row[key] = value.isoformat()
        elif isinstance(value, pd.Timestamp):
             row[key] = value.isoformat()
        elif isinstance(value, (float, int, str, bool)) or value is None:
            pass
        else:
            try:
                row[key] = str(value)
            except:
                row[key] = None
    return row

# --- API Endpoints ---
@app.route('/api/signup', methods=['POST'])
def signup():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    username = data.get('username') # This is 'name' from frontend form
    email = data.get('email')
    phone_number = data.get('phone_number') # This is 'phoneNumber' from frontend form
    password = data.get('password')
    user_type = data.get('userType')
    address = data.get('address')

    if not all([username, email, phone_number, password, user_type]):
        return jsonify({"error": "Missing data. Username, email, phone, password, and userType are required."}), 400
    
    valid_user_types = ['organizer', 'requester', 'local_organisation']
    if user_type not in valid_user_types:
        return jsonify({"error": f"Invalid user type. Must be one of: {', '.join(valid_user_types)}"}), 400
    
    if user_type == 'local_organisation' and not address:
        return jsonify({"error": "Address is required for Local Organisation user type."}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed."}), 500
    
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT id FROM users WHERE username = %s OR email = %s OR phone_number = %s",
                        (username, email, phone_number))
            if cur.fetchone():
                return jsonify({"error": "User with this username, email, or phone number already exists."}), 409
            
            # Insert into users table
            sql_user_insert = """
                INSERT INTO users (username, email, phone_number, password_hash, user_type, address) 
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id, username, email, user_type, address, created_at;
            """
            cur.execute(sql_user_insert, (username, email, phone_number, hashed_password, user_type, address if user_type == 'local_organisation' else None))
            new_user_raw = cur.fetchone()

            if new_user_raw:
                new_user_id = new_user_raw['id']
                
                # MODIFICATION: If the new user is a 'requester', create a basic patient profile
                if user_type == 'requester':
                    try:
                        app.logger.info(f"[signup] New user is a requester (User ID: {new_user_id}). Creating a basic patient record.")
                        cur.execute("""
                            INSERT INTO patients (user_id, name, email, phone_number, camp_id, created_by_organizer_id)
                            VALUES (%s, %s, %s, %s, NULL, NULL)
                        """, (new_user_id, username, email, phone_number))
                        app.logger.info(f"[signup] Basic patient record queued for creation for User ID: {new_user_id}")
                    except psycopg2.Error as pe:
                        app.logger.error(f"[signup] Error queuing basic patient record for User ID {new_user_id}: {pe}")
                        raise 
                
                conn.commit() 
                
                user_data_to_return = row_to_dict(new_user_raw)
                if 'user_type' in user_data_to_return: 
                    user_data_to_return['userType'] = user_data_to_return.pop('user_type')
                return jsonify({"message": "User created successfully! A basic patient profile was also created if you signed up as a patient.", "user": user_data_to_return}), 201
            else:
                app.logger.error("[signup] User creation seemed successful but no user data was returned from users table.")
                conn.rollback() 
                return jsonify({"error": "User creation failed unexpectedly."}), 500

    except psycopg2.Error as e:
        conn.rollback()
        app.logger.error(f"[signup] Database error during signup: {e}")
        if hasattr(e, 'pgcode') and e.pgcode == '23505': 
             return jsonify({"error": "A user with this username, email, or phone number already exists."}), 409
        return jsonify({"error": "An error occurred during registration. Please try again."}), 500
    finally:
        if conn: conn.close()

@app.route('/api/login', methods=['POST'])
def login():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed."}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT id, username, email, password_hash, user_type, address, created_at FROM users WHERE email = %s", (email,))
            user_raw = cur.fetchone()
            
            if user_raw and bcrypt.check_password_hash(user_raw['password_hash'], password):
                user_info = row_to_dict(user_raw)
                if 'user_type' in user_info:
                    user_info['userType'] = user_info.pop('user_type')
                
                if 'password_hash' in user_info:
                    del user_info['password_hash']

                return jsonify({"message": "Login successful!", "user": user_info}), 200
            else:
                return jsonify({"error": "Invalid email or password."}), 401
    except psycopg2.Error as e:
        app.logger.error(f"Database error during login: {e}")
        return jsonify({"error": "An error occurred during login."}), 500
    finally:
        if conn: conn.close()

@app.route('/api/heatmap_data', methods=['GET'])
def get_heatmap_data():
    state_name_req = request.args.get('state')
    indicator_id_req = request.args.get('indicator_id')
    if not state_name_req or not indicator_id_req:
        return jsonify({"error": "Missing 'state' or 'indicator_id' query parameter"}), 400
    state_name_for_json_path = standardize_name(state_name_req).replace(' ', '_')
    state_name_standardized_filter = standardize_name(state_name_req)
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

@app.route('/api/organizer/camps', methods=['POST'])
def create_camp_endpoint():
    organizer_user_id_str = request.headers.get('X-User-Id')
    if not organizer_user_id_str:
        app.logger.warning("Creating camp without organizer_id (X-User-Id header missing).")
        return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try:
        organizer_user_id = int(organizer_user_id_str)
    except ValueError:
        app.logger.error(f"Invalid X-User-Id format: {organizer_user_id_str}")
        return jsonify({"error": "Invalid user identifier format."}), 400

    if not request.is_json: return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    required_fields = ['name', 'location_latitude', 'location_longitude', 'start_date', 'end_date']
    if not all(field in data and data[field] is not None for field in required_fields):
        missing = [field for field in required_fields if field not in data or data[field] is None]
        return jsonify({"error": f"Missing required camp data for fields: {', '.join(missing)}"}), 400
    
    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (organizer_user_id,))
            user = cur.fetchone()
            if not user or user['user_type'] != 'organizer':
                return jsonify({"error": "Forbidden: Only organizers can create camps."}), 403

            cur.execute(
                """
                INSERT INTO camps (name, description, location_latitude, location_longitude, location_address, start_date, end_date, organizer_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s) 
                RETURNING id, name, description, location_latitude, location_longitude, location_address, start_date, end_date, organizer_id, created_at, status, target_patients;
                """,
                (data['name'], data.get('description'), data['location_latitude'], data['location_longitude'], 
                 data.get('location_address'), data['start_date'], data['end_date'], organizer_user_id)
            )
            new_camp_raw = cur.fetchone()
            conn.commit()
            if new_camp_raw:
                new_camp = row_to_dict(new_camp_raw)
                for key in ['location_latitude', 'location_longitude']:
                    if new_camp.get(key) is not None: new_camp[key] = float(new_camp[key])
                return jsonify({"message": "Camp created successfully", "camp": new_camp}), 201
            else: return jsonify({"error": "Failed to create camp, no data returned."}), 500
    except psycopg2.Error as e:
        conn.rollback()
        app.logger.error(f"Error creating camp: {e}")
        return jsonify({"error": "Failed to create camp"}), 500
    finally:
        if conn: conn.close()

@app.route('/api/organizer/camps', methods=['GET'])
def get_organizer_camps_endpoint():
    organizer_user_id_str = request.headers.get('X-User-Id')
    if not organizer_user_id_str:
        return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try:
        organizer_user_id = int(organizer_user_id_str)
    except ValueError:
        return jsonify({"error": "Invalid User ID format."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (organizer_user_id,))
            user = cur.fetchone()
            if not user or user['user_type'] != 'organizer':
                return jsonify({"error": "Forbidden: Only organizers can view this list of camps."}), 403

            cur.execute("""
                SELECT id, name, description, location_latitude, location_longitude, 
                       location_address, start_date, end_date, organizer_id, status, target_patients
                FROM camps 
                WHERE organizer_id = %s
                ORDER BY start_date DESC
            """, (organizer_user_id,))
            camps_raw = cur.fetchall()
            camps = []
            for row_raw in camps_raw:
                camp = row_to_dict(row_raw)
                camp['lat'] = float(camp.pop('location_latitude')) if camp.get('location_latitude') is not None else None
                camp['lng'] = float(camp.pop('location_longitude')) if camp.get('location_longitude') is not None else None
                camps.append(camp)
            return jsonify(camps), 200
    except psycopg2.Error as e:
        app.logger.error(f"Error fetching camps for organizer {organizer_user_id}: {e}")
        return jsonify({"error": "Failed to fetch camps"}), 500
    finally:
        if conn: conn.close()

@app.route('/api/organizer/camps/<int:camp_id>', methods=['GET'])
def get_camp_details_endpoint(camp_id):
    requesting_user_id_str = request.headers.get('X-User-Id')
    if not requesting_user_id_str:
        return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try:
        requesting_user_id = int(requesting_user_id_str)
    except ValueError:
        return jsonify({"error": "Invalid User ID format."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (requesting_user_id,))
            user = cur.fetchone()
            if not user or user['user_type'] != 'organizer':
                return jsonify({"error": "Forbidden: Only organizers can access camp details."}), 403

            cur.execute("""
                SELECT id, name, description, location_latitude, location_longitude, 
                       location_address, start_date, end_date, organizer_id, status, 
                       target_patients, created_at, updated_at
                FROM camps WHERE id = %s
            """, (camp_id,))
            camp_raw = cur.fetchone()
            
            if not camp_raw: 
                return jsonify({"message": "Camp not found."}), 404
            
            if camp_raw['organizer_id'] != requesting_user_id:
                app.logger.warning(f"Forbidden access attempt: User {requesting_user_id} tried to access camp {camp_id} owned by {camp_raw['organizer_id']}.")
                return jsonify({"error": "Forbidden: You do not have permission to access this camp's details."}), 403
            
            camp = row_to_dict(camp_raw)
            if camp.get('location_latitude') is not None: camp['location_latitude'] = float(camp['location_latitude'])
            if camp.get('location_longitude') is not None: camp['location_longitude'] = float(camp['location_longitude'])
            return jsonify(camp), 200
    except psycopg2.Error as e:
        app.logger.error(f"Error fetching camp details for camp ID {camp_id}, user {requesting_user_id}: {e}")
        return jsonify({"error": "Failed to fetch camp details."}), 500
    finally:
        if conn: conn.close()

@app.route('/api/organizer/camps/<int:camp_id>', methods=['DELETE'])
def delete_camp_endpoint(camp_id):
    organizer_user_id_str = request.headers.get('X-User-Id')
    if not organizer_user_id_str:
        return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try:
        requesting_organizer_id = int(organizer_user_id_str)
    except ValueError:
        return jsonify({"error": "Invalid User ID format."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Verify user is an organizer
            cur.execute("SELECT user_type FROM users WHERE id = %s", (requesting_organizer_id,))
            user_check = cur.fetchone()
            if not user_check or user_check['user_type'] != 'organizer':
                return jsonify({"error": "Forbidden: Only organizers can delete camps."}), 403

            cur.execute("SELECT organizer_id FROM camps WHERE id = %s", (camp_id,))
            camp = cur.fetchone()
            if not camp: 
                return jsonify({"error": "Camp not found."}), 404
            if camp['organizer_id'] != requesting_organizer_id:
                return jsonify({"error": "Forbidden: You are not the organizer of this camp."}), 403
            
            cur.execute("DELETE FROM camps WHERE id = %s", (camp_id,))
            if cur.rowcount == 0: 
                conn.rollback() 
                return jsonify({"error": "Camp not found or failed to delete."}), 404 
            conn.commit()
            return jsonify({"message": f"Camp with ID {camp_id} deleted successfully."}), 200
    except psycopg2.Error as e:
        conn.rollback()
        app.logger.error(f"Error deleting camp {camp_id} by user {requesting_organizer_id}: {e}")
        return jsonify({"error": "Failed to delete camp."}), 500
    finally:
        if conn: conn.close()

@app.route('/api/organizer/camp/<int:camp_id>/resources', methods=['GET'])
def get_camp_resources(camp_id):
    requesting_user_id_str = request.headers.get('X-User-Id')
    if not requesting_user_id_str:
        return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try:
        requesting_user_id = int(requesting_user_id_str)
    except ValueError:
        return jsonify({"error": "Invalid User ID format."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (requesting_user_id,))
            user_check = cur.fetchone()
            if not user_check or user_check['user_type'] != 'organizer':
                return jsonify({"error": "Forbidden: Only organizers can access camp resources."}), 403

            cur.execute("SELECT target_patients, organizer_id FROM camps WHERE id = %s", (camp_id,))
            camp_info = cur.fetchone()
            
            if not camp_info:
                return jsonify({"error": "Camp not found"}), 404
            
            if camp_info['organizer_id'] != requesting_user_id:
                app.logger.warning(f"Forbidden access attempt: User {requesting_user_id} tried to access resources for camp {camp_id} owned by {camp_info['organizer_id']}.")
                return jsonify({"error": "Forbidden: You do not have permission to access resources for this camp."}), 403

            target_patients = camp_info['target_patients']

            cur.execute("SELECT id, name, role, origin, contact, notes FROM camp_staff WHERE camp_id = %s", (camp_id,))
            staff_list = [row_to_dict(row) for row in cur.fetchall()]

            cur.execute("SELECT id, name, unit, quantity_per_patient, notes FROM camp_medicines WHERE camp_id = %s", (camp_id,))
            medicine_list_raw = cur.fetchall()
            medicine_list = []
            for med_raw in medicine_list_raw:
                med = row_to_dict(med_raw)
                if med.get('quantity_per_patient') is not None:
                    med['quantity_per_patient'] = float(med['quantity_per_patient'])
                medicine_list.append(med)

            cur.execute("SELECT id, name, quantity, notes FROM camp_equipment WHERE camp_id = %s", (camp_id,))
            equipment_list = [row_to_dict(row) for row in cur.fetchall()]

            return jsonify({
                "targetPatients": target_patients,
                "staffList": staff_list,
                "medicineList": medicine_list,
                "equipmentList": equipment_list
            }), 200
    except psycopg2.Error as e:
        app.logger.error(f"Error fetching resources for camp {camp_id}, user {requesting_user_id}: {e}")
        return jsonify({"error": "Failed to fetch camp resources"}), 500
    finally:
        if conn: conn.close()

@app.route('/api/organizer/camp/<int:camp_id>/resources', methods=['POST'])
def save_camp_resources(camp_id):
    organizer_user_id_str = request.headers.get('X-User-Id')
    if not organizer_user_id_str:
        return jsonify({"error": "Unauthorized: Organizer ID missing."}), 401
    try:
        requesting_organizer_id = int(organizer_user_id_str)
    except ValueError:
        return jsonify({"error": "Invalid Organizer ID format."}), 400

    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    data = request.get_json()
    target_patients = data.get('targetPatients')
    staff_list = data.get('staffList', [])
    medicine_list = data.get('medicineList', [])
    equipment_list = data.get('equipmentList', [])

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (requesting_organizer_id,))
            user_check = cur.fetchone()
            if not user_check or user_check['user_type'] != 'organizer':
                return jsonify({"error": "Forbidden: Only organizers can save camp resources."}), 403
                
            cur.execute("SELECT organizer_id FROM camps WHERE id = %s", (camp_id,))
            camp_owner = cur.fetchone()
            if not camp_owner:
                return jsonify({"error": "Camp not found."}), 404
            if camp_owner['organizer_id'] != requesting_organizer_id:
                app.logger.warning(f"Unauthorized attempt to save resources for camp {camp_id} by user {requesting_organizer_id}. Camp owner: {camp_owner['organizer_id']}")
                return jsonify({"error": "Forbidden: You are not the organizer of this camp."}), 403

            if target_patients is not None:
                cur.execute("UPDATE camps SET target_patients = %s WHERE id = %s", (target_patients, camp_id))

            cur.execute("DELETE FROM camp_staff WHERE camp_id = %s", (camp_id,))
            for staff in staff_list:
                cur.execute(
                    "INSERT INTO camp_staff (camp_id, name, role, origin, contact, notes) VALUES (%s, %s, %s, %s, %s, %s)",
                    (camp_id, staff.get('name'), staff.get('role'), staff.get('origin'), staff.get('contact'), staff.get('notes'))
                )
            
            cur.execute("DELETE FROM camp_medicines WHERE camp_id = %s", (camp_id,))
            for med in medicine_list:
                cur.execute(
                    "INSERT INTO camp_medicines (camp_id, name, unit, quantity_per_patient, notes) VALUES (%s, %s, %s, %s, %s)",
                    (camp_id, med.get('name'), med.get('unit'), med.get('quantityPerPatient'), med.get('notes'))
                )

            cur.execute("DELETE FROM camp_equipment WHERE camp_id = %s", (camp_id,))
            for equip in equipment_list:
                cur.execute(
                    "INSERT INTO camp_equipment (camp_id, name, quantity, notes) VALUES (%s, %s, %s, %s)",
                    (camp_id, equip.get('name'), equip.get('quantity'), equip.get('notes'))
                )
            
            conn.commit()
            return jsonify({"message": "Camp resources saved successfully."}), 200
    except psycopg2.Error as e:
        conn.rollback()
        app.logger.error(f"Error saving resources for camp {camp_id} by user {requesting_organizer_id}: {e}")
        return jsonify({"error": "Failed to save camp resources"}), 500
    finally:
        if conn: conn.close()

@app.route('/api/organizer/camp/<int:camp_id>/patients', methods=['POST'])
def add_patient_to_camp(camp_id):
    organizer_user_id_str = request.headers.get('X-User-Id')
    app.logger.info(f"[add_patient_to_camp] Camp ID: {camp_id}, Organizer X-User-Id from header: {organizer_user_id_str}")

    if not organizer_user_id_str:
        app.logger.warning("[add_patient_to_camp] Unauthorized: Organizer ID missing from header.")
        return jsonify({"error": "Unauthorized: Organizer ID missing."}), 401
    try:
        current_organizer_id = int(organizer_user_id_str)
    except ValueError:
        app.logger.error(f"[add_patient_to_camp] Invalid Organizer ID format: {organizer_user_id_str}")
        return jsonify({"error": "Invalid Organizer ID format."}), 400

    if not request.is_json:
        app.logger.warning("[add_patient_to_camp] Missing JSON in request.")
        return jsonify({"error": "Missing JSON in request"}), 400
    
    data = request.get_json()
    app.logger.info(f"[add_patient_to_camp] Received patient data: {data}")

    patient_name = data.get('name')
    patient_email = data.get('email')

    if not patient_name or not patient_email: 
        app.logger.warning(f"[add_patient_to_camp] Missing required fields. Name: '{patient_name}', Email: '{patient_email}'")
        return jsonify({"error": "Missing required fields: name, email"}), 400

    conn = get_db_connection()
    if not conn:
        app.logger.error("[add_patient_to_camp] Database connection failed.")
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (current_organizer_id,))
            user_check = cur.fetchone()
            if not user_check or user_check['user_type'] != 'organizer':
                user_type_found = user_check['user_type'] if user_check else 'None (user not found)'
                app.logger.warning(f"[add_patient_to_camp] Forbidden: User {current_organizer_id} is not an organizer (type: {user_type_found}).")
                return jsonify({"error": "Forbidden: Only organizers can add patients."}), 403

            cur.execute("SELECT organizer_id FROM camps WHERE id = %s", (camp_id,))
            camp = cur.fetchone()
            if not camp:
                app.logger.warning(f"[add_patient_to_camp] Camp not found: ID {camp_id}")
                return jsonify({"error": "Camp not found"}), 404
            if camp['organizer_id'] != current_organizer_id:
                app.logger.warning(f"[add_patient_to_camp] Forbidden: User {current_organizer_id} is not the organizer of camp {camp_id} (owner: {camp['organizer_id']}).")
                return jsonify({"error": "Forbidden: You are not the organizer of this camp."}), 403

            cur.execute("SELECT id FROM patients WHERE email = %s AND camp_id = %s", (patient_email, camp_id))
            if cur.fetchone():
                app.logger.info(f"[add_patient_to_camp] Conflict: Patient with email {patient_email} already exists in camp {camp_id}.")
                return jsonify({"error": f"Patient with email {patient_email} already exists in this camp."}), 409

            cur.execute("SELECT id, user_type FROM users WHERE email = %s", (patient_email,))
            existing_user = cur.fetchone()
            patient_user_id_to_link = None
            if existing_user and existing_user['user_type'] == 'requester':
                patient_user_id_to_link = existing_user['id']
                app.logger.info(f"[add_patient_to_camp] Found existing requester user (ID: {patient_user_id_to_link}) for email {patient_email}. Will link patient record.")
            else:
                app.logger.info(f"[add_patient_to_camp] No existing requester user found for email {patient_email}. Patient record will be created with user_id as NULL initially.")
            
            sql_insert_patient = """
                INSERT INTO patients (camp_id, user_id, name, email, phone_number, disease_detected, area_location, organizer_notes, created_by_organizer_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, camp_id, user_id, name, email, phone_number, disease_detected, area_location, organizer_notes, created_by_organizer_id, created_at;
            """
            insert_params = (
                camp_id, patient_user_id_to_link, patient_name, patient_email,
                data.get('phone_number'), data.get('disease_detected'),
                data.get('area_location'), data.get('organizer_notes'),
                current_organizer_id
            )
            app.logger.info(f"[add_patient_to_camp] Executing INSERT with params: {insert_params}")
            cur.execute(sql_insert_patient, insert_params)
            
            new_patient_raw = cur.fetchone()
            if new_patient_raw:
                app.logger.info(f"[add_patient_to_camp] Patient record created (raw from RETURNING): {dict(new_patient_raw)}")
            else:
                app.logger.error("[add_patient_to_camp] INSERT executed but RETURNING clause did not yield a row. This is unexpected. Rolling back.")
                conn.rollback() 
                return jsonify({"error": "Failed to add patient due to an unexpected issue retrieving the created record."}), 500

            conn.commit()
            app.logger.info(f"[add_patient_to_camp] Commit successful for new patient ID: {new_patient_raw['id']}.")

            patient_dict = row_to_dict(new_patient_raw)
            cur.execute("SELECT name FROM camps WHERE id = %s", (patient_dict['camp_id'],))
            camp_details = cur.fetchone()
            patient_dict['camp_name'] = camp_details['name'] if camp_details else None
            patient_dict['is_registered_user'] = patient_dict['user_id'] is not None
            
            app.logger.info(f"[add_patient_to_camp] Patient added successfully. Response data: {patient_dict}")
            return jsonify({"message": "Patient added successfully", "patient": patient_dict}), 201

    except psycopg2.Error as e:
        conn.rollback()
        app.logger.error(f"[add_patient_to_camp] Database error for camp {camp_id}, organizer {current_organizer_id}, patient data {data}. Error details: {e}, SQLSTATE: {e.pgcode}")
        if hasattr(e, 'pgcode') and e.pgcode == '23505': 
             app.logger.warning(f"[add_patient_to_camp] Unique constraint violation (likely duplicate email in camp): {e}")
             return jsonify({"error": f"A record with similar details (e.g., email '{patient_email}' in this camp) might already exist."}), 409
        return jsonify({"error": "An error occurred while adding the patient due to a database issue."}), 500
    except Exception as e: 
        if conn and not conn.closed: 
            try:
                conn.rollback()
            except Exception as rb_e:
                app.logger.error(f"[add_patient_to_camp] Error during rollback: {rb_e}")
        app.logger.error(f"[add_patient_to_camp] Unexpected non-database error for camp {camp_id}, organizer {current_organizer_id}, patient data {data}. Error: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred."}), 500
    finally:
        if conn and not conn.closed: 
            conn.close()

@app.route('/api/organizer/camp/<int:camp_id>/patients', methods=['GET'])
def get_camp_patients(camp_id):
    organizer_user_id_str = request.headers.get('X-User-Id')
    if not organizer_user_id_str:
        return jsonify({"error": "Unauthorized: Organizer ID missing."}), 401
    try:
        current_organizer_id = int(organizer_user_id_str)
    except ValueError:
        return jsonify({"error": "Invalid Organizer ID format."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (current_organizer_id,))
            user_check = cur.fetchone()
            if not user_check or user_check['user_type'] != 'organizer':
                return jsonify({"error": "Forbidden: Only organizers can view patient lists."}), 403

            cur.execute("SELECT organizer_id FROM camps WHERE id = %s", (camp_id,))
            camp = cur.fetchone()
            if not camp:
                return jsonify({"error": "Camp not found"}), 404
            if camp['organizer_id'] != current_organizer_id:
                app.logger.warning(f"Forbidden access attempt: User {current_organizer_id} tried to access patients for camp {camp_id} owned by {camp['organizer_id']}.")
                return jsonify({"error": "Forbidden: You do not have permission to view patients for this camp."}), 403
            
            sql = """
                SELECT p.id, p.camp_id, c.name as camp_name, p.user_id, p.name, p.email, p.phone_number, 
                       p.disease_detected, p.area_location, p.organizer_notes, p.created_by_organizer_id, p.created_at
                FROM patients p
                JOIN camps c ON p.camp_id = c.id
                WHERE p.camp_id = %s
                ORDER BY p.name;
            """
            cur.execute(sql, (camp_id,))
            patients_raw = cur.fetchall()
            patients_list = []
            for p_raw in patients_raw:
                p_dict = row_to_dict(p_raw)
                p_dict['is_registered_user'] = p_dict['user_id'] is not None
                patients_list.append(p_dict)
            return jsonify(patients_list), 200
    except psycopg2.Error as e:
        app.logger.error(f"Database error fetching patients for camp {camp_id}, organizer {current_organizer_id}: {e}")
        return jsonify({"error": "An error occurred while fetching patients."}), 500
    finally:
        if conn: conn.close()

@app.route('/api/patient/my-details', methods=['GET'])
def get_my_patient_details():
    current_user_id_str = request.headers.get('X-User-Id')
    if not current_user_id_str:
        return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try:
        current_user_id = int(current_user_id_str)
    except ValueError:
        return jsonify({"error": "Invalid User ID format."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT email, user_type FROM users WHERE id = %s", (current_user_id,))
            user = cur.fetchone()
            if not user:
                app.logger.warning(f"/api/patient/my-details: User not found for X-User-Id {current_user_id_str}")
                return jsonify({"error": "User not found."}), 404
            
            user_email = user['email']
            
            cur.execute(
                "UPDATE patients SET user_id = %s WHERE email = %s AND user_id IS NULL AND camp_id IS NOT NULL", 
                (current_user_id, user_email)
            )
            conn.commit() 
            app.logger.info(f"/api/patient/my-details: Attempted auto-link for user {current_user_id} with email {user_email} for organizer-added records. Rows affected: {cur.rowcount}")

            sql_by_user_id = """
                SELECT p.id, p.camp_id, c.name as camp_name, p.user_id, p.name, p.email, p.phone_number,
                       p.disease_detected, p.area_location, p.organizer_notes, p.created_by_organizer_id, p.created_at
                FROM patients p
                LEFT JOIN camps c ON p.camp_id = c.id 
                WHERE p.user_id = %s
                ORDER BY p.created_at DESC;
            """
            cur.execute(sql_by_user_id, (current_user_id,))
            patient_profiles_raw = cur.fetchall()
            
            if not patient_profiles_raw: 
                app.logger.info(f"/api/patient/my-details: No patient records found by user_id {current_user_id}. Attempting fallback by email {user_email} for records with a camp_id.")
                sql_by_email = """
                    SELECT p.id, p.camp_id, c.name as camp_name, p.user_id, p.name, p.email, p.phone_number,
                           p.disease_detected, p.area_location, p.organizer_notes, p.created_by_organizer_id, p.created_at
                    FROM patients p
                    JOIN camps c ON p.camp_id = c.id 
                    WHERE p.email = %s AND p.user_id IS NULL 
                    ORDER BY p.created_at DESC;
                """
                cur.execute(sql_by_email, (user_email,))
                patient_profiles_raw = cur.fetchall()

            if not patient_profiles_raw:
                app.logger.info(f"/api/patient/my-details: No patient records found for user {current_user_id} (email: {user_email}) after all checks.")
                return jsonify({"message": "No patient records found for this user."}), 404
            
            patient_profiles_list = [row_to_dict(p_raw) for p_raw in patient_profiles_raw]
            app.logger.info(f"/api/patient/my-details: Found {len(patient_profiles_list)} record(s) for user {current_user_id}.")
            return jsonify(patient_profiles_list), 200
            
    except psycopg2.Error as e:
        if conn: conn.rollback()
        app.logger.error(f"Database error fetching patient details for user {current_user_id_str}: {e}")
        return jsonify({"error": "An error occurred while fetching patient details."}), 500
    finally:
        if conn: conn.close()

# --- Local Organisation and Chat Endpoints (Existing - Unchanged) ---
@app.route('/api/local-organisations', methods=['GET'])
def get_local_organisations():
    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT id, username, email, address, phone_number 
                FROM users 
                WHERE user_type = 'local_organisation'
            """)
            orgs_raw = cur.fetchall()
            organisations = []
            for row_raw in orgs_raw:
                org_data = row_to_dict(row_raw)
                org_data['name'] = org_data.pop('username') 
                organisations.append(org_data)
            return jsonify(organisations), 200
    except psycopg2.Error as e:
        app.logger.error(f"Error fetching local organisations: {e}")
        return jsonify({"error": "Failed to fetch local organisations"}), 500
    finally:
        if conn: conn.close()

@app.route('/api/chat/request', methods=['POST'])
def send_connection_request():
    organizer_id_str = request.headers.get('X-User-Id')
    if not organizer_id_str:
        app.logger.error("Organizer ID (X-User-Id) missing in chat request header.")
        return jsonify({"error": "Unauthorized: Organizer ID missing in request header."}), 401

    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    data = request.get_json()
    camp_id_str = data.get('campId')
    local_org_id_str = data.get('localOrgId')

    try:
        organizer_user_id = int(organizer_id_str)

        if camp_id_str is None: return jsonify({"error": "campId is required."}), 400
        camp_id = int(camp_id_str)
        
        if local_org_id_str is None: return jsonify({"error": "localOrgId is required."}), 400
        local_org_id = int(local_org_id_str)

    except (ValueError, TypeError) as e:
        app.logger.error(f"Invalid ID format in chat request: {e}. CampID: {camp_id_str}, LocalOrgID: {local_org_id_str}, OrganizerID (from header): {organizer_id_str}")
        return jsonify({"error": "Invalid ID format in request. All IDs must be integers."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (organizer_user_id,))
            user_check = cur.fetchone()
            if not user_check or user_check['user_type'] != 'organizer':
                return jsonify({"error": "Forbidden: Only organizers can send connection requests."}), 403

            cur.execute("SELECT id FROM camps WHERE id = %s AND organizer_id = %s", (camp_id, organizer_user_id)) 
            if not cur.fetchone(): 
                app.logger.warning(f"Attempt to create connection request for non-existent camp ID: {camp_id} or not owned by organizer {organizer_user_id}")
                return jsonify({"error": "Camp not found or not owned by this organizer."}), 404
            
            cur.execute("SELECT id FROM users WHERE id = %s AND user_type = 'local_organisation'", (local_org_id,))
            if not cur.fetchone(): 
                app.logger.warning(f"Attempt to create connection request for non-existent or invalid local org ID: {local_org_id}")
                return jsonify({"error": "Local organisation not found or invalid type."}), 404
            
            app.logger.info(f"Attempting to insert connection request: camp_id={camp_id}, organizer_id={organizer_user_id}, local_org_id={local_org_id}")
            cur.execute(
                """
                INSERT INTO connection_requests (camp_id, organizer_id, local_org_id)
                VALUES (%s, %s, %s) RETURNING id, status, requested_at;
                """,
                (camp_id, organizer_user_id, local_org_id)
            )
            new_request_raw = cur.fetchone()
            conn.commit()

            if new_request_raw:
                new_request = row_to_dict(new_request_raw)
                app.logger.info(f"Connection request created successfully: ID {new_request['id']}")
                return jsonify({"message": "Connection request sent successfully.", "request": new_request}), 201
            else:
                app.logger.error("Connection request insert seemed successful but no data returned.")
                return jsonify({"error": "Failed to create connection request, no data returned from DB."}), 500

    except psycopg2.IntegrityError as e: 
        conn.rollback()
        app.logger.warning(f"Integrity error (e.g., duplicate connection request) for camp_id={camp_id}, organizer_id={organizer_user_id}, local_org_id={local_org_id}: {e}")
        return jsonify({"error": "Connection request already exists or involves invalid IDs."}), 409
    except psycopg2.Error as e:
        conn.rollback()
        app.logger.error(f"Database error sending connection request for camp_id={camp_id}, organizer_id={organizer_user_id}, local_org_id={local_org_id}: {e}")
        return jsonify({"error": "Failed to send connection request due to a database error."}), 500
    finally:
        if conn: conn.close()

@app.route('/api/local-organisation/<int:user_id>/requests', methods=['GET'])
def get_local_org_requests(user_id): 
    requesting_user_id_str = request.headers.get('X-User-Id')
    if not requesting_user_id_str: return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try:
        requesting_user_id = int(requesting_user_id_str)
        if requesting_user_id != user_id:
            return jsonify({"error": "Forbidden: You can only access your own requests."}), 403
    except ValueError: return jsonify({"error": "Invalid User ID format."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (user_id,))
            user_details = cur.fetchone()
            if not user_details or user_details['user_type'] != 'local_organisation':
                return jsonify({"error": "Forbidden: User not a local organisation or not found."}), 403

            app.logger.info(f"Fetching PENDING requests for local_org_id: {user_id}")
            cur.execute("""
                SELECT 
                    cr.id as request_id, cr.status, cr.requested_at,
                    c.id as camp_id, c.name as camp_name, c.start_date as camp_start_date,
                    u.id as organizer_id, u.username as organizer_name
                FROM connection_requests cr
                JOIN camps c ON cr.camp_id = c.id
                JOIN users u ON cr.organizer_id = u.id
                WHERE cr.local_org_id = %s AND cr.status = 'pending'
                ORDER BY cr.requested_at DESC;
            """, (user_id,))
            requests_raw = cur.fetchall()
            pending_requests = [row_to_dict(req_raw) for req_raw in requests_raw]
            
            app.logger.info(f"Found {len(pending_requests)} pending requests for local_org_id: {user_id}")
            return jsonify({"pendingRequests": pending_requests}), 200
    except psycopg2.Error as e:
        app.logger.error(f"Error fetching pending requests for local org {user_id}: {e}")
        return jsonify({"error": "Failed to fetch pending requests"}), 500
    finally:
        if conn: conn.close()

@app.route('/api/local-organisation/<int:user_id>/connections', methods=['GET'])
def get_local_org_connections(user_id):
    requesting_user_id_str = request.headers.get('X-User-Id')
    if not requesting_user_id_str: return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try:
        requesting_user_id = int(requesting_user_id_str)
        if requesting_user_id != user_id:
            return jsonify({"error": "Forbidden: You can only access your own connections."}), 403
    except ValueError: return jsonify({"error": "Invalid User ID format."}), 400

    status_filter = request.args.get('status') 

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (user_id,))
            user_details = cur.fetchone()
            if not user_details or user_details['user_type'] != 'local_organisation':
                return jsonify({"error": "Forbidden: User not a local organisation or not found."}), 403

            sql_query = """
                SELECT 
                    cr.id as connection_id, cr.camp_id, c.name as camp_name,
                    cr.organizer_id, u_org.username as organizer_name,
                    cr.status, cr.requested_at, cr.responded_at
                FROM connection_requests cr
                JOIN camps c ON cr.camp_id = c.id
                JOIN users u_org ON cr.organizer_id = u_org.id
                WHERE cr.local_org_id = %s
            """
            params = [user_id]
            if status_filter:
                sql_query += " AND cr.status = %s"
                params.append(status_filter)
            sql_query += " ORDER BY cr.responded_at DESC, cr.requested_at DESC;"
            
            app.logger.info(f"Executing query for local_org_id {user_id}, status '{status_filter}': {sql_query} with params {params}")
            cur.execute(sql_query, tuple(params))
            
            connections_raw = cur.fetchall()
            connections = [row_to_dict(row_raw) for row_raw in connections_raw]
            
            app.logger.info(f"Found {len(connections)} connections for local_org_id: {user_id} with status: {status_filter or 'any'}")
            return jsonify(connections), 200
    except psycopg2.Error as e:
        app.logger.error(f"Error fetching connections for local_org_id {user_id}: {e}")
        return jsonify({"error": "Failed to fetch connections"}), 500
    finally:
        if conn: conn.close()


@app.route('/api/chat/request/<int:request_id>/respond', methods=['PUT'])
def respond_to_connection_request(request_id):
    local_org_user_id_str = request.headers.get('X-User-Id')
    if not local_org_user_id_str: return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try:
        local_org_user_id = int(local_org_user_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID format."}), 400
    
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    new_status = data.get('status')

    if new_status not in ['accepted', 'declined']:
        return jsonify({"error": "Invalid status. Must be 'accepted' or 'declined'."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT id, status, local_org_id FROM connection_requests WHERE id = %s", (request_id,))
            req = cur.fetchone()
            if not req:
                return jsonify({"error": "Request not found."}), 404
            if req['local_org_id'] != local_org_user_id:
                return jsonify({"error": "Forbidden: This request does not belong to you."}), 403
            if req['status'] != 'pending':
                return jsonify({"error": f"Request already responded to (status: {req['status']})."}), 400

            cur.execute(
                """
                UPDATE connection_requests 
                SET status = %s, responded_at = CURRENT_TIMESTAMP 
                WHERE id = %s RETURNING id, status, responded_at;
                """,
                (new_status, request_id)
            )
            updated_request_raw = cur.fetchone()
            conn.commit()

            if updated_request_raw:
                updated_request = row_to_dict(updated_request_raw)
                return jsonify({"message": f"Request {new_status} successfully.", "request": updated_request}), 200
            else:
                app.logger.error(f"Failed to update request {request_id}, no data returned from DB.")
                return jsonify({"error": "Failed to update request status."}), 500
    except psycopg2.Error as e:
        conn.rollback()
        app.logger.error(f"Error responding to request {request_id}: {e}")
        return jsonify({"error": "Failed to respond to request"}), 500
    finally:
        if conn: conn.close()

@app.route('/api/organizer/camp/<int:camp_id>/connections', methods=['GET'])
def get_organizer_camp_connections(camp_id):
    organizer_id_str = request.headers.get('X-User-Id')
    if not organizer_id_str:
        app.logger.warning(f"Attempt to fetch connections for camp {camp_id} without organizer ID.")
        return jsonify({"error": "Unauthorized: Organizer ID missing."}), 401
    try:
        organizer_id = int(organizer_id_str)
    except ValueError:
        app.logger.error(f"Invalid organizer ID format in header: {organizer_id_str}")
        return jsonify({"error": "Invalid organizer ID format."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (organizer_id,))
            user_check = cur.fetchone()
            if not user_check or user_check['user_type'] != 'organizer':
                return jsonify({"error": "Forbidden: Only organizers can view camp connections."}), 403

            cur.execute("SELECT id FROM camps WHERE id = %s AND organizer_id = %s", (camp_id, organizer_id))
            if not cur.fetchone():
                return jsonify({"error": "Camp not found or not owned by this organizer."}), 404

            app.logger.info(f"Fetching connections for organizer_id: {organizer_id} and camp_id: {camp_id}")
            cur.execute("""
                SELECT 
                    cr.id as connection_id, cr.local_org_id,
                    u_local_org.username as local_org_name, cr.status,
                    cr.requested_at, cr.responded_at
                FROM connection_requests cr
                JOIN users u_local_org ON cr.local_org_id = u_local_org.id
                WHERE cr.camp_id = %s AND cr.organizer_id = %s;
            """, (camp_id, organizer_id))
            
            connections_raw = cur.fetchall()
            connections = [row_to_dict(conn_req) for conn_req in connections_raw]
            
            app.logger.info(f"Found {len(connections)} connections for organizer {organizer_id}, camp {camp_id}.")
            return jsonify(connections), 200
    except psycopg2.Error as e:
        app.logger.error(f"Error fetching connections for organizer {organizer_id}, camp {camp_id}: {e}")
        return jsonify({"error": "Failed to fetch connection statuses"}), 500
    finally:
        if conn: conn.close()


@app.route('/api/chat/conversation/<int:connection_id>/messages', methods=['GET'])
def get_chat_messages(connection_id):
    user_id_str = request.headers.get('X-User-Id')
    if not user_id_str: return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try: user_id = int(user_id_str)
    except ValueError: return jsonify({"error": "Invalid User ID format."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT organizer_id, local_org_id, status 
                FROM connection_requests 
                WHERE id = %s
            """, (connection_id,))
            conn_req = cur.fetchone()
            if not conn_req: return jsonify({"error": "Connection not found."}), 404
            if conn_req['status'] != 'accepted': return jsonify({"error": "Chat not active for this connection."}), 403
            if user_id not in [conn_req['organizer_id'], conn_req['local_org_id']]:
                return jsonify({"error": "Forbidden: You are not part of this conversation."}), 403

            cur.execute("""
                SELECT cm.id, cm.sender_id, u.username as sender_name, cm.message_text, cm.sent_at
                FROM chat_messages cm
                JOIN users u ON cm.sender_id = u.id
                WHERE cm.connection_request_id = %s
                ORDER BY cm.sent_at ASC;
            """, (connection_id,))
            messages_raw = cur.fetchall()
            messages = [row_to_dict(msg_raw) for msg_raw in messages_raw]
            return jsonify(messages), 200
    except psycopg2.Error as e:
        app.logger.error(f"Error fetching messages for connection {connection_id}: {e}")
        return jsonify({"error": "Failed to fetch messages"}), 500
    finally:
        if conn: conn.close()

@app.route('/api/chat/conversation/<int:connection_id>/message', methods=['POST'])
def send_chat_message(connection_id):
    sender_id_str = request.headers.get('X-User-Id') 
    if not sender_id_str: return jsonify({"error": "Unauthorized: Sender ID missing from header."}), 401
    try: sender_id = int(sender_id_str)
    except ValueError: return jsonify({"error": "Invalid Sender ID format in header."}), 400

    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    message_text = data.get('text')

    if not message_text or not message_text.strip():
        return jsonify({"error": "Message text cannot be empty."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT cr.organizer_id, cr.local_org_id, cr.status 
                FROM connection_requests cr
                WHERE cr.id = %s
            """, (connection_id,))
            conn_req = cur.fetchone()
            if not conn_req: return jsonify({"error": "Connection not found."}), 404
            if conn_req['status'] != 'accepted': return jsonify({"error": "Chat not active for this connection."}), 403
            if sender_id not in [conn_req['organizer_id'], conn_req['local_org_id']]:
                return jsonify({"error": "Forbidden: Sender not part of this conversation."}), 403

            cur.execute(
                """
                INSERT INTO chat_messages (connection_request_id, sender_id, message_text)
                VALUES (%s, %s, %s) RETURNING id, sender_id, message_text, sent_at;
                """,
                (connection_id, sender_id, message_text)
            )
            new_message_raw = cur.fetchone()
            conn.commit()
            
            if new_message_raw:
                new_message = row_to_dict(new_message_raw)
                cur.execute("SELECT username FROM users WHERE id = %s", (new_message['sender_id'],))
                sender_details = cur.fetchone()
                if sender_details: new_message['sender_name'] = sender_details['username']
                
                return jsonify({"message": "Message sent successfully.", "chatMessage": new_message}), 201
            else:
                app.logger.error(f"Failed to send message for connection {connection_id}, no data returned from DB.")
                return jsonify({"error": "Failed to send message, no data returned from DB."}), 500

    except psycopg2.Error as e:
        conn.rollback()
        app.logger.error(f"Error sending message for connection {connection_id}: {e}")
        return jsonify({"error": "Failed to send message"}), 500
    finally:
        if conn: conn.close()

# --- NEW: Local Hugging Face Translation Function ---
def translate_text_local_hf(text, target_lang_simple, source_lang_simple="auto"):
    global local_translation_pipeline, LOCAL_TRANSLATION_MODEL_INIT_STATUS, LANGUAGE_CODE_MAP_NLLB

    if LOCAL_TRANSLATION_MODEL_INIT_STATUS == "pending":
        app.logger.warning("Local translation model accessed before explicit initialization. Attempting now.")
        initialize_local_translation_model()

    if LOCAL_TRANSLATION_MODEL_INIT_STATUS == "failed" or local_translation_pipeline is None:
        app.logger.error(f"Local translation model ({HF_TRANSLATION_MODEL_ID}) is not available or failed to initialize.")
        return text # Return original text if translation model failed

    if not text or not text.strip():
        app.logger.warning("translate_text_local_hf: Received empty text, returning as is.")
        return text

    # Map simple language codes to NLLB specific codes
    nllb_target_lang = LANGUAGE_CODE_MAP_NLLB.get(target_lang_simple)
    nllb_source_lang = None

    if source_lang_simple == "auto":
        # NLLB requires src_lang. If 'auto', we need a strategy.
        # For chatbot: if translating user message TO 'en', source is user's language.
        # If translating bot message FROM 'en', source is 'en'.
        # This function is now more general, so 'auto' is harder.
        # For now, if 'auto' and target is 'en', we can't know source.
        # If 'auto' and target is NOT 'en', assume source is 'en'.
        if target_lang_simple == "en":
            app.logger.warning(f"translate_text_local_hf: 'auto' source language specified for target 'en'. NLLB requires explicit source. Cannot translate.")
            return text # Or raise an error
        else: # Assume source is English if auto and target is not English
            nllb_source_lang = LANGUAGE_CODE_MAP_NLLB.get("en")
            app.logger.info(f"translate_text_local_hf: 'auto' source, assuming 'en' ({nllb_source_lang}) as source for target '{target_lang_simple}'.")
    else:
        nllb_source_lang = LANGUAGE_CODE_MAP_NLLB.get(source_lang_simple)

    if not nllb_target_lang:
        app.logger.error(f"Unsupported target language for NLLB: '{target_lang_simple}'. Check LANGUAGE_CODE_MAP_NLLB.")
        return text
    if not nllb_source_lang:
        app.logger.error(f"Unsupported source language for NLLB: '{source_lang_simple}'. Check LANGUAGE_CODE_MAP_NLLB.")
        return text
    
    if nllb_source_lang == nllb_target_lang:
        return text # No translation needed

    try:
        app.logger.info(f"Attempting local translation from {nllb_source_lang} to {nllb_target_lang} for text: '{text[:50]}...'")
        # NLLB pipeline call:
        result = local_translation_pipeline(text, src_lang=nllb_source_lang, tgt_lang=nllb_target_lang)
        
        if result and isinstance(result, list) and result[0] and "translation_text" in result[0]:
            translated_text = result[0]["translation_text"]
            app.logger.info(f"Successfully translated with local NLLB model. Result: '{translated_text[:50]}...'")
            return translated_text
        else:
            app.logger.error(f"Unexpected response format from local NLLB translation: {result}")
            return text
    except Exception as e:
        app.logger.error(f"Error during local NLLB translation: {e}", exc_info=True)
        return text


# --- AI Chatbot Helper Function (Local Hugging Face Model) ---
def query_huggingface_model_local(prompt_text):
    global local_chatbot_pipeline, local_chatbot_tokenizer, LOCAL_CHATBOT_MODEL_INIT_STATUS

    if LOCAL_CHATBOT_MODEL_INIT_STATUS == "pending":
        app.logger.warning("Local chatbot model accessed before explicit initialization. Attempting now.")
        initialize_local_chatbot_model() 

    if LOCAL_CHATBOT_MODEL_INIT_STATUS == "failed" or local_chatbot_pipeline is None or local_chatbot_tokenizer is None:
        app.logger.error(f"Local chatbot model ({HF_CHATBOT_MODEL_ID}) is not available or failed to initialize.")
        return "Chatbot is currently unavailable (local model issue)."

    try:
        app.logger.info(f"Querying local model {HF_CHATBOT_MODEL_ID} with prompt (first 50 chars): '{prompt_text[:50]}...'")
        
        prompt_token_ids = local_chatbot_tokenizer.encode(prompt_text, return_tensors='pt')
        prompt_length = prompt_token_ids.shape[1]
        
        max_new_tokens = 150 
        calculated_max_length = prompt_length + max_new_tokens

        model_max_seq_len = local_chatbot_pipeline.model.config.max_position_embeddings
        if calculated_max_length > model_max_seq_len:
            calculated_max_length = model_max_seq_len
            app.logger.warning(f"Calculated max_length ({prompt_length + max_new_tokens}) exceeded model max sequence length ({model_max_seq_len}). Truncating to {calculated_max_length}.")
            if prompt_length >= calculated_max_length: 
                app.logger.error(f"Prompt length ({prompt_length}) is too long for the model's max sequence length ({model_max_seq_len}). Cannot generate new tokens.")
                return "The input message is too long for the chatbot to process."

        results = local_chatbot_pipeline(
            prompt_text,
            max_length=calculated_max_length, 
            num_return_sequences=1,
        )
        
        if results and isinstance(results, list) and results[0] and "generated_text" in results[0]:
            full_generated_text = results[0]["generated_text"]
            
            bot_response = full_generated_text
            if full_generated_text.startswith(prompt_text):
                bot_response = full_generated_text[len(prompt_text):].strip()
            else:
                 app.logger.warning(f"Generated text from local model did not start with the exact prompt. Prompt: '{prompt_text[:50]}...', Generated: '{full_generated_text[:100]}...' Using heuristic to strip prompt.")
                 if "Assistant:" in full_generated_text:
                     parts = full_generated_text.split("Assistant:", 1)
                     if len(parts) > 1:
                         bot_response = parts[1].strip()

            app.logger.info(f"Local model {HF_CHATBOT_MODEL_ID} response successfully retrieved: '{bot_response[:100]}...'")
            return bot_response
        else:
            app.logger.error(f"Unexpected local model response format from {HF_CHATBOT_MODEL_ID}: {results}")
            return "Chatbot received an unexpected response from the local model."
            
    except Exception as e:
        app.logger.error(f"An unexpected error occurred while querying local Hugging Face model {HF_CHATBOT_MODEL_ID}: {e}", exc_info=True)
        return "An error occurred while communicating with the local chatbot model."

# --- Patient Chatbot, Feedback, and Translation Endpoints ---
@app.route('/api/translate', methods=['POST'])
def translate_api_endpoint():
    user_id_str = request.headers.get('X-User-Id') 
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    text_to_translate = data.get('text')
    target_lang_simple = data.get('target_lang') # e.g., 'hi'
    source_lang_simple = data.get('source_lang', 'auto') # e.g., 'en' or 'auto'

    if not text_to_translate or not target_lang_simple:
        return jsonify({"error": "Missing 'text' or 'target_lang'"}), 400

    # --- UPDATED to use local HF translation ---
    translated_text = translate_text_local_hf(text_to_translate, target_lang_simple, source_lang_simple)
    
    detected_source_for_response = source_lang_simple
    if source_lang_simple == 'auto' and LANGUAGE_CODE_MAP_NLLB.get(target_lang_simple) != LANGUAGE_CODE_MAP_NLLB.get("en"):
        # If auto was used and we assumed English source because target wasn't English
        detected_source_for_response = "en (assumed)"
    elif source_lang_simple == 'auto':
        detected_source_for_response = "auto (detection not implemented for NLLB, source must be explicit or assumed)"


    if translated_text == text_to_translate and target_lang_simple != source_lang_simple : 
        app.logger.warning(f"Local translation from '{source_lang_simple}' to '{target_lang_simple}' for '{text_to_translate[:50]}...' might have failed or text was identical, returning original.")
    
    return jsonify({"translated_text": translated_text, "source_lang_detected": detected_source_for_response}), 200


@app.route('/api/patient/chatbot', methods=['POST'])
def patient_chatbot():
    current_user_id_str = request.headers.get('X-User-Id')
    if not current_user_id_str:
        return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try:
        current_user_id = int(current_user_id_str)
    except ValueError:
        return jsonify({"error": "Invalid User ID format."}), 400

    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    user_message = data.get('message')
    target_language_simple = data.get('language', 'en') # User's language, e.g., 'hi'
    patient_record_id = data.get('patient_record_id') 

    if not user_message:
        return jsonify({"error": "Message is required."}), 400

    disease_info = "not specified"
    location_info = "not specified"
    patient_name = "Patient" 

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed for chatbot context."}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if patient_record_id:
                cur.execute("SELECT name, disease_detected, area_location FROM patients WHERE id = %s AND user_id = %s", 
                            (patient_record_id, current_user_id))
                patient_context = cur.fetchone()
                if patient_context:
                    patient_name = patient_context['name']
                    disease_info = patient_context['disease_detected'] or disease_info
                    location_info = patient_context['area_location'] or location_info
            else: 
                cur.execute("SELECT name, disease_detected, area_location FROM patients WHERE user_id = %s ORDER BY created_at DESC LIMIT 1", 
                            (current_user_id,))
                patient_context = cur.fetchone()
                if patient_context:
                    patient_name = patient_context['name']
                    disease_info = patient_context['disease_detected'] or disease_info
                    location_info = patient_context['area_location'] or location_info
            
            cur.execute(
                """
                INSERT INTO patient_chat_messages (patient_user_id, patient_record_id, message_text, sender_type, language)
                VALUES (%s, %s, %s, 'user', %s)
                """, (current_user_id, patient_record_id, user_message, target_language_simple)
            )
            conn.commit()

    except psycopg2.Error as e:
        app.logger.error(f"Database error fetching patient context for chatbot: {e}")
    finally:
        if conn: conn.close()
    
    message_for_bot = user_message
    if target_language_simple != 'en':
        app.logger.info(f"Translating user message from '{target_language_simple}' to 'en' for bot. Original: '{user_message[:100]}...'")
        # --- UPDATED to use local HF translation ---
        message_for_bot = translate_text_local_hf(user_message, "en", target_language_simple) # target='en', source=user's lang
        if message_for_bot == user_message and target_language_simple != "en":
             app.logger.warning(f"User message translation from '{target_language_simple}' to 'en' did not change the text. Using original for bot.")


    prompt = f"""You are a helpful medical information assistant for GoMedCamp.
A patient, {patient_name}, is asking for information.
Patient's detected condition: {disease_info}.
Patient's location: {location_info}.
The patient says (translated to English for you, if originally not in English): "{message_for_bot}"

Please provide helpful, general information. 
Do NOT give specific medical diagnoses or treatment plans.
Always advise the patient to consult with a qualified healthcare professional for any medical concerns or before making any health decisions.
If asked about where to go, suggest looking for local clinics, hospitals, or specialists in their area ({location_info}) and consulting the camp organizers for referrals if applicable.
Keep your response concise and easy to understand. Respond in English.

Assistant: """

    app.logger.info(f"Sending prompt to Local Chatbot Model: (first 50 chars) '{prompt[:50]}...'")
    bot_response_en = query_huggingface_model_local(prompt) # This is in English

    final_bot_response = bot_response_en
    if target_language_simple != 'en' and bot_response_en and (bot_response_en not in INTERNAL_BOT_ERROR_MESSAGES):
        app.logger.info(f"Attempting to translate bot response from 'en' to '{target_language_simple}'. Original: '{bot_response_en[:100]}...'")
        # --- UPDATED to use local HF translation ---
        final_bot_response = translate_text_local_hf(bot_response_en, target_language_simple, "en") # target=user's lang, source='en'
        if final_bot_response == bot_response_en and target_language_simple != "en": 
            app.logger.warning(f"Translation of bot response from 'en' to '{target_language_simple}' did not change the text. Response was: '{bot_response_en[:100]}...'")
    elif bot_response_en in INTERNAL_BOT_ERROR_MESSAGES:
        app.logger.info(f"Bot response is an internal error message, not translating: {bot_response_en}")
    elif not bot_response_en:
        app.logger.warning("Bot response was empty, not attempting translation.")


    conn_store_bot = get_db_connection()
    if conn_store_bot:
        try:
            with conn_store_bot.cursor() as cur_store_bot:
                cur_store_bot.execute(
                    """
                    INSERT INTO patient_chat_messages (patient_user_id, patient_record_id, message_text, sender_type, language)
                    VALUES (%s, %s, %s, 'bot', %s)
                    """, (current_user_id, patient_record_id, final_bot_response, target_language_simple)
                )
                conn_store_bot.commit()
        except psycopg2.Error as e_store:
            app.logger.error(f"Database error storing bot chat message: {e_store}")
        finally:
            conn_store_bot.close()

    return jsonify({"reply": final_bot_response, "language": target_language_simple}), 200


@app.route('/api/patient/feedback', methods=['POST'])
def patient_feedback():
    current_user_id_str = request.headers.get('X-User-Id')
    if not current_user_id_str:
        return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try:
        current_user_id = int(current_user_id_str)
    except ValueError:
        return jsonify({"error": "Invalid User ID format."}), 400

    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    feedback_text = data.get('feedback_text')
    rating = data.get('rating') 
    patient_record_id = data.get('patient_record_id') 
    language = data.get('language', 'en') 

    if not feedback_text:
        return jsonify({"error": "Feedback text is required."}), 400
    
    if rating is not None:
        try:
            rating = int(rating)
            if not (1 <= rating <= 5):
                return jsonify({"error": "Rating must be an integer between 1 and 5."}), 400
        except ValueError:
            return jsonify({"error": "Invalid rating format."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed."}), 500
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO patient_feedback (patient_user_id, patient_record_id, feedback_text, rating, language)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (current_user_id, patient_record_id, feedback_text, rating, language)
            )
            conn.commit()
        return jsonify({"message": "Feedback submitted successfully."}), 201
    except psycopg2.Error as e:
        conn.rollback()
        app.logger.error(f"Database error submitting feedback: {e}")
        return jsonify({"error": "Failed to submit feedback."}), 500
    finally:
        if conn: conn.close()

# --- NEW Endpoints for Reviews and Follow-ups ---

@app.route('/api/camps', methods=['GET']) # New endpoint for patients to list camps for review
def get_all_camps_for_review():
    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT id, name FROM camps WHERE status IN ('active', 'completed', 'planned') ORDER BY name ASC")
            camps_raw = cur.fetchall()
            camps = [row_to_dict(camp) for camp in camps_raw]
            return jsonify(camps), 200
    except psycopg2.Error as e:
        app.logger.error(f"Error fetching all camps for review: {e}")
        return jsonify({"error": "Failed to fetch camps"}), 500
    finally:
        if conn: conn.close()

@app.route('/api/reviews', methods=['POST']) # New endpoint for patients to submit reviews
def submit_camp_review():
    patient_user_id_str = request.headers.get('X-User-Id')
    if not patient_user_id_str:
        return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try:
        patient_user_id = int(patient_user_id_str)
    except ValueError:
        return jsonify({"error": "Invalid User ID format."}), 400

    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    camp_id = data.get('campId')
    rating = data.get('rating')
    comment = data.get('comment')

    if not camp_id or rating is None:
        return jsonify({"error": "Missing required fields: campId, rating."}), 400
    
    try:
        rating = int(rating)
        if not (1 <= rating <= 5):
            return jsonify({"error": "Rating must be between 1 and 5."}), 400
    except ValueError:
        return jsonify({"error": "Invalid rating format."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (patient_user_id,))
            user = cur.fetchone()
            if not user or user['user_type'] != 'requester': # Patients are 'requester' type
                return jsonify({"error": "Forbidden: Only registered patients (requesters) can submit reviews."}), 403

            cur.execute("SELECT id FROM camps WHERE id = %s", (camp_id,))
            if not cur.fetchone():
                return jsonify({"error": "Camp not found."}), 404
            
            cur.execute("SELECT id FROM camp_reviews WHERE camp_id = %s AND patient_user_id = %s", (camp_id, patient_user_id))
            if cur.fetchone():
                return jsonify({"error": "You have already reviewed this camp."}), 409

            cur.execute(
                """
                INSERT INTO camp_reviews (camp_id, patient_user_id, rating, comment)
                VALUES (%s, %s, %s, %s) RETURNING id;
                """,
                (camp_id, patient_user_id, rating, comment)
            )
            new_review_id = cur.fetchone()['id']
            conn.commit()
            return jsonify({"message": "Review submitted successfully.", "review_id": new_review_id}), 201
    except psycopg2.Error as e:
        conn.rollback()
        app.logger.error(f"Error submitting review for camp {camp_id} by user {patient_user_id}: {e}")
        return jsonify({"error": "Failed to submit review."}), 500
    finally:
        if conn: conn.close()

@app.route('/api/camps/<int:camp_id>/reviews', methods=['GET']) # New endpoint for organizers to view reviews
def get_camp_reviews_for_organizer(camp_id):
    organizer_user_id_str = request.headers.get('X-User-Id')
    if not organizer_user_id_str:
        return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try:
        organizer_user_id = int(organizer_user_id_str)
    except ValueError:
        return jsonify({"error": "Invalid User ID format."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (organizer_user_id,))
            user = cur.fetchone()
            if not user or user['user_type'] != 'organizer':
                return jsonify({"error": "Forbidden: Only organizers can view camp reviews."}), 403

            cur.execute("SELECT organizer_id FROM camps WHERE id = %s", (camp_id,))
            camp = cur.fetchone()
            if not camp:
                return jsonify({"error": "Camp not found."}), 404
            if camp['organizer_id'] != organizer_user_id:
                return jsonify({"error": "Forbidden: You are not the organizer of this camp."}), 403
            
            cur.execute(
                """
                SELECT cr.id, cr.patient_user_id, u.username as patient_name, cr.rating, cr.comment, cr.created_at
                FROM camp_reviews cr
                JOIN users u ON cr.patient_user_id = u.id
                WHERE cr.camp_id = %s
                ORDER BY cr.created_at DESC;
                """, (camp_id,)
            )
            reviews_raw = cur.fetchall()
            reviews = [row_to_dict(review) for review in reviews_raw]
            return jsonify(reviews), 200
    except psycopg2.Error as e:
        app.logger.error(f"Error fetching reviews for camp {camp_id}, organizer {organizer_user_id}: {e}")
        return jsonify({"error": "Failed to fetch reviews."}), 500
    finally:
        if conn: conn.close()

@app.route('/api/camps/<int:camp_id>/patients/followup', methods=['POST']) # New endpoint for organizers
def add_patient_for_followup(camp_id):
    organizer_user_id_str = request.headers.get('X-User-Id')
    if not organizer_user_id_str:
        return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try:
        organizer_user_id = int(organizer_user_id_str)
    except ValueError:
        return jsonify({"error": "Invalid User ID format."}), 400

    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    patient_identifier = data.get('patientIdentifier')
    notes = data.get('notes')

    if not patient_identifier:
        return jsonify({"error": "Patient identifier is required."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (organizer_user_id,))
            user = cur.fetchone()
            if not user or user['user_type'] != 'organizer':
                return jsonify({"error": "Forbidden: Only organizers can add patients for follow-up."}), 403

            cur.execute("SELECT organizer_id FROM camps WHERE id = %s", (camp_id,))
            camp = cur.fetchone()
            if not camp:
                return jsonify({"error": "Camp not found."}), 404
            if camp['organizer_id'] != organizer_user_id:
                return jsonify({"error": "Forbidden: You are not the organizer of this camp."}), 403
            
            linked_patient_user_id = None
            cur.execute("SELECT id FROM users WHERE (email = %s OR phone_number = %s) AND user_type = 'requester'", (patient_identifier, patient_identifier))
            matched_user = cur.fetchone()
            if matched_user:
                linked_patient_user_id = matched_user['id']

            cur.execute(
                """
                INSERT INTO camp_follow_ups (camp_id, patient_identifier, notes, added_by_organizer_id, linked_patient_user_id)
                VALUES (%s, %s, %s, %s, %s) RETURNING id, patient_identifier, notes, created_at;
                """,
                (camp_id, patient_identifier, notes, organizer_user_id, linked_patient_user_id)
            )
            new_followup_raw = cur.fetchone()
            conn.commit()
            new_followup = row_to_dict(new_followup_raw)
            return jsonify({"message": "Patient added for follow-up.", "follow_up": new_followup}), 201
    except psycopg2.Error as e:
        conn.rollback()
        app.logger.error(f"Error adding patient for follow-up to camp {camp_id} by organizer {organizer_user_id}: {e}")
        return jsonify({"error": "Failed to add patient for follow-up."}), 500
    finally:
        if conn: conn.close()

@app.route('/api/camps/<int:camp_id>/patients/followup', methods=['GET']) # New endpoint for organizers
def get_camp_followup_patients(camp_id):
    organizer_user_id_str = request.headers.get('X-User-Id')
    if not organizer_user_id_str:
        return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try:
        organizer_user_id = int(organizer_user_id_str)
    except ValueError:
        return jsonify({"error": "Invalid User ID format."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT user_type FROM users WHERE id = %s", (organizer_user_id,))
            user = cur.fetchone()
            if not user or user['user_type'] != 'organizer':
                return jsonify({"error": "Forbidden: Only organizers can view follow-up lists."}), 403

            cur.execute("SELECT organizer_id FROM camps WHERE id = %s", (camp_id,))
            camp = cur.fetchone()
            if not camp:
                return jsonify({"error": "Camp not found."}), 404
            if camp['organizer_id'] != organizer_user_id:
                return jsonify({"error": "Forbidden: You are not the organizer of this camp."}), 403
            
            cur.execute(
                """
                SELECT id, patient_identifier, notes, created_at, linked_patient_user_id
                FROM camp_follow_ups
                WHERE camp_id = %s
                ORDER BY created_at DESC;
                """, (camp_id,)
            )
            followups_raw = cur.fetchall()
            followups = [row_to_dict(fu) for fu in followups_raw]
            return jsonify(followups), 200
    except psycopg2.Error as e:
        app.logger.error(f"Error fetching follow-up list for camp {camp_id}, organizer {organizer_user_id}: {e}")
        return jsonify({"error": "Failed to fetch follow-up list."}), 500
    finally:
        if conn: conn.close()

@app.route('/api/patient/followup-eligibility', methods=['GET']) # New endpoint for patients
def check_patient_followup_eligibility():
    patient_user_id_str = request.headers.get('X-User-Id')
    if not patient_user_id_str:
        return jsonify({"error": "Unauthorized: User ID missing."}), 401
    try:
        patient_user_id = int(patient_user_id_str)
    except ValueError:
        return jsonify({"error": "Invalid User ID format."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT email, phone_number, user_type FROM users WHERE id = %s", (patient_user_id,))
            user = cur.fetchone()
            if not user or user['user_type'] != 'requester': # Patients are 'requester' type
                return jsonify({"error": "Forbidden: User not a patient or not found."}), 403
            
            patient_email = user['email']
            patient_phone = user['phone_number']
            
            cur.execute(
                """
                SELECT cf.id, cf.notes, c.name as camp_name
                FROM camp_follow_ups cf
                JOIN camps c ON cf.camp_id = c.id
                WHERE cf.linked_patient_user_id = %s 
                   OR cf.patient_identifier = %s
                   OR (%s IS NOT NULL AND cf.patient_identifier = %s)
                ORDER BY cf.created_at DESC
                LIMIT 1; 
                """,
                (patient_user_id, patient_email, patient_phone, patient_phone)
            )
            eligible_followup = cur.fetchone()
            
            if eligible_followup:
                message = f"You have a follow-up scheduled regarding camp '{eligible_followup['camp_name']}'."
                if eligible_followup['notes']:
                    message += f" Notes: {eligible_followup['notes']}"
                return jsonify({"eligible": True, "message": message, "follow_up_details": row_to_dict(eligible_followup)}), 200
            else:
                return jsonify({"eligible": False, "message": "You are not currently scheduled for any follow-ups via the voice assistant."}), 200
    except psycopg2.Error as e:
        app.logger.error(f"Error checking follow-up eligibility for user {patient_user_id}: {e}")
        return jsonify({"error": "Failed to check follow-up eligibility."}), 500
    finally:
        if conn: conn.close()
# --- END NEW Endpoints ---


@app.route('/')
def index():
    return "GoMedCamp Backend is running!"

if __name__ == '__main__':
    log_level = logging.DEBUG if app.debug else logging.INFO
    if not logging.getLogger().hasHandlers() and not app.logger.handlers:
        logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s: %(name)s: %(message)s [in %(pathname)s:%(lineno)d]')
    
    app.logger.setLevel(log_level)

    app.logger.info("Application starting up...")
    if not check_db_env_vars():
        app.logger.critical("Application may not start properly due to missing DB env vars.")
    
    app.logger.info(f"Expecting indicator JSONs in: {os.path.abspath(BASE_JSON_DIR)}")
    if not os.path.isdir(BASE_JSON_DIR): app.logger.warning(f"APP_BASE_JSON_DIR ('{os.path.abspath(BASE_JSON_DIR)}') not found.")
    app.logger.info(f"Expecting geographic points CSV at: {os.path.abspath(CSV_POINTS_PATH)}")
    if not os.path.isfile(CSV_POINTS_PATH): app.logger.warning(f"APP_CSV_POINTS_PATH ('{os.path.abspath(CSV_POINTS_PATH)}') not found.")

    app.logger.info(f"Hugging Face Chatbot Model ID (Local): {HF_CHATBOT_MODEL_ID}")
    initialize_local_chatbot_model()
    if LOCAL_CHATBOT_MODEL_INIT_STATUS == "success":
        app.logger.info(f"Local chatbot model '{HF_CHATBOT_MODEL_ID}' ready.")
    else:
        app.logger.error(f"Local chatbot model '{HF_CHATBOT_MODEL_ID}' FAILED to initialize. Chatbot functionality will be impaired.")

    app.logger.info(f"Hugging Face Translation Model ID (Local): {HF_TRANSLATION_MODEL_ID}")
    initialize_local_translation_model()
    if LOCAL_TRANSLATION_MODEL_INIT_STATUS == "success":
        app.logger.info(f"Local translation model '{HF_TRANSLATION_MODEL_ID}' ready.")
    else:
        app.logger.error(f"Local translation model '{HF_TRANSLATION_MODEL_ID}' FAILED to initialize. Translation functionality will be impaired.")
    
    # app.logger.info(f"Lingva API URL: {LINGVA_API_URL}") # Lingva no longer used

    create_tables() 
    
    app.logger.info(f"Starting Flask development server on port 5001. Debug mode: {app.debug}")
    app.run(host='0.0.0.0', port=5001, debug=True)
