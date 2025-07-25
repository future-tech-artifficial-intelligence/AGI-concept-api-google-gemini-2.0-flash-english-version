import os
import uuid
import shutil
import mimetypes
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import sqlite3
from datetime import datetime, timedelta
import json

# Libraries for text extraction
import PyPDF2
# Correction of python-docx module import
try:
    import docx  # The imported module name is 'docx', not 'python_docx'
except ImportError:
    # Fallback if the module is not available
    docx = None
import csv
try:
    import pandas as pd
except ImportError:
    # Fallback if pandas is not available
    pd = None
import chardet

from database import DB_PATH

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {
    'pdf', 'doc', 'docx', 'txt', 'csv', 'xls', 'xlsx',
    'json', 'html', 'xml', 'md', 'rtf'
}
FILE_RETENTION_DAYS = 7  # File retention duration

# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def init_file_db():
    """Initializes the files table in the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Creation of the files table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS uploaded_files (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        original_filename TEXT NOT NULL,
        stored_filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_type TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        content_summary TEXT,
        extracted_text TEXT,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expiry_date TIMESTAMP,
        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    conn.commit()
    conn.close()

def allowed_file(filename: str) -> bool:
    """Checks if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_safe_filename(original_filename: str) -> str:
    """Generates a safe and unique filename"""
    # Get the original file extension
    if '.' in original_filename:
        ext = original_filename.rsplit('.', 1)[1].lower()
    else:
        ext = ''

    # Generate a unique identifier
    unique_id = str(uuid.uuid4())

    # Create the new filename
    if ext:
        return f"{unique_id}.{ext}"
    else:
        return unique_id

def get_file_type(file_path: str) -> str:
    """Determines the file type based on the extension"""
    ext = os.path.splitext(file_path)[1].lower()

    file_types = {
        '.pdf': 'PDF',
        '.doc': 'Word',
        '.docx': 'Word',
        '.txt': 'Text',
        '.csv': 'CSV',
        '.xls': 'Excel',
        '.xlsx': 'Excel',
        '.json': 'JSON',
        '.html': 'HTML',
        '.xml': 'XML',
        '.md': 'Markdown',
        '.rtf': 'Rich Text'
    }

    return file_types.get(ext, 'Unknown')

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + "\n"
    except Exception as e:
        text = f"Error extracting text from PDF: {str(e)}"

    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extracts text from a Word file (docx)"""
    text = ""
    try:
        if docx is not None:
            # Use the docx module if available
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            # Alternative message if the module is not available
            text = "Word document content (extraction unavailable). To analyze this document, install python-docx with the command 'pip install python-docx'."
    except Exception as e:
        text = f"Error extracting text from Word document: {str(e)}"

    return text

def extract_text_from_txt(file_path: str) -> str:
    """Extracts text from a text file, with encoding detection"""
    try:
        # Detect file encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'

        # Read the file with the detected encoding
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            text = file.read()

        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def extract_text_from_csv(file_path: str) -> str:
    """Extracs text from a CSV file"""
    try:
        # Detect encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read(4096)  # Read a sample for detection
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'

        # Try to determine the delimiter
        sniffer = csv.Sniffer()
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            sample = file.read(4096)
            try:
                dialect = sniffer.sniff(sample)
                delimiter = dialect.delimiter
            except:
                delimiter = ','  # Default to a comma

        # Read the CSV
        text = ""
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            reader = csv.reader(file, delimiter=delimiter)
            for row in reader:
                text += " | ".join(row) + "\n"

        # For large files, limit extracted text
        if len(text) > 50000:
            text = text[:50000] + "...\n[Content truncated for size reasons]"

        return text
    except Exception as e:
        return f"Error extracting CSV text: {str(e)}"

def extract_text_from_excel(file_path: str) -> str:
    """Extracts text from an Excel file"""
    try:
        if pd is not None:
            # Use pandas if available
            df_dict = pd.read_excel(file_path, sheet_name=None)
            text = ""

            for sheet_name, df in df_dict.items():
                text += f"Sheet: {sheet_name}\n"
                sheet_text = df.to_string(index=False)

                # For large tables, limit size
                if len(sheet_text) > 25000:
                    sheet_text = sheet_text[:25000] + "...\n[Sheet content truncated]"

                text += sheet_text + "\n\n"

            # Limit total size
            if len(text) > 50000:
                text = text[:50000] + "...\n[Overall content truncated]"
        else:
            text = "Excel file content (extraction unavailable). To analyze this document, install pandas and openpyxl with the command 'pip install pandas openpyxl'."

        return text
    except Exception as e:
        text = f"Error extracting Excel text: {str(e)}"

    return text

def extract_text_from_file(file_path: str) -> str:
    """
    Extracts text from a file based on its type
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    # Dispatch to the appropriate extraction function
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    elif file_extension == '.csv':
        return extract_text_from_csv(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        return extract_text_from_excel(file_path)
    else:
        return f"File type not supported for text extraction: {file_extension}"

def generate_content_summary(text: str, max_length: int = 500) -> str:
    """
    Generates a summary of the file content
    """
    if not text or len(text.strip()) == 0:
        return "No textual content detected."

    # Take the first non-empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    summary = ""

    for line in lines:
        if len(summary) + len(line) <= max_length:
            summary += line + "\n"
        else:
            remaining = max_length - len(summary)
            if remaining > 3:  # At least enough for "..."
                summary += line[:remaining-3] + "..."
            break

    return summary or "Unanalyzable content."

def store_file(user_id: int, file, original_filename: str) -> Dict:
    """
    Stores an uploaded file and saves it to the database
    """
    # Generate a safe filename
    stored_filename = generate_safe_filename(original_filename)
    file_path = os.path.join(UPLOAD_FOLDER, stored_filename)

    # Save the file
    file.save(file_path)

    # Get file size
    file_size = os.path.getsize(file_path)

    # Determine file type
    file_type = get_file_type(original_filename)

    # Extract text
    extracted_text = extract_text_from_file(file_path)

    # Generate content summary
    content_summary = generate_content_summary(extracted_text)

    # Calculate expiration date
    expiry_date = (datetime.now() + timedelta(days=FILE_RETENTION_DAYS)).strftime("%Y-%m-%d %H:%M:%S")

    # Store information in the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO uploaded_files
    (user_id, original_filename, stored_filename, file_path, file_type, file_size, content_summary, extracted_text, expiry_date)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        original_filename,
        stored_filename,
        file_path,
        file_type,
        file_size,
        content_summary,
        extracted_text,
        expiry_date
    ))

    file_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Return file information
    return {
        'id': file_id,
        'original_filename': original_filename,
        'file_type': file_type,
        'file_size': file_size,
        'content_summary': content_summary
    }

def get_file_info(file_id: int) -> Optional[Dict]:
    """
    Retrieves file information
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    SELECT id, user_id, original_filename, file_path, file_type, file_size,
           content_summary, extracted_text, upload_date, expiry_date
    FROM uploaded_files
    WHERE id = ?
    ''', (file_id,))

    file_info = cursor.fetchone()

    if file_info:
        # Update last accessed date
        cursor.execute('''
        UPDATE uploaded_files
        SET last_accessed = CURRENT_TIMESTAMP
        WHERE id = ?
        ''', (file_id,))
        conn.commit()

        # Format information
        info = {
            'id': file_info[0],
            'user_id': file_info[1],
            'original_filename': file_info[2],
            'file_path': file_info[3],
            'file_type': file_info[4],
            'file_size': file_info[5],
            'content_summary': file_info[6],
            'extracted_text': file_info[7],
            'upload_date': file_info[8],
            'expiry_date': file_info[9]
        }
        conn.close()
        return info

    conn.close()
    return None

def get_user_files(user_id: int) -> List[Dict]:
    """
    Retrieves the list of files for a user
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    SELECT id, original_filename, file_type, file_size, content_summary, upload_date
    FROM uploaded_files
    WHERE user_id = ?
    ORDER BY upload_date DESC
    ''', (user_id,))

    files = cursor.fetchall()
    conn.close()

    file_list = []
    for file in files:
        file_list.append({
            'id': file[0],
            'filename': file[1],
            'file_type': file[2],
            'file_size': file[3],
            'content_summary': file[4],
            'upload_date': file[5]
        })

    return file_list

def cleanup_expired_files():
    """
    Cleans up expired files
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Find expired files
    cursor.execute('''
    SELECT id, file_path
    FROM uploaded_files
    WHERE expiry_date < CURRENT_TIMESTAMP
    ''')

    expired_files = cursor.fetchall()

    for file_id, file_path in expired_files:
        # Delete the file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

        # Delete the entry from the database
        cursor.execute('''
        DELETE FROM uploaded_files
        WHERE id = ?
        ''', (file_id,))

    conn.commit()
    conn.close()

def get_file_extension_icon(file_type: str) -> str:
    """
    Returns the Font Awesome icon corresponding to the file type
    """
    icons = {
        'PDF': 'fa-file-pdf',
        'Word': 'fa-file-word',
        'Text': 'fa-file-alt',
        'CSV': 'fa-file-csv',
        'Excel': 'fa-file-excel',
        'JSON': 'fa-file-code',
        'HTML': 'fa-file-code',
        'XML': 'fa-file-code',
        'Markdown': 'fa-file-alt',
        'Rich Text': 'fa-file-alt'
    }

    return icons.get(file_type, 'fa-file')

def format_file_size(size_bytes: int) -> str:
    """
    Formats file size in KB, MB, etc.
    """
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
