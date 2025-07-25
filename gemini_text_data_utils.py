import os
import shutil
from direct_file_access import ensure_text_data_folder_exists, TEXT_DATA_FOLDER, get_all_text_files

def add_text_file(filename, content):
    """Adds a new text file to the data folder"""
    ensure_text_data_folder_exists()
    file_path = os.path.join(TEXT_DATA_FOLDER, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"File {filename} added successfully"

def delete_text_file(filename):
    """Deletes a text file from the data folder"""
    file_path = os.path.join(TEXT_DATA_FOLDER, filename)
    if os.path.exists(file_path) and filename.endswith('.txt'):
        os.remove(file_path)
        return f"File {filename} deleted successfully"
    return f"File {filename} not found"

def get_folder_stats():
    """Gets statistics for the text data folder"""
    files = get_all_text_files()
    total_size = 0
    for file in files:
        file_path = os.path.join(TEXT_DATA_FOLDER, file)
        total_size += os.path.getsize(file_path)
    
    return {
        "num_files": len(files),
        "total_size_bytes": total_size,
        "total_size_mb": total_size / (1024 * 1024),
        "file_list": files
    }

def backup_text_folder(backup_location):
    """Performs a backup of the text data folder"""
    if not os.path.exists(backup_location):
        os.makedirs(backup_location)
    
    backup_folder = os.path.join(backup_location, 'gemini_text_data_backup')
    if os.path.exists(backup_folder):
        shutil.rmtree(backup_folder)
    
    shutil.copytree(TEXT_DATA_FOLDER, backup_folder)
    return f"Backup performed successfully in {backup_folder}"
