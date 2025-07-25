#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to view database users
"""

import sqlite3
import os
from tabulate import tabulate  # Added tabulate for better display

def show_users():
    """Displays all users"""
    db_path = 'gemini_chat.db'
    
    if not os.path.exists(db_path):
        print("Database not found")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the users table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            print("The 'users' table does not exist in the database.")
            return
            
        cursor.execute("SELECT id, username, email, created_at FROM users ORDER BY id")
        users = cursor.fetchall()
        
        if not users:
            print("No users found")
            return
        
        print("\n=== USER LIST ===")
        
        # Using tabulate for table display
        headers = ["ID", "Username", "Email", "Creation Date"]
        print(tabulate(users, headers=headers, tablefmt="grid"))
        
        print(f"\nTotal: {len(users)} user(s)")
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            print("Database connection closed")

if __name__ == "__main__":
    try:
        # Check if tabulate is installed
        import tabulate
    except ImportError:
        print("The 'tabulate' module is required. Installing...")
        import subprocess
        try:
            subprocess.check_call(["pip", "install", "tabulate"])
            print("Module 'tabulate' installed successfully!")
        except Exception as e:
            print(f"Could not install 'tabulate' module: {e}")
            print("Use the command: pip install tabulate")
            print("Displaying in simple mode...")
            
    show_users()
