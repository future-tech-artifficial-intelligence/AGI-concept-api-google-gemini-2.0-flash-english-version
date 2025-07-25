#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to list all users in the GeminiChat website  database
"""

import sqlite3
import os
from datetime import datetime

# Path to the database
DB_PATH = 'gemini_chat.db'

def list_all_users():
    """
    Displays a list of all registered users in the database
    """
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # To access columns by name
        cursor = conn.cursor()
        
        # Retrieve all users
        cursor.execute("""
            SELECT id, username, email, created_at 
            FROM users 
            ORDER BY created_at DESC
        """)
        
        users = cursor.fetchall()
        
        if not users:
            print("📭 No users found in the database.")
            return
        
        print("👥 USER LIST")
        print("=" * 60)
        print(f"{'ID':<5} {'Username':<20} {'Email':<25} {'Creation Date':<15}")
        print("-" * 60)
        
        for user in users:
            # Format the date
            date_str = user['created_at']
            if date_str:
                try:
                    date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    date_formatted = date_obj.strftime('%d/%m/%Y')
                except:
                    date_formatted = date_str[:10] if len(date_str) >= 10 else date_str
            else:
                date_formatted = "N/A"
            
            print(f"{user['id']:<5} {user['username']:<20} {user['email']:<25} {date_formatted:<15}")
        
        print("-" * 60)
        print(f"📊 Total: {len(users)} user(s)")
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if conn:
            conn.close()

def get_user_details(user_id):
    """
    Displays the details of a specific user
    """
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Retrieve user information
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            print(f"❌ User with ID {user_id} not found.")
            return
        
        print(f"👤 USER DETAILS (ID: {user_id})")
        print("=" * 50)
        print(f"Username        : {user['username']}")
        print(f"Email           : {user['email']}")
        print(f"Creation Date   : {user['created_at']}")
        
        # Count user conversations
        cursor.execute("SELECT COUNT(*) FROM conversation_sessions WHERE user_id = ?", (user_id,))
        conv_count = cursor.fetchone()[0]
        print(f"Conversations   : {conv_count}")
        
        # Count user messages
        cursor.execute("""
            SELECT COUNT(*) 
            FROM messages m 
            JOIN conversation_sessions cs ON m.session_id = cs.session_id 
            WHERE cs.user_id = ?
        """, (user_id,))
        msg_count = cursor.fetchone()[0]
        print(f"Messages        : {msg_count}")
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if conn:
            conn.close()

def search_users(search_term):
    """
    Searches for users by username or email
    """
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, username, email, created_at 
            FROM users 
            WHERE username LIKE ? OR email LIKE ?
            ORDER BY created_at DESC
        """, (f'%{search_term}%', f'%{search_term}%'))
        
        users = cursor.fetchall()
        
        if not users:
            print(f"🔍 No users found for '{search_term}'")
            return
        
        print(f"🔍 SEARCH RESULTS FOR '{search_term}'")
        print("=" * 60)
        print(f"{'ID':<5} {'Username':<20} {'Email':<25} {'Creation Date':<15}")
        print("-" * 60)
        
        for user in users:
            date_str = user['created_at']
            date_formatted = date_str[:10] if date_str and len(date_str) >= 10 else "N/A"
            print(f"{user['id']:<5} {user['username']:<20} {user['email']:<25} {date_formatted:<15}")
        
        print("-" * 60)
        print(f"📊 {len(users)} result(s) found")
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # No argument - list all users
        list_all_users()
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg.isdigit():
            # Numeric argument - display user details
            get_user_details(int(arg))
        else:
            # Text argument - search users
            search_users(arg)
    else:
        print("Usage:")
        print("  python list_users.py                 # List all users")
        print("  python list_users.py <ID>            # User details")
        print("  python list_users.py <search_term> # Search users")
