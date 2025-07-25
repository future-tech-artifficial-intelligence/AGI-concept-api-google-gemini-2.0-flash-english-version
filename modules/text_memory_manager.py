"""
Textual Memory Management Module for Artificial Intelligence Conversations
GOOGLE GEMINI 2.0 FLASH API

This module allows storing and retrieving conversations as text files,
complementing SQLite storage.
"""

import os
import json
import logging
from datetime import datetime
import pathlib
from typing import Dict, List, Any, Optional, Union

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("text_memory_manager")

# Base directory for textual conversations
BASE_CONVERSATIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'data', 'conversations_text')
                                     
# Directory for image uploads
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          'data', 'uploads')

# Create directories if they don't exist
os.makedirs(BASE_CONVERSATIONS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

class TextMemoryManager:
    """
    Textual memory manager that allows storing and retrieving
    conversations as text files.
    """
    
    @staticmethod
    def get_user_dir(user_id: int) -> str:
        """
        Gets the path to the user's directory for text conversations.
        
        Args:
            user_id: User ID
            
        Returns:
            Path to the user directory
        """
        user_dir = os.path.join(BASE_CONVERSATIONS_DIR, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        return user_dir
        
    @staticmethod
    def get_conversation_file_path(user_id: int, session_id: str) -> str:
        """
        Gets the path to the conversation file.
        
        Args:
            user_id: User ID
            session_id: Session ID
            
        Returns:
            Path to the conversation file
        """
        user_dir = TextMemoryManager.get_user_dir(user_id)
        return os.path.join(user_dir, f"{session_id}.txt")
        
    @staticmethod
    def save_message(user_id: int, 
                    session_id: str, 
                    message_type: str, 
                    content: str,
                    image_path: Optional[str] = None,
                    title: Optional[str] = None) -> bool:
        """
        Saves a message to the conversation file.
        
        Args:
            user_id: User ID
            session_id: Session ID
            message_type: Message type ('user' or 'assistant')
            content: Message content
            image_path: Image path (optional)
            title: Conversation title (optional, for the first entry)
            
        Returns:
            True if the message was saved successfully
        """
        file_path = TextMemoryManager.get_conversation_file_path(user_id, session_id)
        file_exists = os.path.exists(file_path)
        
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                # If the file was just created, add a header
                if not file_exists:
                    conversation_title = title or f"Conversation from {datetime.now().strftime('%d/%m/%Y')}"
                    f.write(f"# {conversation_title}\n")
                    f.write(f"Session ID: {session_id}\n")
                    f.write(f"User ID: {user_id}\n")
                    f.write(f"Creation date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
                    f.write("---\n\n")
                
                # Format the message
                timestamp = datetime.now().strftime('%H:%M:%S')
                prefix = "User" if message_type == "user" else "Assistant"
                
                f.write(f"**{prefix}** ({timestamp}):\n")
                f.write(f"{content}\n\n")
                
                # Add a reference to the image if present
                if image_path:
                    f.write(f"[Image: {os.path.basename(image_path)}]\n\n")
                    
                f.write("---\n\n")
            
            return True
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            return False

    @staticmethod
    def read_conversation(user_id: int, session_id: str) -> Optional[str]:
        """
        Reads the content of a conversation from a text file.
        
        Args:
            user_id: User ID
            session_id: Session ID
            
        Returns:
            Conversation content or None if the file does not exist
        """
        file_path = TextMemoryManager.get_conversation_file_path(user_id, session_id)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading conversation: {e}")
            return None
    
    @staticmethod
    def list_conversations(user_id: int) -> List[Dict[str, Any]]:
        """
        Lists all conversations for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of user conversations
        """
        user_dir = TextMemoryManager.get_user_dir(user_id)
        conversations = []
        
        try:
            for file_name in os.listdir(user_dir):
                if file_name.endswith('.txt'):
                    session_id = file_name.replace('.txt', '')
                    file_path = os.path.join(user_dir, file_name)
                    
                    # Get file metadata
                    file_stats = os.stat(file_path)
                    creation_time = datetime.fromtimestamp(file_stats.st_ctime)
                    modification_time = datetime.fromtimestamp(file_stats.st_mtime)
                    
                    # Read title from file
                    title = ""
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            first_line = f.readline().strip()
                            if first_line.startswith('# '):
                                title = first_line[2:]
                    except:
                        title = f"Conversation {session_id}"
                    
                    conversations.append({
                        'session_id': session_id,
                        'title': title,
                        'created_at': creation_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'last_updated': modification_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'file_path': file_path
                    })
            
            # Sort by modification date (most recent first)
            conversations.sort(key=lambda x: x['last_updated'], reverse=True)
            
            return conversations
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return []

    @staticmethod
    def delete_conversation(user_id: int, session_id: str) -> bool:
        """
        Deletes a conversation.
        
        Args:
            user_id: User ID
            session_id: Session ID
            
        Returns:
            True if deletion was successful
        """
        file_path = TextMemoryManager.get_conversation_file_path(user_id, session_id)
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                return True
            except Exception as e:
                logger.error(f"Error deleting conversation: {e}")
                return False
        return False

    @staticmethod
    def search_conversations(user_id: int, query: str) -> List[Dict[str, Any]]:
        """
        Searches conversations for a user.
        
        Args:
            user_id: User ID
            query: Text to search
            
        Returns:
            List of conversations containing the query
        """
        conversations = TextMemoryManager.list_conversations(user_id)
        results = []
        
        for conversation in conversations:
            file_path = conversation['file_path']
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if query.lower() in content.lower():
                        # Add an excerpt with context
                        lines = content.split('\n')
                        matching_lines = []
                        for i, line in enumerate(lines):
                            if query.lower() in line.lower():
                                start = max(0, i - 2)
                                end = min(len(lines), i + 3)
                                context = '\n'.join(lines[start:end])
                                matching_lines.append(context)
                        
                        conversation['matching_excerpts'] = matching_lines[:5]  # Limit to 5 excerpts
                        results.append(conversation)
            except Exception as e:
                logger.error(f"Error searching in {file_path}: {e}")
        
        return results
        
    @staticmethod
    def get_user_uploads_dir(user_id: int) -> str:
        """
        Gets the user's uploads directory.
        
        Args:
            user_id: User ID
            
        Returns:
            Path to the user's uploads directory
        """
        user_uploads_dir = os.path.join(UPLOADS_DIR, str(user_id))
        os.makedirs(user_uploads_dir, exist_ok=True)
        return user_uploads_dir
        
    @staticmethod
    def save_uploaded_image(user_id: int, image_data: str, filename: Optional[str] = None) -> Optional[str]:
        """
        Saves an uploaded image.
        
        Args:
            user_id: User ID
            image_data: Image data in base64
            filename: Filename (optional)
            
        Returns:
            Relative path of the saved image or None on error
        """
        import base64
        
        user_uploads_dir = TextMemoryManager.get_user_uploads_dir(user_id)
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"upload_{timestamp}.jpg"
        
        file_path = os.path.join(user_uploads_dir, filename)
        
        try:
            # Check if image data is empty
            if not image_data:
                logger.error("Empty image data")
                return None
                
            # Check if it's already in base64 format or if it's a Data URI format
            if isinstance(image_data, str):
                # Extract binary image data (remove the data:image/xxx;base64, prefix)
                if ',' in image_data:
                    image_data = image_data.split(',', 1)[1]
                # Clean non-base64 characters that might cause problems
                image_data = image_data.strip()
                
                # Decode and save the image
                try:
                    with open(file_path, 'wb') as f:
                        f.write(base64.b64decode(image_data, validate=True))
                    logger.info(f"Image saved successfully at {file_path}")
                except Exception as decode_error:
                    logger.error(f"Base64 decoding error: {str(decode_error)}")
                    return None
            else:
                logger.error("Unsupported image format")
                return None
                
            # Return the relative path
            return os.path.relpath(file_path, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return None

    @staticmethod
    def list_uploaded_images(user_id: int) -> List[Dict[str, Any]]:
        """
        Lists images uploaded by a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of uploaded images
        """
        user_uploads_dir = TextMemoryManager.get_user_uploads_dir(user_id)
        images = []
        
        try:
            for file_name in os.listdir(user_uploads_dir):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    file_path = os.path.join(user_uploads_dir, file_name)
                    file_stats = os.stat(file_path)
                    
                    images.append({
                        'filename': file_name,
                        'path': file_path,
                        'relative_path': os.path.relpath(file_path, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        'size': file_stats.st_size,
                        'created_at': datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            return images
        except Exception as e:
            logger.error(f"Error listing images: {e}")
            return []
