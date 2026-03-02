# auth_system.py
import json
import hashlib
import secrets
from datetime import datetime, timedelta
import os
from typing import Dict, Optional

class UserAuthSystem:
    def __init__(self, users_file='users.json'):
        self.users_file = users_file
        self.sessions = {}  # session_token -> (username, expiry)
        self.load_users()
    
    def load_users(self):
        """Load users from JSON file"""
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        else:
            self.users = {}
            self.save_users()
    
    def save_users(self):
        """Save users to JSON file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def hash_password(self, password: str, salt: str = None) -> Dict:
        """Hash password with salt using SHA-256"""
        if not salt:
            salt = secrets.token_hex(16)
        
        # Combine password and salt, then hash
        password_hash = hashlib.sha256(
            (password + salt).encode()
        ).hexdigest()
        
        return {
            'hash': password_hash,
            'salt': salt
        }
    
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash"""
        computed = self.hash_password(password, salt)
        return computed['hash'] == stored_hash
    
    def register_user(self, username: str, password: str, email: str = None) -> Dict:
        """Register a new user"""
        if username in self.users:
            return {'success': False, 'error': 'Username already exists'}
        
        # Hash password
        password_data = self.hash_password(password)
        
        # Create user record
        self.users[username] = {
            'username': username,
            'email': email,
            'password_hash': password_data['hash'],
            'salt': password_data['salt'],
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'total_sessions': 0,
            'preferences': {
                'theme': 'light',
                'notifications': True,
                'privacy_mode': True
            }
        }
        
        self.save_users()
        return {'success': True, 'message': 'User registered successfully'}
    
    def login(self, username: str, password: str) -> Dict:
        """Authenticate user and create session"""
        if username not in self.users:
            return {'success': False, 'error': 'Invalid username or password'}
        
        user = self.users[username]
        
        # Verify password
        if not self.verify_password(password, user['password_hash'], user['salt']):
            return {'success': False, 'error': 'Invalid username or password'}
        
        # Create session token
        session_token = secrets.token_urlsafe(32)
        expiry = datetime.now() + timedelta(hours=24)
        
        self.sessions[session_token] = (username, expiry)
        
        # Update user stats
        user['last_login'] = datetime.now().isoformat()
        user['total_sessions'] += 1
        self.save_users()
        
        return {
            'success': True,
            'session_token': session_token,
            'expires_at': expiry.isoformat(),
            'username': username,
            'preferences': user['preferences']
        }
    
    def validate_session(self, session_token: str) -> Optional[str]:
        """Validate session token and return username if valid"""
        if session_token not in self.sessions:
            return None
        
        username, expiry = self.sessions[session_token]
        
        if datetime.now() > expiry:
            # Session expired
            del self.sessions[session_token]
            return None
        
        return username
    
    def logout(self, session_token: str):
        """End user session"""
        if session_token in self.sessions:
            del self.sessions[session_token]
    
    def change_password(self, username: str, old_password: str, new_password: str) -> Dict:
        """Change user password"""
        if username not in self.users:
            return {'success': False, 'error': 'User not found'}
        
        user = self.users[username]
        
        # Verify old password
        if not self.verify_password(old_password, user['password_hash'], user['salt']):
            return {'success': False, 'error': 'Current password is incorrect'}
        
        # Hash new password
        password_data = self.hash_password(new_password)
        
        # Update user record
        user['password_hash'] = password_data['hash']
        user['salt'] = password_data['salt']
        user['password_changed_at'] = datetime.now().isoformat()
        
        self.save_users()
        return {'success': True, 'message': 'Password changed successfully'}
    
    def get_user_stats(self, username: str) -> Dict:
        """Get user statistics (privacy-preserving)"""
        if username not in self.users:
            return {}
        
        user = self.users[username]
        
        # Return only non-sensitive information
        return {
            'username': user['username'],
            'member_since': user['created_at'],
            'last_login': user['last_login'],
            'total_sessions': user['total_sessions'],
            'preferences': user['preferences']
        }
    
    def delete_account(self, username: str, password: str) -> Dict:
        """Permanently delete user account"""
        if username not in self.users:
            return {'success': False, 'error': 'User not found'}
        
        user = self.users[username]
        
        # Verify password
        if not self.verify_password(password, user['password_hash'], user['salt']):
            return {'success': False, 'error': 'Invalid password'}
        
        # Delete user
        del self.users[username]
        
        # Invalidate all sessions for this user
        to_delete = []
        for token, (uname, _) in self.sessions.items():
            if uname == username:
                to_delete.append(token)
        
        for token in to_delete:
            del self.sessions[token]
        
        self.save_users()
        return {'success': True, 'message': 'Account deleted successfully'}