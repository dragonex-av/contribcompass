"""Local account and chat storage. Every query is scoped to an authenticated owner."""
from __future__ import annotations
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import time
from pathlib import Path


class WorkspaceStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL,
                salt TEXT NOT NULL, password_hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
                messages TEXT NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS github_identities (
                github_id INTEGER PRIMARY KEY, user_id TEXT UNIQUE NOT NULL,
                login TEXT NOT NULL);
            ''')

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def password_hash(password, salt):
        return hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), 600000).hex()

    def register(self, username, password):
        username = username.strip().lower()
        if not re.fullmatch(r'[a-z0-9_]{3,30}', username):
            raise ValueError('Use 3–30 letters, numbers, or underscores for your username.')
        if len(password) < 10:
            raise ValueError('Use a password with at least 10 characters.')
        user_id, salt = secrets.token_hex(16), secrets.token_hex(16)
        try:
            with self.connect() as db:
                db.execute('INSERT INTO users VALUES (?, ?, ?, ?)',
                           (user_id, username, salt, self.password_hash(password, salt)))
        except sqlite3.IntegrityError:
            raise ValueError('That username is already in use.') from None
        return {'id': user_id, 'username': username}

    def login(self, username, password):
        with self.connect() as db:
            row = db.execute('SELECT id, username, salt, password_hash FROM users WHERE username=?',
                             (username.strip().lower(),)).fetchone()
        # Do the expensive hash even for unknown usernames.
        actual = self.password_hash(password, row[2] if row else '00' * 16)
        if row and hmac.compare_digest(actual, row[3]):
            return {'id': row[0], 'username': row[1]}
        raise ValueError('Username or password is incorrect.')

    def chats(self, owner):
        with self.connect() as db:
            rows = db.execute('SELECT id,title,messages,updated FROM chats WHERE owner=? ORDER BY updated DESC', (owner,)).fetchall()
        return {r[0]: {'id': r[0], 'title': r[1], 'messages': json.loads(r[2]), 'updated': r[3]} for r in rows}

    def github_identity(self, profile):
        """Only call after GitHub /user verification, never with browser form data."""
        github_id, login = profile.get('id'), profile.get('login')
        if type(github_id) is not int or github_id <= 0 or not isinstance(login, str) or not 1 <= len(login) <= 100:
            raise ValueError('Invalid GitHub identity.')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT user_id FROM github_identities WHERE github_id=?', (github_id,)).fetchone()
            if row:
                user_id = row[0]
                db.execute('UPDATE github_identities SET login=? WHERE github_id=?', (login, github_id))
            else:
                user_id = secrets.token_hex(16)
                # Reserved namespace prevents linking by a coincidentally matching name.
                db.execute('INSERT INTO users VALUES (?,?,?,?)',
                           (user_id, f'github:{github_id}', secrets.token_hex(16), ''))
                db.execute('INSERT INTO github_identities VALUES (?,?,?)', (github_id, user_id, login))
        return {'id':user_id, 'username':login, 'provider':'github', 'github_id':github_id}

    def save(self, owner, chat):
        with self.connect() as db:
            existing = db.execute('SELECT owner FROM chats WHERE id=?', (chat['id'],)).fetchone()
            if existing and existing[0] != owner:
                raise PermissionError('This conversation belongs to a different workspace.')
            db.execute('INSERT INTO chats VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,messages=excluded.messages,updated=excluded.updated',
                       (chat['id'], owner, chat['title'], json.dumps(chat['messages']), time.time()))

    def delete(self, owner, chat_id):
        with self.connect() as db:
            db.execute('DELETE FROM chats WHERE owner=? AND id=?', (owner, chat_id))
