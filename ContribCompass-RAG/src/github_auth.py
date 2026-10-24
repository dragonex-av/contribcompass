"""GitHub OAuth web flow with PKCE and an original-tab-bound login handoff.

Streamlit starts a new session after an external redirect. The callback therefore
verifies GitHub and marks a short-lived flow ready; only the initiating tab's
secret can claim that identity. Neither OAuth tokens nor client secrets are stored.
"""
from __future__ import annotations
import base64
import hashlib
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode, urlparse

import httpx


class GitHubAuthError(ValueError):
    pass


@dataclass(frozen=True)
class GitHubConfig:
    client_id: str
    client_secret: str = field(repr=False)
    redirect_uri: str

    @classmethod
    def from_env(cls):
        return cls(os.getenv('GITHUB_CLIENT_ID', '').strip(),
                   os.getenv('GITHUB_CLIENT_SECRET', '').strip(),
                   os.getenv('GITHUB_REDIRECT_URI', 'http://localhost:8501/').strip())

    @property
    def configured(self):
        return bool(self.client_id and self.client_secret)

    def validate(self):
        if not self.configured:
            raise GitHubAuthError('GitHub sign-in has not been configured by the app owner yet.')
        url = urlparse(self.redirect_uri)
        local = url.hostname in {'localhost', '127.0.0.1', '::1'}
        if not url.hostname or url.username or url.password or url.query or url.fragment or (url.scheme != 'https' and not (local and url.scheme == 'http')):
            raise GitHubAuthError('Use an HTTPS callback URL, or HTTP on localhost for local development.')


class GitHubAuth:
    TTL = 600

    def __init__(self, path: Path, config: GitHubConfig, transport=None):
        self.path, self.config, self.transport = path, config, transport
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS github_oauth_flows (
                state_hash TEXT PRIMARY KEY, claim_hash TEXT NOT NULL,
                verifier TEXT NOT NULL, expires REAL NOT NULL, status TEXT NOT NULL,
                github_id INTEGER, github_login TEXT)''')

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def digest(value):
        return hashlib.sha256(value.encode()).hexdigest()

    def begin(self):
        self.config.validate()
        state, claim, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(32), secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
        expires = time.time() + self.TTL
        with self.connect() as db:
            db.execute('DELETE FROM github_oauth_flows WHERE expires < ?', (time.time(),))
            db.execute('INSERT INTO github_oauth_flows VALUES (?,?,?,?,?,NULL,NULL)',
                       (self.digest(state), self.digest(claim), verifier, expires, 'pending'))
        url = 'https://github.com/login/oauth/authorize?' + urlencode({
            'client_id':self.config.client_id, 'redirect_uri':self.config.redirect_uri,
            'state':state, 'code_challenge':challenge, 'code_challenge_method':'S256',
            'scope':'', 'prompt':'select_account'})
        return {'state':state, 'claim':claim, 'expires':expires, 'url':url}

    def cancel(self, flow):
        with self.connect() as db:
            db.execute('DELETE FROM github_oauth_flows WHERE state_hash=? AND claim_hash=?',
                       (self.digest(flow['state']), self.digest(flow['claim'])))

    def _request(self, client, method, url, **kwargs):
        try:
            response = client.request(method, url, **kwargs)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError('Not an object')
            return payload
        except (httpx.HTTPError, ValueError):
            # Do not surface response bodies, request parameters or tokens.
            raise GitHubAuthError('GitHub could not complete sign-in. Start again and check the OAuth app configuration or connection.') from None

    def complete_callback(self, state, code=None, denied=False):
        self.config.validate()
        if not state or len(state) > 256 or (not denied and (not code or len(code) > 1024)):
            raise GitHubAuthError('Invalid GitHub callback. Start sign-in again.')
        state_hash = self.digest(state)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT verifier, expires, status FROM github_oauth_flows WHERE state_hash=?', (state_hash,)).fetchone()
            if not row or row[1] <= time.time() or row[2] != 'pending':
                raise GitHubAuthError('This sign-in request has expired or was already used. Start again from the original tab.')
            db.execute('UPDATE github_oauth_flows SET status=? WHERE state_hash=?', ('denied' if denied else 'exchanging', state_hash))
        if denied:
            return False
        try:
            with httpx.Client(timeout=20, follow_redirects=False, transport=self.transport,
                              headers={'Accept':'application/json', 'User-Agent':'ContribCompass'}) as client:
                result = self._request(client, 'POST', 'https://github.com/login/oauth/access_token', data={
                    'client_id':self.config.client_id, 'client_secret':self.config.client_secret,
                    'code':code, 'redirect_uri':self.config.redirect_uri, 'code_verifier':row[0]})
                token = result.get('access_token')
                if result.get('error') or not isinstance(token, str) or not token:
                    raise GitHubAuthError('GitHub rejected this sign-in. Start again and verify the OAuth app settings.')
                profile = self._request(client, 'GET', 'https://api.github.com/user', headers={
                    'Authorization':'Bearer ' + token, 'Accept':'application/vnd.github+json',
                    'X-GitHub-Api-Version':'2022-11-28'})
            github_id, login = profile.get('id'), profile.get('login')
            if type(github_id) is not int or github_id <= 0 or not isinstance(login, str) or not 1 <= len(login) <= 100 or profile.get('type') != 'User':
                raise GitHubAuthError('GitHub did not return a valid personal account.')
            with self.connect() as db:
                updated = db.execute('UPDATE github_oauth_flows SET status=?,verifier=?,github_id=?,github_login=? WHERE state_hash=? AND status=? AND expires>?',
                                     ('ready', '', github_id, login, state_hash, 'exchanging', time.time()))
                if not updated.rowcount:
                    raise GitHubAuthError('This sign-in was cancelled or expired. Start again.')
            return True
        except Exception:
            with self.connect() as db:
                db.execute('UPDATE github_oauth_flows SET status=?,verifier=? WHERE state_hash=?', ('failed', '', state_hash))
            raise

    def claim(self, flow):
        """Single-use completion: knowing the callback URL alone cannot log in."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT claim_hash,expires,status,github_id,github_login FROM github_oauth_flows WHERE state_hash=?', (self.digest(flow['state']),)).fetchone()
            if not row or not secrets.compare_digest(row[0], self.digest(flow['claim'])):
                raise GitHubAuthError('The sign-in session is invalid. Start again.')
            if row[1] <= time.time():
                raise GitHubAuthError('GitHub sign-in expired. Start again.')
            if row[2] in {'pending', 'exchanging'}:
                return None
            if row[2] != 'ready':
                raise GitHubAuthError('GitHub sign-in was declined or failed. Start again.')
            db.execute('DELETE FROM github_oauth_flows WHERE state_hash=?', (self.digest(flow['state']),))
        return {'id':row[3], 'login':row[4]}
