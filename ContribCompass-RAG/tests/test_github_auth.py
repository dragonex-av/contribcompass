from urllib.parse import parse_qs, urlparse
import hashlib
import base64
import sqlite3
import httpx
import pytest

from src.github_auth import GitHubAuth, GitHubAuthError, GitHubConfig
from src.workspace import WorkspaceStore


@pytest.fixture
def config():
    return GitHubConfig('test-client', 'test-secret', 'http://localhost:8501/')


def api_transport(seen, invalid_profile=False, token_error=False):
    def handler(request):
        seen.append(request)
        if request.url.path == '/login/oauth/access_token':
            return httpx.Response(200, json={'error':'bad_verification_code'} if token_error else {'access_token':'private-token'})
        assert request.url == 'https://api.github.com/user'
        assert request.headers['Authorization'] == 'Bearer private-token'
        return httpx.Response(200, json={'id':42, 'login':'octocat', 'type':'Organization' if invalid_profile else 'User'})
    return httpx.MockTransport(handler)


def test_oauth_round_trip_pkce_binding_and_single_use(tmp_path, config):
    seen = []
    path = tmp_path / 'auth.db'
    auth = GitHubAuth(path, config, api_transport(seen))
    flow = auth.begin()
    params = parse_qs(urlparse(flow['url']).query, keep_blank_values=True)
    assert params['scope'] == ['']
    assert params['code_challenge_method'] == ['S256']
    assert params['redirect_uri'] == [config.redirect_uri]
    assert config.client_secret not in flow['url']
    assert auth.claim(flow) is None
    with pytest.raises(GitHubAuthError):
        auth.complete_callback('forged-state', 'code')
    assert not seen
    assert auth.complete_callback(flow['state'], 'valid-code')
    token_form = parse_qs(seen[0].content.decode())
    verifier = token_form['code_verifier'][0]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
    assert challenge == params['code_challenge'][0]
    with pytest.raises(GitHubAuthError):
        auth.claim(dict(flow, claim='different-browser'))
    assert auth.claim(flow) == {'id':42, 'login':'octocat'}
    with pytest.raises(GitHubAuthError):
        auth.claim(flow)
    with pytest.raises(GitHubAuthError):
        auth.complete_callback(flow['state'], 'valid-code')
    assert b'private-token' not in path.read_bytes()
    assert b'test-secret' not in path.read_bytes()


def test_expired_cancelled_and_denied_flows(tmp_path, config):
    seen = []
    auth = GitHubAuth(tmp_path/'auth.db', config, api_transport(seen))
    expired = auth.begin()
    with auth.connect() as db:
        db.execute('UPDATE github_oauth_flows SET expires=0')
    with pytest.raises(GitHubAuthError):
        auth.complete_callback(expired['state'], 'code')
    flow = auth.begin()
    assert auth.complete_callback(flow['state'], denied=True) is False
    with pytest.raises(GitHubAuthError):
        auth.claim(flow)
    cancelled = auth.begin()
    auth.cancel(cancelled)
    with pytest.raises(GitHubAuthError):
        auth.complete_callback(cancelled['state'], 'code')
    assert not seen


@pytest.mark.parametrize('invalid_profile, token_error', [(True,False),(False,True)])
def test_bad_provider_response_never_authenticates(tmp_path, config, invalid_profile, token_error):
    auth = GitHubAuth(tmp_path/'auth.db', config, api_transport([], invalid_profile, token_error))
    flow = auth.begin()
    with pytest.raises(GitHubAuthError):
        auth.complete_callback(flow['state'], 'code')
    with pytest.raises(GitHubAuthError):
        auth.claim(flow)


def test_configuration_and_network_errors_are_safe(tmp_path, config):
    with pytest.raises(GitHubAuthError):
        GitHubAuth(tmp_path/'none.db', GitHubConfig('', '', 'http://localhost:8501/')).begin()
    with pytest.raises(GitHubAuthError):
        GitHubConfig('id', 'secret', 'http://example.com').validate()
    def offline(request):
        raise httpx.ConnectError('internal details private-token')
    auth = GitHubAuth(tmp_path/'auth.db', config, httpx.MockTransport(offline))
    flow = auth.begin()
    with pytest.raises(GitHubAuthError) as error:
        auth.complete_callback(flow['state'], 'code')
    assert 'private-token' not in str(error.value)


def test_github_identity_uses_id_and_does_not_link_matching_local_username(tmp_path):
    store = WorkspaceStore(tmp_path/'users.db')
    local = store.register('octocat', 'local-password-123')
    github = store.github_identity({'id':42, 'login':'octocat'})
    assert github['id'] != local['id']
    store.save(github['id'], {'id':'chat', 'title':'My work', 'messages':[]})
    renamed = store.github_identity({'id':42, 'login':'new-name'})
    assert renamed['id'] == github['id']
    assert renamed['username'] == 'new-name'
    assert 'chat' in store.chats(renamed['id'])
    assert not store.chats(local['id'])
    assert store.github_identity({'id':43, 'login':'octocat'})['id'] != github['id']
    with pytest.raises(ValueError):
        store.login('github:42', '')
