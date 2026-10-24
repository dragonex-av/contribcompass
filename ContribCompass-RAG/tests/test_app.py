import sys
from pathlib import Path
import streamlit as st
from streamlit.testing.v1 import AppTest
from src import rag_engine


def test_chat_workspace_and_tools(monkeypatch, tmp_path):
    def old_answer(self, question, top_k=5, threshold=.24):
        raise AssertionError('Stale engine must reload')
    monkeypatch.setattr(rag_engine.HybridRAG, 'answer', old_answer)
    monkeypatch.setitem(sys.modules, 'sentence_transformers', None)
    monkeypatch.setenv('GOOGLE_API_KEY', '')
    monkeypatch.setenv('CONTRIBCOMPASS_DATA_DIR', str(tmp_path))
    st.cache_resource.clear()
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run(timeout=30)
    assert not app.exception
    first_id = app.session_state.active_chat
    app.button(key='q0').click().run(timeout=30)
    assert not app.exception
    messages = app.session_state.chats[first_id]['messages']
    assert [m['role'] for m in messages] == ['user', 'assistant']
    app.chat_input[0].set_value('Can this app book my flight?').run(timeout=30)
    assert not app.exception
    assert app.session_state.chats[first_id]['messages'][-1]['response']['refused']
    app.button(key='new_chat').click().run()
    assert not app.exception
    assert app.session_state.active_chat != first_id
    app.button(key='chat_' + first_id).click().run()
    assert not app.exception
    assert app.session_state.active_chat == first_id
    app.radio(key='chat_mode').set_value('Code + impact').run()
    app.chat_input[0].set_value('How do I install requirements for contributing?').run(timeout=30)
    assert not app.exception
    assert 'Code evidence' in app.session_state.chats[first_id]['messages'][-1]['response']['mode']
    app.radio(key='page').set_value('Evaluation').run(timeout=30)
    app.button(key='compare').click().run(timeout=30)
    assert not app.exception
    assert not app.session_state.comparison['no_rag']['available']
    app.button(key='build_journey').click().run(timeout=30)
    assert not app.exception
    assert len(app.session_state.journey) == 5
    app.button(key='evaluate').click().run(timeout=30)
    assert not app.exception
    assert len(app.session_state.results) == 18
    app.button(key='run_matrix').click().run(timeout=30)
    assert not app.exception
    assert len(app.session_state.matrix) == 9
    app.radio(key='page').set_value('Settings').run()
    app.text_input(key='_api_key').set_value('test-session-key').run()
    app.radio(key='page').set_value('Documents').run()
    assert not app.exception
    app.radio(key='page').set_value('Chat').run()
    assert app.session_state.model_api_key == 'test-session-key'
    assert not app.exception


def test_local_account_saves_and_restores_chats(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'sentence_transformers', None)
    monkeypatch.setenv('GOOGLE_API_KEY', '')
    monkeypatch.setenv('CONTRIBCOMPASS_DATA_DIR', str(tmp_path))
    st.cache_resource.clear()
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run(timeout=30)
    app.radio(key='page').set_value('Settings').run()
    next(r for r in app.radio if r.label == 'Account action').set_value('Create account')
    next(t for t in app.text_input if t.label == 'Username').set_value('test_user')
    next(t for t in app.text_input if t.label == 'Password').set_value('test-long-password')
    next(b for b in app.button if b.label == 'Continue').click().run(timeout=30)
    assert not app.exception
    assert app.session_state.user['username'] == 'test_user'
    app.button(key='q0').click().run(timeout=30)
    saved_id = app.session_state.active_chat
    assert len(app.session_state.chats[saved_id]['messages']) == 2
    app.button(key='sign_out').click().run()
    assert not app.exception
    assert app.session_state.user is None
    assert saved_id not in app.session_state.chats
    app.radio(key='page').set_value('Settings').run()
    next(t for t in app.text_input if t.label == 'Username').set_value('test_user')
    next(t for t in app.text_input if t.label == 'Password').set_value('test-long-password')
    next(b for b in app.button if b.label == 'Continue').click().run(timeout=30)
    assert not app.exception
    assert saved_id in app.session_state.chats
    assert len(app.session_state.chats[saved_id]['messages']) == 2


def test_code_impact_report_renders(monkeypatch, tmp_path):
    import json
    monkeypatch.setitem(sys.modules, 'sentence_transformers', None)
    monkeypatch.setenv('GOOGLE_API_KEY', '')
    monkeypatch.setenv('CONTRIBCOMPASS_DATA_DIR', str(tmp_path))
    st.cache_resource.clear()
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run(timeout=30)
    report = {'supported':True, 'summary':'Install the dependencies [S1].', 'code':'uv sync', 'language':'bash',
              'code_kind':'exact source example', 'effects':[{'area':'Environment', 'before':'Unknown', 'after':'Dependencies installed', 'evidence':'[S1]'}],
              'risks':['Version compatibility.'], 'checks':['Run the tests.'], 'assumptions':['Repository not executed.']}
    monkeypatch.setattr(rag_engine.HybridRAG, '_generate', staticmethod(lambda *args: json.dumps(report)))
    app.session_state.model_api_key = 'test-only'
    app.radio(key='chat_mode').set_value('Code + impact').run()
    app.chat_input[0].set_value('How do I install requirements for contributing?').run(timeout=30)
    assert not app.exception
    assert any(code.value == 'uv sync' for code in app.code)
    assert len(app.dataframe) == 1
    messages = app.session_state.chats[app.session_state.active_chat]['messages']
    assert 'Predicted impact' in messages[-1]['content']
    assert messages[-1]['response']['impact']['effects'][0]['area'] == 'Environment'


def test_github_callback_returns_identity_only_to_original_tab(monkeypatch, tmp_path):
    from src.github_auth import GitHubAuth
    monkeypatch.setitem(sys.modules, 'sentence_transformers', None)
    monkeypatch.setenv('GOOGLE_API_KEY', '')
    monkeypatch.setenv('GITHUB_CLIENT_ID', 'test-client')
    monkeypatch.setenv('GITHUB_CLIENT_SECRET', 'test-secret')
    monkeypatch.setenv('GITHUB_REDIRECT_URI', 'http://localhost:8501/')
    monkeypatch.setenv('CONTRIBCOMPASS_DATA_DIR', str(tmp_path))
    def provider(self, client, method, url, **kwargs):
        if url.endswith('/access_token'):
            return {'access_token':'temporary-test-token'}
        assert kwargs['headers']['Authorization'] == 'Bearer temporary-test-token'
        return {'id':1234, 'login':'github-tester', 'type':'User'}
    monkeypatch.setattr(GitHubAuth, '_request', provider)
    st.cache_resource.clear()
    path = str(Path(__file__).parents[1] / 'app.py')
    original = AppTest.from_file(path).run(timeout=30)
    original.button(key='github_sidebar').click().run()
    original.button(key='github_start').click().run()
    assert not original.exception
    flow = original.session_state.github_flow
    callback = AppTest.from_file(path)
    callback.query_params['state'] = flow['state']
    callback.query_params['code'] = 'valid-test-code'
    callback.run()
    assert not callback.exception
    assert any('GitHub verified' in s.value for s in callback.success)
    assert not callback.query_params
    # Verification in a fresh tab doesn't itself authenticate that browser session.
    assert 'user' not in callback.session_state
    original.button(key='github_finish').click().run(timeout=30)
    assert not original.exception
    assert original.session_state.user['provider'] == 'github'
    assert original.session_state.user['username'] == 'github-tester'
    assert 'github_flow' not in original.session_state
    original.button(key='sign_out').click().run()
    assert original.session_state.user is None
