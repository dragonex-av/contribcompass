from __future__ import annotations

import dataclasses
import hashlib
import html
import importlib
import io
import json
import os
import secrets
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src import rag_engine
from src.evaluation import evaluate
from src.workspace import WorkspaceStore
from src.github_auth import GitHubAuth, GitHubAuthError, GitHubConfig

ROOT = Path(__file__).parent
DATA = Path(os.getenv('CONTRIBCOMPASS_DATA_DIR', str(ROOT / '.data')))
DATA.mkdir(exist_ok=True)
store = WorkspaceStore(DATA / 'workspaces.sqlite3')
st.set_page_config(page_title='ContribCompass · Project copilot', page_icon='✳', layout='wide')
st.markdown(f'<style>{(ROOT / "static/app.css").read_text()}</style>', unsafe_allow_html=True)
github_config = GitHubConfig.from_env()
github_auth = GitHubAuth(DATA / 'workspaces.sqlite3', github_config)

# GitHub opens in a separate tab. A redirect verifies the account but cannot
# sign in that tab: the initiating tab must redeem its session-only claim secret.
if 'code' in st.query_params or 'error' in st.query_params:
    callback_state = st.query_params.get('state', '')
    callback_code = st.query_params.get('code', '')
    callback_denied = 'error' in st.query_params
    st.query_params.clear()
    st.title('GitHub sign-in')
    try:
        with st.spinner('Verifying your GitHub account…'):
            approved = github_auth.complete_callback(callback_state, callback_code, callback_denied)
        if approved:
            st.success('GitHub verified. Return to your original ContribCompass tab and click Finish sign-in.')
        else:
            st.info('Sign-in was cancelled. Return to your original tab to try again.')
    except GitHubAuthError as error:
        st.error(str(error))
    except Exception:
        st.error('Sign-in could not be completed. Return to the original tab and start again.')
    st.caption('You can close this tab. Your workspace opens in the tab where you started.')
    st.stop()


def fresh_chat():
    chat_id = secrets.token_hex(16)
    chat = {'id': chat_id, 'title': 'New conversation', 'messages': [], 'updated': time.time()}
    st.session_state.chats[chat_id] = chat
    st.session_state.active_chat = chat_id
    return chat


if 'workspace_session' not in st.session_state:
    st.session_state.workspace_session = secrets.token_hex(16)
    st.session_state.chats = {}
    st.session_state.user = None
    fresh_chat()
if 'active_chat' not in st.session_state or st.session_state.active_chat not in st.session_state.chats:
    fresh_chat()

user = st.session_state.user
owner = user['id'] if user else st.session_state.workspace_session
upload_dir = DATA / 'documents' / owner
upload_dir.mkdir(parents=True, exist_ok=True)
chat = st.session_state.chats[st.session_state.active_chat]
api_key = st.session_state.get('model_api_key', '') or os.getenv('GOOGLE_API_KEY', '')
for preference, default in [('corpus', 'FastAPI demo'), ('chat_mode', 'Ask docs')]:
    st.session_state[preference] = st.session_state.get(preference, default)


def persist():
    chat['updated'] = time.time()
    if user:
        store.save(user['id'], chat)


def switch_identity(identity):
    # Credentials, documents, conversation state and results never cross accounts.
    for key in list(st.session_state):
        del st.session_state[key]
    st.session_state.user = identity
    st.session_state.workspace_session = secrets.token_hex(16)
    st.session_state.chats = store.chats(identity['id']) if identity else {}
    if st.session_state.chats:
        st.session_state.active_chat = next(iter(st.session_state.chats))
    else:
        fresh_chat()
    st.rerun()


def pack(response):
    return json.loads(json.dumps(response, default=lambda x: dataclasses.asdict(x) if dataclasses.is_dataclass(x) else str(x)))


def fingerprint(folder):
    return '|'.join(f'{p.name}:{p.stat().st_mtime_ns}:{p.stat().st_size}' for p in sorted(folder.iterdir()) if p.is_file())


@st.cache_resource(show_spinner=False, max_entries=12)
def get_engine(scope, document_path, revision, document_stamp, seed_stamp):
    module = importlib.reload(rag_engine)
    corpus = Path(document_path)
    if scope == 'FastAPI demo':
        return module.HybridRAG.from_folder(ROOT / 'documents', source_urls=json.loads((ROOT / 'documents/sources.json').read_text()))
    return module.HybridRAG.from_folder(corpus)


def select_chat(chat_id):
    st.session_state.active_chat = chat_id
    st.session_state.page = 'Chat'


def open_account():
    st.session_state.page = 'Settings'


with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-icon">✳</span> ContribCompass</div>', unsafe_allow_html=True)
    if st.button('＋  New chat', use_container_width=True, type='primary', key='new_chat'):
        fresh_chat()
        st.session_state.page = 'Chat'
        st.rerun()
    page = st.radio('Workspace navigation', ['Chat', 'Documents', 'Evaluation', 'Settings'], key='page', label_visibility='collapsed')
    st.markdown('<div class="section-label">Your conversations</div>', unsafe_allow_html=True)
    search = st.text_input('Search conversations', placeholder='⌕  Search chats', label_visibility='collapsed')
    histories = sorted(st.session_state.chats.values(), key=lambda c: c['updated'], reverse=True)
    for item in histories:
        if search.lower() in item['title'].lower():
            st.button(('◉  ' if item['id'] == chat['id'] else '◌  ') + item['title'][:38],
                      key='chat_' + item['id'], use_container_width=True,
                      on_click=select_chat, args=(item['id'],))
    st.markdown('<div class="account-note">' + (f'● {html.escape(user["username"])} · saved workspace' if user else '◌ Guest workspace · session history') + '</div>', unsafe_allow_html=True)
    if not user:
        st.button('Sign in with GitHub', key='github_sidebar', on_click=open_account, use_container_width=True)
        st.caption('Sign in to keep your chats between visits.')
    elif st.button('Sign out', key='sign_out'):
        switch_identity(None)


def evidence(response, key):
    with st.expander(f"Sources · {len(response.get('results', []))} passages", expanded=False):
        st.caption('Similarity measures retrieval relevance, not answer correctness.')
        st.caption('Search: ' + response.get('query', ''))
        for i, item in enumerate(response.get('results', []), 1):
            chunk = item['chunk']
            st.markdown(f"**[S{i}] {chunk['source']} · {chunk['heading']}**")
            st.text(chunk['text'])
            if chunk.get('source_url', '').startswith(('https://', 'http://')):
                st.link_button('Open original source', chunk['source_url'])
        st.caption(f"Relevance {response.get('relevance', 0):.2f} · {response.get('latency_ms', 0)/1000:.2f}s")


def render_response(response, key):
    st.markdown(response['answer'])
    report = response.get('impact')
    if report:
        st.caption(report['code_kind'])
        st.code(report['code'], language=report['language'])
        st.markdown('#### What changes in your project')
        st.caption('Predicted from documentation. This code has not been executed or benchmarked.')
        if report['effects']:
            st.dataframe(pd.DataFrame(report['effects']).rename(columns={'area':'Affected area','before':'Before','after':'Expected after','evidence':'Source'}), hide_index=True, use_container_width=True)
        risks, checks = st.columns(2)
        with risks:
            st.markdown('**Risks & trade-offs**')
            for line in report['risks']:
                st.markdown('• ' + line)
        with checks:
            st.markdown('**How to verify**')
            for line in report['checks']:
                st.markdown('• ' + line)
        if report['assumptions']:
            with st.expander('Assumptions & missing context'):
                for line in report['assumptions']:
                    st.markdown('• ' + line)
    st.caption(response['mode'])
    evidence(response, key)


if page == 'Settings':
    st.title('Make yourself at home.')
    st.caption('Your account, model connection, and retrieval preferences.')
    account_tab, connection_tab, retrieval_tab = st.tabs(['Account', 'Model connection', 'Retrieval'])
    with account_tab:
        if user:
            st.success(f"Signed in as {user['username']}. Chats and documents stay in your local workspace.")
            if user.get('provider') == 'github':
                st.caption('Connected with GitHub. Signing out ends this app session.')
        else:
            st.subheader('Continue with GitHub')
            st.caption('Use your GitHub identity for a personal workspace. Only public profile access is requested.')
            if not github_config.configured:
                st.button('Continue with GitHub', disabled=True, key='github_start')
                st.info('GitHub sign-in is ready in the app, but the app owner must configure the OAuth credentials first.')
                with st.expander('App owner setup'):
                    st.markdown('Create a GitHub OAuth app. Set its homepage and authorization callback URL to `http://localhost:8501/` for this local demo. Add these values to `.env`, then restart the app.')
                    st.code('GITHUB_CLIENT_ID=your_client_id\nGITHUB_CLIENT_SECRET=your_client_secret\nGITHUB_REDIRECT_URI=http://localhost:8501/', language='bash')
                    st.link_button('Register a GitHub OAuth app', 'https://github.com/settings/applications/new')
            else:
                flow = st.session_state.get('github_flow')
                if flow is None:
                    if st.button('Continue with GitHub', type='primary', key='github_start'):
                        try:
                            st.session_state.github_flow = github_auth.begin()
                            st.rerun()
                        except GitHubAuthError as error:
                            st.error(str(error))
                else:
                    st.link_button('Open GitHub to authorize ↗', flow['url'], type='primary')
                    st.caption('Authorize in the new tab, then return here. This request expires after 10 minutes.')
                    complete, cancel = st.columns(2)
                    with complete:
                        if st.button('Finish sign-in', key='github_finish'):
                            try:
                                profile = github_auth.claim(flow)
                                if profile:
                                    switch_identity(store.github_identity(profile))
                                else:
                                    st.info('Still waiting for GitHub. Complete authorization in the other tab first.')
                            except GitHubAuthError as error:
                                st.error(str(error))
                    with cancel:
                        if st.button('Start again', key='github_cancel'):
                            github_auth.cancel(flow)
                            st.session_state.pop('github_flow', None)
                            st.rerun()
            st.divider()
            st.markdown('**Or use a local account**')
            st.info('Guest chats last for this session. Local accounts save future chats and uploads on this computer. Guest content is not copied when you sign in.')
            with st.form('account_form'):
                action = st.radio('Account action', ['Sign in', 'Create account'], horizontal=True)
                username = st.text_input('Username', max_chars=30)
                password = st.text_input('Password', type='password')
                submitted = st.form_submit_button('Continue', type='primary')
            if submitted:
                blocked_until = st.session_state.get('login_blocked_until', 0)
                if time.time() < blocked_until:
                    st.error('Too many attempts. Wait a minute before trying again.')
                else:
                    try:
                        identity = store.register(username, password) if action == 'Create account' else store.login(username, password)
                        switch_identity(identity)
                    except ValueError as error:
                        st.error(str(error))
                        st.session_state.login_attempts = st.session_state.get('login_attempts', 0) + 1
                        if st.session_state.login_attempts >= 5:
                            st.session_state.login_blocked_until = time.time() + 60
                            st.session_state.login_attempts = 0
    with connection_tab:
        st.subheader('Connect your AI model')
        st.caption('The key stays in this session and is never saved with chats. A server-configured key works automatically.')
        def save_key():
            st.session_state.model_api_key = st.session_state._api_key
        st.text_input('Google AI Studio API key', type='password', key='_api_key', value=st.session_state.get('model_api_key', ''), on_change=save_key, placeholder='Paste your key here')
        st.caption('Model: ' + os.getenv('GEMMA_MODEL', 'gemma-4-26b-a4b-it'))
        left, right = st.columns(2)
        with left:
            if st.button('Test connection', type='primary'):
                key = st.session_state.get('model_api_key') or os.getenv('GOOGLE_API_KEY')
                if not key:
                    st.warning('Enter an API key first.')
                else:
                    with st.spinner('Checking connection…'):
                        try:
                            reply = rag_engine.HybridRAG._generate('Reply with READY.', os.getenv('GEMMA_MODEL', 'gemma-4-26b-a4b-it'), key)
                            if not reply:
                                raise ValueError('Empty model response')
                            st.success('Connected. Generated answers and Code + impact are ready.')
                        except Exception as error:
                            st.error(f'Connection failed ({type(error).__name__}). Check the key, model access, and quota.')
        with right:
            st.link_button('Get a Google AI Studio key ↗', 'https://aistudio.google.com/apikey')
    with retrieval_tab:
        def save_retrieval():
            st.session_state.top_k = st.session_state._top_k
            st.session_state.threshold = st.session_state._threshold
        st.slider('Source passages', 2, 8, st.session_state.get('top_k', 5), key='_top_k', on_change=save_retrieval)
        st.slider('Minimum relevance', .05, .80, st.session_state.get('threshold', .24), .01, key='_threshold', on_change=save_retrieval)
        st.caption('Higher thresholds refuse more often. Scores are similarities, not probabilities.')
    st.stop()

if page == 'Documents':
    st.title('Your project, in context.')
    st.caption('Upload documentation to give the copilot your APIs, architecture, commands, and project conventions.')
    st.info('For the best Code + impact report, include API usage examples, architecture notes, and a description of the behavior you want to change.')
    files = st.file_uploader('Add project documentation', type=['md', 'txt', 'pdf'], accept_multiple_files=True, key='uploads')
    if st.button('Add to my workspace', type='primary', disabled=not files):
        staged = []
        try:
            for upload in files:
                payload = upload.getvalue()
                if len(payload) > 20 * 1024 * 1024:
                    raise ValueError(f'{upload.name}: maximum file size is 20 MB.')
                if upload.name.lower().endswith('.pdf'):
                    from pypdf import PdfReader
                    text = '\n'.join(p.extract_text() or '' for p in PdfReader(io.BytesIO(payload)).pages)
                else:
                    text = payload.decode('utf-8', errors='replace')
                if not text.strip():
                    raise ValueError(f'{upload.name}: no readable text. Scanned PDFs need OCR first.')
                if not any(c.isalnum() for c in text):
                    raise ValueError(f'{upload.name}: add meaningful document text.')
                staged.append((Path(upload.name).name, payload))
            for name, payload in staged:
                (upload_dir / name).write_bytes(payload)
            st.session_state.corpus = 'My documents'
            st.success(f'{len(staged)} document(s) added. Open Chat and ask about your project.')
        except Exception as error:
            st.error(f'Upload could not be completed: {error}')
    st.subheader('My documents')
    own_files = [p for p in sorted(upload_dir.iterdir()) if p.is_file()]
    if not own_files:
        st.caption('No documents yet. The FastAPI demo workspace is available in Chat.')
    for path in own_files:
        name_col, size_col, action_col = st.columns([6, 2, 2])
        name_col.write('▤  ' + path.name)
        size_col.caption(f'{path.stat().st_size / 1024:.1f} KB')
        if action_col.button('Remove', key='remove_' + path.name):
            path.unlink()
            st.rerun()
    with st.expander('Included FastAPI demo sources'):
        for name, url in json.loads((ROOT / 'documents/sources.json').read_text()).items():
            st.markdown(f'[{name}]({url})')
    st.stop()

scope_options = ['FastAPI demo', 'My documents']
header, scope_column = st.columns([3, 2])
with header:
    st.markdown('<div class="topline"><b>Project copilot</b><span>/</span> Understand. Build. Verify.</div>', unsafe_allow_html=True)
with scope_column:
    scope = st.selectbox('Knowledge workspace', scope_options, key='corpus', label_visibility='collapsed')
if scope == 'My documents' and not any(upload_dir.iterdir()):
    st.info('Your document workspace is empty. Open Documents to add your project docs, or select FastAPI demo.')
    st.stop()
try:
    with st.spinner('Preparing your knowledge workspace…'):
        engine = get_engine(scope, str(upload_dir), hashlib.sha256(Path(rag_engine.__file__).read_bytes()).hexdigest(), fingerprint(upload_dir), fingerprint(ROOT / 'documents'))
except Exception as error:
    st.error(f'Could not index this workspace ({type(error).__name__}). Check your documents and try again.')
    st.stop()
top_k, threshold = st.session_state.get('top_k', 5), st.session_state.get('threshold', .24)

if page == 'Evaluation':
    st.title('Evidence over guesswork.')
    st.caption('Benchmark the demo corpus, compare real model outputs, and plan your first contribution.')
    benchmark, compare, journey = st.tabs(['Quality lab', 'RAG comparison', 'Onboarding'])
    with benchmark:
        if scope != 'FastAPI demo':
            st.warning('This benchmark expects FastAPI facts. Select FastAPI demo for meaningful scores.')
        live = st.checkbox('Evaluate generated answers (18 API calls)')
        st.caption('Default runs measure retrieval and refusal gating. A rubric pass is not proof of semantic correctness.')
        if st.button('Run evaluation', key='evaluate', type='primary', disabled=scope != 'FastAPI demo'):
            with st.spinner('Checking 18 expected-answer cases…'):
                st.session_state.results = evaluate(engine, top_k, threshold, generate=live, api_key=api_key)
        if 'results' in st.session_state:
            df = pd.DataFrame(st.session_state.results)
            st.metric('Rubric pass rate', f'{(df.Outcome == "Pass").mean():.1%}')
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.download_button('Export results', df.to_csv(index=False), 'evaluation.csv')
        if st.button('Run experiment matrix', key='run_matrix', disabled=scope != 'FastAPI demo'):
            rows = []
            with st.spinner('Comparing retrieval settings…'):
                for k in (3, 5, 7):
                    for gate in (.15, .24, .30):
                        result = evaluate(engine, k, gate)
                        rows.append({'Top-k':k, 'Threshold':gate, 'Passed':sum(r['Outcome']=='Pass' for r in result), 'Total':len(result)})
            st.session_state.matrix = rows
        if 'matrix' in st.session_state:
            st.dataframe(st.session_state.matrix, hide_index=True, use_container_width=True)
    with compare:
        question = st.text_input('Question to compare', value='What coding style rules should I follow?')
        if st.button('Compare answers', key='compare') and question.strip():
            with st.spinner('Generating comparison…'):
                st.session_state.comparison = pack(engine.compare_rag_vs_no_rag(question, top_k, threshold, api_key=api_key))
        if 'comparison' in st.session_state:
            result = st.session_state.comparison
            st.caption(result['summary'])
            a, b = st.columns(2)
            with a:
                st.subheader('Without retrieval')
                st.markdown(result['no_rag']['answer'])
                st.caption(result['no_rag']['mode'])
            with b:
                st.subheader('With evidence')
                render_response(result['rag'], 'compare')
    with journey:
        if st.button('Build my onboarding path', key='build_journey', type='primary'):
            with st.spinner('Building your first five steps…'):
                st.session_state.journey = engine.onboarding_checklist(api_key=api_key)
        for i, step in enumerate(st.session_state.get('journey', [])):
            with st.expander(f'{i+1}. {step["title"]}'):
                st.checkbox('Completed', key=f'step_{i}')
                st.markdown(step['guidance'])
    st.stop()

mode_col, menu_col = st.columns([4, 1])
with mode_col:
    mode = st.radio('Response mode', ['Ask docs', 'Code + impact'], horizontal=True, key='chat_mode', label_visibility='collapsed')
with menu_col:
    with st.popover('•••'):
        new_title = st.text_input('Conversation title', value=chat['title'], key='rename_' + chat['id'])
        if st.button('Rename chat') and new_title.strip():
            chat['title'] = new_title.strip()[:80]
            persist()
            st.rerun()
        transcript = '\n\n'.join(m['role'].upper() + '\n' + m['content'] for m in chat['messages'])
        st.download_button('Export conversation', transcript, 'conversation.txt')
        if st.button('Delete conversation', key='delete_chat'):
            if user:
                store.delete(user['id'], chat['id'])
            del st.session_state.chats[chat['id']]
            fresh_chat()
            st.rerun()

def choose_suggestion(question, impact):
    st.session_state.pending_question = question
    if impact:
        st.session_state.chat_mode = 'Code + impact'


picked = st.session_state.pop('pending_question', None)
if not chat['messages']:
    st.markdown('''<div class="hero"><div class="spark">✳</div><h1>Understand the code.<br>See what changes.</h1><p>Your docs become a project copilot. Get grounded answers,<br>useful code, and a clearer picture of its impact.</p><small>Built on your sources. Designed for your next step.</small></div>''', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;margin-bottom:28px"><span class="tag">↗ Source-backed answers</span><span class="tag">⌘ Code + impact</span></div>', unsafe_allow_html=True)
    suggestions = [('Set up my project', 'How do I install requirements for contributing?'),
                   ('Understand a code change', 'Show the documented setup commands and explain their effect on the development environment.'),
                   ('Follow the coding standards', 'What coding style rules should I follow?'),
                   ('Plan my first contribution', 'What should I do before sending a pull request?')]
    for row in range(2):
        cols = st.columns(2)
        for offset in range(2):
            index = row * 2 + offset
            cols[offset].button(suggestions[index][0] + '  ↗', key=f'q{index}', use_container_width=True,
                                on_click=choose_suggestion, args=(suggestions[index][1], index == 1))
    if not api_key:
        st.caption('Evidence mode is ready. Connect a model in Settings for generated code and impact predictions.')
else:
    for index, message in enumerate(chat['messages']):
        with st.chat_message(message['role'], avatar='✳' if message['role']=='assistant' else None):
            if message.get('response'):
                render_response(message['response'], str(index))
            else:
                st.markdown(message['content'])
            if message.get('workspace'):
                st.caption(message['workspace'])

question = st.chat_input('Ask about your project, or describe a change you want to make…') or picked
if question and question.strip():
    history = [{'role':m['role'], 'content':m['content']} for m in chat['messages'] if m.get('workspace', scope) == scope]
    chat['messages'].append({'role':'user', 'content':question, 'workspace':scope})
    if chat['title'] == 'New conversation':
        chat['title'] = question[:55]
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant', avatar='✳'):
        with st.spinner('Reading your sources and connecting the details…'):
            try:
                if mode == 'Code + impact':
                    response = engine.code_impact(question, top_k, threshold, history=history, api_key=api_key)
                else:
                    response = engine.answer(question, top_k, threshold, history=history, api_key=api_key)
                response = pack(response)
            except Exception as error:
                response = {'answer':'This request could not be completed. Try rephrasing it or check your model connection in Settings.', 'mode':f'Request error · {type(error).__name__}', 'results':[], 'refused':True}
        render_response(response, 'latest')
    # Save the full report as conversational content so exports retain code and impact.
    content = response['answer']
    if response.get('impact'):
        report = response['impact']
        content += '\n\n```' + report['language'] + '\n' + report['code'] + '\n```\n\nPredicted impact:\n' + json.dumps(report['effects'], indent=2)
        content += '\n\nRisks:\n' + '\n'.join(report['risks']) + '\n\nVerification:\n' + '\n'.join(report['checks'])
    chat['messages'].append({'role':'assistant', 'content':content, 'response':response, 'workspace':scope})
    persist()
    st.rerun()
