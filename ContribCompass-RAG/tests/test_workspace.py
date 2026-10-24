import pytest
from src.workspace import WorkspaceStore


def test_accounts_and_owner_isolation(tmp_path):
    store = WorkspaceStore(tmp_path / 'app.db')
    alice = store.register('alice', 'correct-horse-battery')
    bob = store.register('bob', 'different-password')
    assert store.login('ALICE', 'correct-horse-battery') == alice
    with pytest.raises(ValueError):
        store.login('alice', 'bad-password')
    with pytest.raises(ValueError):
        store.register('alice', 'another-password')
    chat = {'id':'private-chat', 'title':'Private', 'messages':[{'role':'user', 'content':'private content'}]}
    store.save(alice['id'], chat)
    assert store.chats(bob['id']) == {}
    with pytest.raises(PermissionError):
        store.save(bob['id'], chat)
    store.delete(bob['id'], chat['id'])
    assert len(store.chats(alice['id'])) == 1
    assert WorkspaceStore(tmp_path / 'app.db').chats(alice['id'])['private-chat']['messages'] == chat['messages']
    store.delete(alice['id'], chat['id'])
    assert store.chats(alice['id']) == {}
