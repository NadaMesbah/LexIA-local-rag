"""Unit tests for the persistence layer (pure SQLite — no Ollama/Chroma needed)."""

import json
import sqlite3

import pytest


def test_email_signup_and_login_lookup(fresh_db):
    db = fresh_db
    u = db.create_user("alice", "hash", role="user", first_name="A", last_name="L",
                       email="Alice@Example.com")
    assert u["email"] == "Alice@Example.com"
    # Email lookup is case-insensitive; empty never matches.
    assert db.get_user_by_email("alice@example.com")["username"] == "alice"
    assert db.get_user_by_email("ALICE@EXAMPLE.COM")["username"] == "alice"
    assert db.get_user_by_email("") is None
    assert db.get_user_by_email("nope@x.com") is None


def test_duplicate_email_rejected(fresh_db):
    db = fresh_db
    db.create_user("a", "h", email="dup@x.com")
    with pytest.raises(sqlite3.IntegrityError):
        db.create_user("b", "h", email="DUP@x.com")  # case-insensitive uniqueness


def test_find_document_id_disambiguates_org_vs_personal(fresh_db):
    db = fresh_db
    admin = db.create_user("admin", "h", role="admin", email="ad@x.com")
    alice = db.create_user("alice", "h", role="user", email="al@x.com")
    org = db.get_or_create_document("news.pdf", scope="org", owner_id=None)
    pa = db.get_or_create_document("news.pdf", scope="personal", owner_id=admin["id"])
    pb = db.get_or_create_document("news.pdf", scope="personal", owner_id=alice["id"])
    assert db.find_document_id("news.pdf", "org", None) == org["id"]
    assert db.find_document_id("news.pdf", "personal", admin["id"]) == pa["id"]
    assert db.find_document_id("news.pdf", "personal", alice["id"]) == pb["id"]
    assert db.find_document_id("missing.pdf", "org", None) is None


def test_reconcile_input_includes_personal_docs(fresh_db):
    """The reconcile bug: list_documents without role dropped personal docs."""
    db = fresh_db
    alice = db.create_user("alice", "h", role="user", email="al@x.com")
    org = db.get_or_create_document("o.pdf", scope="org", owner_id=None)
    perso = db.get_or_create_document("p.pdf", scope="personal", owner_id=alice["id"])
    db.add_version(org["id"], "o.pdf"); db.mark_version_latest(org["id"], 1)
    db.add_version(perso["id"], "p.pdf"); db.mark_version_latest(perso["id"], 1)

    buggy = {d["id"] for d in db.list_documents(include_deleted=True)}
    assert perso["id"] not in buggy           # reproduces the bug
    fixed = {d["id"] for d in db.list_documents(include_deleted=True, role="admin")}
    assert org["id"] in fixed and perso["id"] in fixed


def test_per_user_document_isolation(fresh_db):
    db = fresh_db
    alice = db.create_user("alice", "h", role="user", email="al@x.com")
    bob = db.create_user("bob", "h", role="user", email="bo@x.com")
    org = db.get_or_create_document("shared.pdf", scope="org", owner_id=None)
    pa = db.get_or_create_document("mine.pdf", scope="personal", owner_id=alice["id"])
    for d in (org, pa):
        db.add_version(d["id"], d["name"]); db.mark_version_latest(d["id"], 1)

    alice_ids = {d["id"] for d in db.list_documents(owner_id=alice["id"], role="user")}
    bob_ids = {d["id"] for d in db.list_documents(owner_id=bob["id"], role="user")}
    assert org["id"] in alice_ids and pa["id"] in alice_ids
    assert org["id"] in bob_ids and pa["id"] not in bob_ids   # bob can't see alice's


def test_message_attachments_round_trip(fresh_db):
    db = fresh_db
    u = db.create_user("admin", "h", role="admin", email="a@x.com")
    s = db.create_session(u["id"], "t")
    att = json.dumps([{"id": "d1", "filename": "notes.pdf"}])
    db.add_message(s["id"], "user", "hi", parent_id=None, attachments=att)
    path = db.get_active_path(s["id"])
    assert path[0]["attachments"] == att


def test_version_downloadable_flag_and_no_path_leak(fresh_db):
    db = fresh_db
    doc = db.get_or_create_document("d.pdf", scope="org", owner_id=None)
    ver = db.add_version(doc["id"], "d.pdf")
    docs = db.list_documents(role="admin")
    latest = docs[0]["latest"]
    assert latest["downloadable"] is False and "stored_path" not in latest
    db.set_version_file(ver["id"], "/tmp/whatever.pdf")
    db.mark_version_latest(doc["id"], 1)
    latest = db.list_documents(role="admin")[0]["latest"]
    assert latest["downloadable"] is True and "stored_path" not in latest


def test_email_migration_on_legacy_users_table(fresh_db, tmp_path, monkeypatch):
    """A users table created before the email column should migrate cleanly."""
    import importlib
    import db as dbmod
    legacy = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(legacy)
    conn.executescript(
        """
        CREATE TABLE users (
          id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
          first_name TEXT NOT NULL DEFAULT '', last_name TEXT NOT NULL DEFAULT '',
          role TEXT NOT NULL DEFAULT 'user', created_at REAL NOT NULL
        );
        INSERT INTO users (id,username,password_hash,role,created_at)
        VALUES ('x','legacy','h','admin',1700000000.0);
        """
    )
    conn.commit(); conn.close()
    monkeypatch.setenv("SQLITE_PATH", legacy)
    importlib.reload(dbmod)
    dbmod.init_db()  # must add the email column without error
    assert dbmod.get_user_by_username("legacy").get("email", "") == ""
    dbmod.set_user_email("x", "legacy@x.com")
    assert dbmod.get_user_by_email("legacy@x.com")["username"] == "legacy"
    # restore the module to the shared test DB
    monkeypatch.undo()
    importlib.reload(dbmod)
