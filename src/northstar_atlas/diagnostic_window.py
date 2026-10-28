"""Private, digest-approved single-invocation exclusion contract.

This coordinates updated, cooperating writers. It is not an OS sandbox for
unreviewed clients. Operators must stop other writers and automatic restarts.
"""
from datetime import date
import hashlib
import json
import os
import re

from .live_budget import GRAPH, LINEAR, _fields, _require, _unique
from .llm import MODEL_ID
from .project import contained_path

WINDOW_NAME = 'diagnostic-window.json'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value))


def identity(path):
    info = path.lstat()
    return [info.st_dev, info.st_ino]


def validate_contract(value, *, profile=GRAPH):
    _require(profile in (GRAPH, LINEAR))
    _fields(value, 'invocation_id question_sha256 mode backend profile reference_date model corpus_sha256')
    _require(isinstance(value['invocation_id'], str)
             and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value['invocation_id'])
             and value['mode'] in ('approved_only', 'draft_review')
             and value['backend'] == ('chroma' if profile == GRAPH else 'pinecone')
             and value['profile'] == profile
             and value['model'] == MODEL_ID)
    digest(value['question_sha256']); digest(value['corpus_sha256'])
    _require(date.fromisoformat(value['reference_date']).isoformat() == value['reference_date'])


def validate_plan(budget, plan):
    linear = isinstance(plan, dict) and plan.get('kind') == 'v2-linear-demo-amendment'
    fields = 'version kind authority directory_identity lock_identity ledger_identity '
    fields += ('source_sha256 candidate_sha256 scope billing_scope model old_cap new_cap cost_cap '
               'pricing_sha256 request')
    if linear:
        fields += ' previous_window_sha256'
        digest(plan.get('previous_window_sha256'))
    _fields(plan, fields)
    _require(type(plan['version']) is int and plan['version'] == (2 if linear else 1)
             and plan['kind'] == ('v2-linear-demo-amendment' if linear else 'v2-diagnostic-amendment')
             and plan['authority'] == str(budget.authority_root)
             and plan['scope'] == budget.scope and plan['billing_scope'] == budget.billing_scope
             and plan['model'] == MODEL_ID and type(plan['old_cap']) is int
             and plan['old_cap'] == (15 if linear else 8)
             and type(plan['new_cap']) is int and plan['new_cap'] == (16 if linear else 15)
             and plan['cost_cap'] == '0.100000')
    for key in ('source_sha256', 'candidate_sha256', 'pricing_sha256'): digest(plan[key])
    for key in ('directory_identity', 'lock_identity', 'ledger_identity'):
        pair = plan[key]
        _require(isinstance(pair, list) and len(pair) == 2
                 and all(type(n) is int and n >= 0 for n in pair))
    _require(plan['directory_identity'] == identity(budget.directory)
             and plan['lock_identity'] == identity(budget.lock_path))
    validate_contract(plan['request'], profile=LINEAR if linear else GRAPH)


def read_private(budget, path, *, root, limit):
    """Bounded, private regular-file read without importing operator CLIs."""
    contained_path(root, str(path.relative_to(root)))
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        budget._opened_file(stream.fileno(), path)
        raw = stream.read(limit+1)
    _require(len(raw) <= limit)
    return raw


def read_window(budget, approval):
    """Called under the authoritative lock, including on every process restart."""
    digest(approval)
    raw = read_private(budget, budget.directory/WINDOW_NAME, root=budget.authority_root, limit=16384)
    _require(sha(raw) == approval)
    plan = json.loads(raw, object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    validate_plan(budget, plan)
    _require(budget.policy.max_attempts == plan['new_cap'])
    return plan


def check_request(budget, plan, request, corpus, ledger):
    expected = dict(invocation_id=request.invocation_id, question_sha256=sha(request.question.encode('utf-8')),
        mode=request.mode, backend=request.backend, profile=request.profile,
        reference_date=request.reference_date.isoformat(), model=MODEL_ID, corpus_sha256=corpus)
    contract_hash = sha(json.dumps(expected, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    _require(expected == plan['request'] and sha(budget._read_bytes()) == plan['candidate_sha256']
             and ledger.get('diagnostic_request_sha256') == contract_hash)
