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
RECOVERY_KIND = 'v2-pinecone-graph-recovery'
DEMO_KIND = 'v2-dedicated-demo-amendment'
DEMO_BILLING_SCOPE = 'northstar-atlas-week2-demo'


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


def snapshot_binding():
    from .live_graph import SNAPSHOT_ID, SNAPSHOT_SHA256, CORPUS_FINGERPRINT
    return dict(snapshot_id=SNAPSHOT_ID, snapshot_sha256=SNAPSHOT_SHA256,
                corpus_fingerprint=CORPUS_FINGERPRINT)


def verified_snapshot_binding(budget):
    value = snapshot_binding()
    path = contained_path(budget.root, f".local/pinecone/snapshots/{value['snapshot_id']}.json")
    _require(path.is_file() and sha(path.read_bytes()) == value['snapshot_sha256'])
    return value


def validate_recovery_contract(value):
    _fields(value, 'invocation_id question_sha256 mode backend profile reference_date model corpus_sha256 '
                   'snapshot_id snapshot_sha256 corpus_fingerprint')
    _require(isinstance(value['invocation_id'], str)
             and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value['invocation_id'])
             and value['mode'] == 'approved_only' and value['backend'] == 'pinecone'
             and value['profile'] == GRAPH and value['model'] == MODEL_ID)
    digest(value['question_sha256']); digest(value['corpus_sha256'])
    _require(date.fromisoformat(value['reference_date']).isoformat() == value['reference_date'])
    _require(all(value[k] == v for k, v in snapshot_binding().items()))


def _validate_demo_plan(budget, plan):
    _fields(plan, 'version kind authority authority_identity local_identity directory_identity lock_identity '
                  'ledger_identity source_sha256 candidate_sha256 scope billing_scope model old_cap new_cap '
                  'cost_cap max_call_reserved_usd pricing_sha256 request')
    _require(type(plan['version']) is int and plan['version'] == 4
             and plan['kind'] == DEMO_KIND and plan['authority'] == str(budget.authority_root)
             and plan['scope'] == budget.scope
             and plan['billing_scope'] == budget.billing_scope == DEMO_BILLING_SCOPE
             and plan['model'] == MODEL_ID and type(plan['old_cap']) is int and plan['old_cap'] == 1
             and type(plan['new_cap']) is int and plan['new_cap'] == 2
             and plan['cost_cap'] == '0.100000' and plan['max_call_reserved_usd'] == '0.010000')
    for key in ('source_sha256', 'candidate_sha256', 'pricing_sha256'): digest(plan[key])
    for key in ('authority_identity', 'local_identity', 'directory_identity', 'lock_identity', 'ledger_identity'):
        pair = plan[key]
        _require(isinstance(pair, list) and len(pair) == 2
                 and all(type(n) is int and n >= 0 for n in pair))
    validate_contract(plan['request'], profile=LINEAR)
    _require(plan['request']['mode'] == 'approved_only')


def _validate_plan_fields(budget, plan):
    """Historical structure only. NOT authority to use an old filesystem identity."""
    if isinstance(plan, dict) and plan.get('kind') == 'v2-unused-demo-batch-pricing-renewal':
        from .demo_batch_pricing import validate_renewal_plan
        validate_renewal_plan(budget, plan)
        return
    if isinstance(plan, dict) and plan.get('kind') == 'v2-five-question-demo-batch':
        from .demo_batch import validate_batch_plan
        validate_batch_plan(budget, plan)
        return
    if isinstance(plan, dict) and plan.get('kind') == DEMO_KIND:
        _validate_demo_plan(budget, plan)
        return
    linear = isinstance(plan, dict) and plan.get('kind') == 'v2-linear-demo-amendment'
    recovery = isinstance(plan, dict) and plan.get('kind') == RECOVERY_KIND
    fields = 'version kind authority directory_identity lock_identity ledger_identity '
    fields += ('source_sha256 candidate_sha256 scope billing_scope model old_cap new_cap cost_cap '
               'pricing_sha256 request')
    if linear or recovery:
        fields += ' previous_window_sha256'
        digest(plan.get('previous_window_sha256'))
    if recovery:
        fields += ' authority_identity local_identity continuity_sha256'
        digest(plan.get('continuity_sha256'))
    _fields(plan, fields)
    _require(type(plan['version']) is int and plan['version'] == (3 if recovery else 2 if linear else 1)
             and plan['kind'] == (RECOVERY_KIND if recovery else
                                 'v2-linear-demo-amendment' if linear else 'v2-diagnostic-amendment')
             and plan['authority'] == str(budget.authority_root)
             and plan['scope'] == budget.scope and plan['billing_scope'] == budget.billing_scope
             and plan['model'] == MODEL_ID and type(plan['old_cap']) is int
             and plan['old_cap'] == (15 if linear or recovery else 8)
             and type(plan['new_cap']) is int and plan['new_cap'] == (22 if recovery else 16 if linear else 15)
             and plan['cost_cap'] == '0.100000')
    for key in ('source_sha256', 'candidate_sha256', 'pricing_sha256'): digest(plan[key])
    keys = ('directory_identity', 'lock_identity', 'ledger_identity')
    for key in (*keys, *(('authority_identity', 'local_identity') if recovery else ())):
        pair = plan[key]
        _require(isinstance(pair, list) and len(pair) == 2
                 and all(type(n) is int and n >= 0 for n in pair))
    if recovery:
        validate_recovery_contract(plan['request'])
    else:
        validate_contract(plan['request'], profile=LINEAR if linear else GRAPH)


def validate_plan(budget, plan):
    _validate_plan_fields(budget, plan)
    _require(plan['directory_identity'] == identity(budget.directory)
             and plan['lock_identity'] == identity(budget.lock_path))
    if plan['kind'] in (RECOVERY_KIND, DEMO_KIND, 'v2-five-question-demo-batch',
                        'v2-unused-demo-batch-pricing-renewal'):
        _require(plan['authority_identity'] == identity(budget.authority_root)
                 and plan['local_identity'] == identity(budget.authority_root / '.local'))


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
    if plan['kind'] in ('v2-five-question-demo-batch', 'v2-unused-demo-batch-pricing-renewal'):
        from .demo_batch import check_batch_request
        check_batch_request(budget, plan, request, corpus, ledger)
        return
    expected = dict(invocation_id=request.invocation_id, question_sha256=sha(request.question.encode('utf-8')),
        mode=request.mode, backend=request.backend, profile=request.profile,
        reference_date=request.reference_date.isoformat(), model=MODEL_ID, corpus_sha256=corpus)
    if plan['kind'] == RECOVERY_KIND:
        expected.update(verified_snapshot_binding(budget))
    contract_hash = sha(json.dumps(expected, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    _require(expected == plan['request'] and sha(budget._read_bytes()) == plan['candidate_sha256']
             and ledger.get('diagnostic_request_sha256') == contract_hash)
