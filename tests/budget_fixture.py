"""Synthetic budget documents only; never reads or initializes a real ledger."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from northstar_atlas.llm import MODEL_ID, GenerationRequest, GenerationResponse, QwenTokenBound, TokenUsage

SCOPE = '1' * 32
BILLING = 'synthetic-billing-scope'


def document(*, calls=3, legacy=None):
    now = datetime.now(timezone.utc)
    return dict(version=2, scope=SCOPE, billing_scope=BILLING, provider='nebius', model=MODEL_ID,
                max_attempts=calls, max_reserved_usd='0.100000', legacy=legacy,
                pricing_revisions=[dict(id='fixture-price', input_per_million_usd='0.1',
                    output_per_million_usd='0.3', source='https://nebius.com/prices',
                    confirmed_at=(now-timedelta(minutes=5)).isoformat(),
                    valid_until=(now+timedelta(hours=1)).isoformat())],
                active_pricing='fixture-price', workflows=[])


def install(root, value):
    directory = root / '.local/controlled-live'
    directory.mkdir(parents=True, mode=0o700)
    directory.chmod(0o700)
    for name, text in (('budget.lock', ''), ('budget.json', json.dumps(value))):
        path = directory/name
        path.write_text(text)
        path.chmod(0o600)
    return directory/'budget.json'


def request(output=2048, context=32768, bound=None):
    minimum = QwenTokenBound().request_bound('system', 'user', {})+512
    return GenerationRequest('system', 'user', {}, MODEL_ID, output, context, bound or minimum)


class Provider:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def generate(self, req):
        self.calls.append(req)
        if self.failure:
            raise self.failure
        return GenerationResponse('{}', MODEL_ID, 'stop', TokenUsage(10, 2, 12), 0.01, 1)
