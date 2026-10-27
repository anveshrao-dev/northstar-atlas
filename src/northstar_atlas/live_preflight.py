"""Fail-closed access checks and explicit read-only controlled-live preparation."""
import http.client
import ipaddress
import logging
import os
from pathlib import Path
import re
import socket
import ssl
from urllib.parse import urlsplit

import httpx


class LivePreflightError(ValueError):
    pass


def _public_ipv4(value):
    address = ipaddress.ip_address(value)
    if (address.version != 4 or not address.is_global or address.is_reserved
            or address.is_multicast or address.is_unspecified):
        raise ValueError("not a public gateway address")
    return str(address)


def _resolve_public_addresses(host):
    """Independent HTTPS DNS, never workspace DNS or a cached privacy assertion.

    Accept only bounded, direct A answers for this exact hostname. Aliases,
    truncation, private addresses, redirects and unavailable DNS fail closed.
    """
    try:
        with httpx.Client(timeout=10, follow_redirects=False, trust_env=False) as resolver:
            response = resolver.get("https://dns.google/resolve", params={"name": host, "type": "A"})
            if response.status_code != 200:
                raise ValueError
            data = response.json()
        question, answers = data["Question"], data["Answer"]
        if (type(data["Status"]) is not int or data["Status"] != 0
                or data.get("TC", False) is not False
                or not isinstance(question, list) or len(question) != 1
                or question[0]["name"].rstrip(".").lower() != host
                or type(question[0]["type"]) is not int or question[0]["type"] != 1
                or not isinstance(answers, list) or not 1 <= len(answers) <= 8):
            raise ValueError
        addresses = set()
        for answer in answers:
            if (type(answer["type"]) is not int or answer["type"] != 1
                    or answer["name"].rstrip(".").lower() != host):
                raise ValueError
            addresses.add(_public_ipv4(answer["data"]))
        return tuple(sorted(addresses))
    except (httpx.HTTPError, OSError, ValueError, TypeError, KeyError, AttributeError):
        raise LivePreflightError("independent public DNS verification unavailable; inference denied") from None


class _PublicGatewayTransport(httpx.BaseTransport):
    """Pin TCP to public DNS; retain the original HTTPS Host and TLS SNI.

    No proxy, redirects, credentials, cookie jar or workspace-hostname fallback.
    Only Location is exposed from responses; no cookie/challenge data is logged.
    """
    def __init__(self, host, address):
        self.host, self.address = host, _public_ipv4(address)

    def handle_request(self, request):
        connection = raw = response = None
        try:
            if (request.method != "GET" or request.url.scheme != "https"
                    or request.url.host != self.host or request.url.port not in (None, 443)
                    or request.url.query or request.url.path not in
                    {"/", "/_stcore/health", "/_stcore/host-config", "/_stcore/stream"}):
                raise ValueError
            context = ssl.create_default_context()
            connection = http.client.HTTPSConnection(self.host, timeout=10, context=context)
            connection.auto_open = 0  # A lost socket must never fall back to workspace DNS.
            raw = socket.create_connection((self.address, 443), timeout=10)
            if _public_ipv4(raw.getpeername()[0]) != self.address:
                raise ValueError
            connection.sock = context.wrap_socket(raw, server_hostname=self.host)
            connection.request("GET", request.url.path,
                               headers={"Host": self.host, "Accept": "*/*", "Connection": "close"})
            response = connection.getresponse()
            location = response.getheader("Location")
            return httpx.Response(response.status, content=b"",
                                  headers={"Location": location} if location is not None else {},
                                  extensions={"external_peer": self.address})
        except (OSError, ValueError, http.client.HTTPException):
            raise LivePreflightError("public gateway connection could not be verified; inference denied") from None
        finally:
            if response is not None:
                response.close()
            if connection is not None:
                connection.close()
            if raw is not None:
                raw.close()


def _denies_anonymous_access(response):
    if response.status_code in (401, 403):
        return True
    if response.status_code in (302, 303, 307, 308):
        try:
            location = urlsplit(response.headers.get("location", ""))
            return (location.scheme == "https" and location.hostname in {"replit.com", "www.replit.com"}
                    and location.port in (None, 443)
                    and location.path in ("/login", "/signin", "/__replshield")
                    and not location.username and not location.password)
        except ValueError:
            pass
    return False


def check_private_access(host: str):
    """Verify all public gateway addresses, not workspace-internal HTTP health."""
    if (not isinstance(host, str) or len(host) > 253 or not re.fullmatch(
            r"(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+replit\.dev", host)):
        raise LivePreflightError("expected this workspace's Replit development hostname")
    host = host.lower()
    try:
        addresses = _resolve_public_addresses(host)
        if not addresses or len(addresses) > 8:
            raise ValueError
        for address in addresses:
            address = _public_ipv4(address)
            with httpx.Client(transport=_PublicGatewayTransport(host, address), timeout=10,
                              follow_redirects=False, trust_env=False) as client:
                for path in ("/", "/_stcore/health", "/_stcore/host-config", "/_stcore/stream"):
                    client.cookies.clear()
                    response = client.get("https://" + host + path)
                    if response.extensions.get("external_peer") != address:
                        raise LivePreflightError("response route is not a verified public gateway; inference denied")
                    if not _denies_anonymous_access(response):
                        raise LivePreflightError("external anonymous access is not demonstrably blocked")
        return True
    except LivePreflightError:
        raise
    except (httpx.HTTPError, OSError, ValueError):
        raise LivePreflightError("independent external access verification failed; inference denied") from None


def require_private_preview():
    # The application cannot inspect the Replit UI toggle. Require an explicit
    # operator assertion AND independently denied anonymous HTTP access.
    if os.environ.get("NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED") != "1":
        raise LivePreflightError("operator confirmation of Private preview is required")
    check_private_access(os.environ.get("REPLIT_DEV_DOMAIN", ""))


def require_secret_environment(backend: str = 'chroma'):
    # The hosted launcher uses process Secrets; ordinary Mac loading is unchanged.
    if backend not in ('chroma', 'pinecone'):
        raise LivePreflightError('unsupported retrieval backend')
    required = ('NEBIUS_API_KEY',)
    if backend == 'pinecone':
        required += ('PINECONE_API_KEY', 'PINECONE_INDEX_NAME', 'PINECONE_CLOUD', 'PINECONE_REGION')
    if not all(os.environ.get(name, '').strip() for name in required):
        raise LivePreflightError('required Replit Secret environment is incomplete')


def main():
    """Explicit launcher preflight only. No inference, writes, or budget setup.

    Network access checks and selected-backend read checks happen only when this
    entry point is deliberately run. Imports/UI construction remain offline.
    """
    from .live_budget import configured_budget, GRAPH
    from .nebius import NebiusProvider, load_settings
    from .qa import open_retriever
    from .project import contained_path, validate_project_root
    from .llm import MODEL_ID
    stage = 'local budget'
    previous_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        root = validate_project_root(Path.cwd())
        for relative in ('.local/tmp', '.local/cache', '.venv/bin'):
            contained_path(root, relative)
        budget = configured_budget(root)
        if os.environ.get('NORTHSTAR_UI_ENABLE_GRAPH') == '1':
            # Graph startup is local only. External checks and provider creation
            # must wait for an explicit per-submission full-workflow reservation.
            from .live_graph import verify_graph_assets
            if (os.environ.get('NORTHSTAR_CONTROLLED_LIVE') != '1'
                    or os.environ.get('VECTOR_STORE', 'chroma') != 'chroma'
                    or os.environ.get('NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED') != '1'):
                raise LivePreflightError('graph activation configuration is incomplete')
            budget.check_ready(profile=GRAPH)
            stage = 'local graph assets'
            verify_graph_assets(root)
            stage = 'Replit Secret presence'
            require_secret_environment('chroma')
            print('Local graph preflight passed; external access checks await a reserved submission. No inference authorized yet.')
            return 0
        budget.check_ready()
        stage = 'private preview'
        require_private_preview()
        stage = 'Replit Secret presence'
        backend = os.environ.get('VECTOR_STORE', 'chroma')
        require_secret_environment(backend)
        stage = 'retrieval readiness'
        # Existing adapters verify source/configuration/publication identity.
        # Chroma remains the default; Pinecone is selected only by the operator.
        with open_retriever(root, backend) as (store, _embedder, fingerprint):
            status = store.status()
            if status['state'] != 'ready' or (backend == 'chroma' and
                    status.get('corpus_fingerprint') != fingerprint):
                raise LivePreflightError('retrieval index is not ready for the current corpus')
        stage = 'Nebius model availability'
        settings = load_settings(root)
        if settings.model != MODEL_ID:
            raise LivePreflightError('configured model differs from the approved model')
        with NebiusProvider(settings, allow_paid=False, max_attempts=1) as provider:
            if not provider.preflight()['model_listed']:
                raise LivePreflightError('configured model is unavailable')
        print('Controlled-live preflight passed; no inference requests were sent.')
        return 0
    except Exception:
        print(f'Controlled-live preflight stopped at {stage}; no inference authorized.')
        return 1
    finally:
        logging.disable(previous_logging)


if __name__ == '__main__':
    raise SystemExit(main())
