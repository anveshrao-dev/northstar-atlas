"""Independent gateway verification using mocked DNS, sockets, TLS and HTTP."""
from contextlib import contextmanager
import copy
import os
import ssl
import unittest
from unittest.mock import MagicMock, patch

import httpx

from northstar_atlas import live_preflight as access

HOST = "example.replit.dev"
PUBLIC_IP = "8.8.8.8"  # Synthetic gateway fixture, never contacted.
OTHER_PUBLIC_IP = "1.1.1.1"
INTERNAL_IP = "10.20.30.40"


class GatewayChecks(unittest.TestCase):
    def setUp(self):
        resolver = patch.object(access, "_resolve_public_addresses", return_value=(PUBLIC_IP,))
        self.resolve = resolver.start()
        self.addCleanup(resolver.stop)

    @contextmanager
    def gateway(self, send, *, peer=PUBLIC_IP):
        def respond(request):
            response = send(request)
            if peer is not None:
                response.extensions["external_peer"] = peer
            return response
        with patch.object(access, "_PublicGatewayTransport",
                          side_effect=lambda host, ip: httpx.MockTransport(respond)) as transport:
            yield transport

    def test_authentication_denials_and_all_recognized_redirects(self):
        responses = [httpx.Response(401), httpx.Response(403)]
        for status in (302, 303, 307, 308):
            for path in ("/login", "/signin", "/__replshield"):
                responses.append(httpx.Response(status, headers={"Location": "https://replit.com" + path}))
        for response in responses:
            with self.subTest(status=response.status_code, location=response.headers.get("location")):
                calls = []
                def send(request):
                    calls.append(request)
                    self.assertNotIn("Authorization", request.headers)
                    self.assertNotIn("Cookie", request.headers)
                    return response
                with self.gateway(send):
                    self.assertTrue(access.check_private_access(HOST))
                self.assertEqual([r.url.path for r in calls],
                                 ["/", "/_stcore/health", "/_stcore/host-config", "/_stcore/stream"])

    def test_internal_200_is_a_route_failure_not_public_access_evidence(self):
        with self.gateway(lambda r: httpx.Response(200), peer=INTERNAL_IP):
            with self.assertRaises(access.LivePreflightError) as caught:
                access.check_private_access(HOST)
        self.assertNotIn("public access", str(caught.exception))
        # The presence of a healthy internal app must not stop an independently
        # pinned external shield response from proving gateway authentication.
        with self.gateway(lambda r: httpx.Response(307, headers={"Location": "https://replit.com/__replshield"})):
            self.assertTrue(access.check_private_access(HOST))

    def test_missing_or_wrong_peer_evidence_cannot_authorize_live_access(self):
        for peer in (None, "127.0.0.1", OTHER_PUBLIC_IP):
            with self.subTest(peer=peer), self.gateway(lambda r: httpx.Response(403), peer=peer):
                with self.assertRaises(access.LivePreflightError):
                    access.check_private_access(HOST)

    def test_external_200_even_a_login_page_is_ambiguous_and_fails_closed(self):
        for body in ("Streamlit", "Log in to access a private app"):
            with self.gateway(lambda r: httpx.Response(200, text=body)):
                with self.assertRaises(access.LivePreflightError):
                    access.check_private_access(HOST)

    def test_unknown_redirects_and_error_responses_fail_closed(self):
        locations = ("https://evil.example/login", "http://replit.com/login",
                     "https://replit.com:444/login", "https://replit.com/unknown",
                     "https://user:password@replit.com/login", "/login",
                     "https://replit.com.evil.example/__replshield", "https://[invalid")
        responses = [httpx.Response(s) for s in (204, 404, 429, 500, 503)]
        responses += [httpx.Response(307, headers={"Location": target}) for target in locations]
        for response in responses:
            with self.gateway(lambda r: response), self.assertRaises(access.LivePreflightError):
                access.check_private_access(HOST)

    def test_one_public_route_is_enough_to_reject(self):
        with self.gateway(lambda r: httpx.Response(403 if r.url.path == "/" else 200)):
            with self.assertRaises(access.LivePreflightError):
                access.check_private_access(HOST)

    def test_every_public_dns_address_must_deny_access(self):
        self.resolve.return_value = (PUBLIC_IP, OTHER_PUBLIC_IP)
        calls = []
        def factory(host, address):
            def respond(request):
                calls.append(address)
                return httpx.Response(403 if address == PUBLIC_IP else 200,
                                      extensions={"external_peer": address})
            return httpx.MockTransport(respond)
        with patch.object(access, "_PublicGatewayTransport", side_effect=factory):
            with self.assertRaises(access.LivePreflightError):
                access.check_private_access(HOST)
        self.assertEqual(calls, [PUBLIC_IP] * 4 + [OTHER_PUBLIC_IP])

    def test_dns_failure_and_internal_answers_have_no_local_fallback(self):
        for addresses in ((), (INTERNAL_IP,), ("127.0.0.1",), ("100.64.0.1",)):
            self.resolve.return_value = addresses
            with patch.object(access, "_PublicGatewayTransport") as transport:
                with self.assertRaises(access.LivePreflightError):
                    access.check_private_access(HOST)
                transport.assert_not_called()
        self.resolve.side_effect = access.LivePreflightError("external DNS unavailable")
        with patch.object(access, "_PublicGatewayTransport") as transport:
            with self.assertRaises(access.LivePreflightError):
                access.check_private_access(HOST)
            transport.assert_not_called()

    def test_gateway_timeouts_and_tls_failures_fail_closed(self):
        for failure in (httpx.ReadTimeout("mock timeout"), ssl.SSLError("mock TLS failure")):
            def send(request):
                raise failure
            with self.gateway(send), self.assertRaises(access.LivePreflightError):
                access.check_private_access(HOST)

    def test_hostname_validation_precedes_any_resolution(self):
        for host in ("localhost", "https://example.replit.dev", "example.com", "../example.replit.dev",
                     "example..replit.dev", "example.replit.dev:443", "-bad.replit.dev", ""):
            with self.assertRaises(access.LivePreflightError):
                access.check_private_access(host)
        self.resolve.assert_not_called()

    def test_operator_confirmation_still_required_and_not_sufficient(self):
        with patch.dict(os.environ, NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED="0"):
            with self.assertRaises(access.LivePreflightError):
                access.require_private_preview()
        with patch.dict(os.environ, NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED="1", REPLIT_DEV_DOMAIN=HOST), \
             patch.object(access, "check_private_access", side_effect=access.LivePreflightError("unverified")) as check:
            with self.assertRaises(access.LivePreflightError):
                access.require_private_preview()
            check.assert_called_once_with(HOST)


class PublicDNSChecks(unittest.TestCase):
    def answer(self):
        return {"Status": 0, "TC": False, "Question": [{"name": HOST + ".", "type": 1}],
                "Answer": [{"name": HOST + ".", "type": 1, "data": PUBLIC_IP}]}

    @contextmanager
    def resolver(self, value, status=200):
        calls = []
        def send(request):
            calls.append(request)
            return httpx.Response(status, json=value)
        client = httpx.Client(transport=httpx.MockTransport(send))
        with patch.object(access.httpx, "Client", return_value=client) as factory:
            yield calls, factory

    def test_independent_https_dns_not_system_hostname_resolution(self):
        with self.resolver(self.answer()) as (calls, factory), \
             patch("socket.getaddrinfo", side_effect=AssertionError("workspace DNS forbidden")):
            self.assertEqual(access._resolve_public_addresses(HOST), (PUBLIC_IP,))
        factory.assert_called_once_with(timeout=10, follow_redirects=False, trust_env=False)
        self.assertEqual(str(calls[0].url), "https://dns.google/resolve?name=example.replit.dev&type=A")
        self.assertNotIn("Authorization", calls[0].headers)

    def test_malformed_truncated_private_alias_or_wrong_question_answers_rejected(self):
        variants = [{}, [], self.answer(), self.answer(), self.answer(), self.answer(),
                    self.answer(), self.answer(), self.answer()]
        variants[2]["Status"] = 3
        variants[3]["TC"] = True
        variants[4]["Question"][0]["name"] = "other.replit.dev."
        variants[5]["Answer"][0]["data"] = INTERNAL_IP
        variants[6]["Answer"][0]["name"] = "alias.example."
        variants[7]["Answer"][0]["type"] = 5
        variants[8]["Answer"] = []
        for value in variants:
            with self.resolver(value), self.assertRaises(access.LivePreflightError):
                access._resolve_public_addresses(HOST)

    def test_resolver_redirects_and_http_failure_do_not_fall_back(self):
        for status in (302, 307, 403, 500):
            with self.resolver(self.answer(), status), self.assertRaises(access.LivePreflightError):
                access._resolve_public_addresses(HOST)

    def test_dns_timeout_or_invalid_json_fails_closed(self):
        def timeout(request):
            raise httpx.ReadTimeout("synthetic timeout")
        for handler in (timeout, lambda r: httpx.Response(200, text="not JSON")):
            client = httpx.Client(transport=httpx.MockTransport(handler))
            with patch.object(access.httpx, "Client", return_value=client), \
                 self.assertRaises(access.LivePreflightError):
                access._resolve_public_addresses(HOST)

    def test_mixed_public_private_records_and_invalid_ip_are_rejected(self):
        for address in ("127.0.0.1", "169.254.169.254", "224.0.0.1", "not-an-ip"):
            value = copy.deepcopy(self.answer())
            value["Answer"].append({"name": HOST, "type": 1, "data": address})
            with self.resolver(value), self.assertRaises(access.LivePreflightError):
                access._resolve_public_addresses(HOST)


class PinnedTransportChecks(unittest.TestCase):
    def test_public_tcp_pin_original_host_and_verified_tls_sni_without_cookies(self):
        raw = MagicMock()
        raw.getpeername.return_value = (PUBLIC_IP, 443)
        context, secure, connection = MagicMock(), MagicMock(), MagicMock()
        context.wrap_socket.return_value = secure
        connection.getresponse.return_value.status = 307
        connection.getresponse.return_value.getheader.return_value = "https://replit.com/__replshield"
        with patch.object(access.socket, "create_connection", return_value=raw) as connect, \
             patch.object(access.ssl, "create_default_context", return_value=context), \
             patch.object(access.http.client, "HTTPSConnection", return_value=connection):
            result = access._PublicGatewayTransport(HOST, PUBLIC_IP).handle_request(
                httpx.Request("GET", "https://" + HOST + "/", headers={"Cookie": "synthetic", "Authorization": "synthetic"}))
        connect.assert_called_once_with((PUBLIC_IP, 443), timeout=10)
        context.wrap_socket.assert_called_once_with(raw, server_hostname=HOST)
        self.assertEqual(connection.auto_open, 0)
        connection.request.assert_called_once_with("GET", "/", headers={
            "Host": HOST, "Accept": "*/*", "Connection": "close"})
        self.assertEqual(result.extensions["external_peer"], PUBLIC_IP)
        self.assertNotIn("set-cookie", result.headers)
        connection.close.assert_called_once()
        raw.close.assert_called_once()
        # Standard-library default context verifies both trust and hostname.
        actual_context = ssl.create_default_context()
        self.assertTrue(actual_context.check_hostname)
        self.assertEqual(actual_context.verify_mode, ssl.CERT_REQUIRED)

    def test_private_or_wrong_socket_peer_never_sends_http(self):
        for peer in (INTERNAL_IP, OTHER_PUBLIC_IP):
            raw, connection = MagicMock(), MagicMock()
            raw.getpeername.return_value = (peer, 443)
            with patch.object(access.socket, "create_connection", return_value=raw), \
                 patch.object(access.http.client, "HTTPSConnection", return_value=connection), \
                 self.assertRaises(access.LivePreflightError):
                access._PublicGatewayTransport(HOST, PUBLIC_IP).handle_request(httpx.Request("GET", "https://" + HOST + "/"))
            connection.request.assert_not_called()
            raw.close.assert_called_once()

    def test_failed_tls_verification_never_falls_back_or_sends_http(self):
        raw, connection, context = MagicMock(), MagicMock(), MagicMock()
        raw.getpeername.return_value = (PUBLIC_IP, 443)
        context.wrap_socket.side_effect = ssl.SSLError("synthetic certificate failure")
        with patch.object(access.socket, "create_connection", return_value=raw) as connect, \
             patch.object(access.ssl, "create_default_context", return_value=context), \
             patch.object(access.http.client, "HTTPSConnection", return_value=connection), \
             self.assertRaises(access.LivePreflightError):
            access._PublicGatewayTransport(HOST, PUBLIC_IP).handle_request(httpx.Request("GET", "https://" + HOST + "/"))
        connect.assert_called_once_with((PUBLIC_IP, 443), timeout=10)
        connection.request.assert_not_called()
        connection.close.assert_called_once()
        raw.close.assert_called_once()
