"""HTTPX fake transports: no live requests, credentials or charges."""
import json
import logging
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import httpx


class NebiusTests(unittest.TestCase):
    def setUp(self):
        from northstar_atlas import nebius, llm
        self.n=nebius; self.l=llm
        self.key="synthetic-test-secret-not-a-real-key"
        self.settings=nebius.NebiusSettings(api_key=self.key)
        bound=llm.QwenTokenBound().request_bound("system","user",{})+512
        self.request=llm.GenerationRequest("system","user",{},llm.MODEL_ID,1024,32768,bound)

    def provider(self, handler, **kwargs):
        return self.n.NebiusProvider(self.settings, transport=httpx.MockTransport(handler), **kwargs)

    def response(self, request, **overrides):
        data=dict(model=self.l.MODEL_ID,choices=[{"message":{"role":"assistant","content":"{}"},"finish_reason":"stop"}],
                  usage={"prompt_tokens":40,"completion_tokens":10,"total_tokens":50})
        data.update(overrides)
        return httpx.Response(200,json=data,request=request)

    def test_paid_guard_prevents_any_http_request(self):
        with self.provider(lambda r: self.fail("network attempt")) as p:
            with self.assertRaises(self.n.ProviderError): p.generate(self.request)

    def test_request_format_and_usage(self):
        requests=[]
        def send(r): requests.append(r); return self.response(r)
        with self.provider(send,allow_paid=True) as p: result=p.generate(self.request)
        body=json.loads(requests[0].content)
        self.assertEqual(str(requests[0].url),"https://api.tokenfactory.nebius.com/v1/chat/completions")
        self.assertEqual(body["max_tokens"],1024)
        self.assertEqual(body["response_format"]["type"],"json_schema")
        self.assertFalse(body["stream"]); self.assertFalse(body["store"])
        self.assertEqual(result.usage.total_tokens,50)

    def test_model_list_preflight_never_posts(self):
        def send(r):
            self.assertEqual(r.method,"GET")
            return httpx.Response(200,json={"data":[{"id":self.l.MODEL_ID}]},request=r)
        with self.provider(send) as p: self.assertTrue(p.preflight()["model_listed"])

    def test_unavailable_model_is_not_silently_replaced(self):
        with self.provider(lambda r:httpx.Response(200,json={"data":[{"id":"alternative"}]},request=r)) as p:
            self.assertFalse(p.preflight()["model_listed"])

    def test_bounded_retry_honors_retry_after(self):
        calls=[]; waits=[]
        def send(r):
            calls.append(r)
            return httpx.Response(429,headers={"Retry-After":"1"},request=r) if len(calls)==1 else self.response(r)
        with self.provider(send,allow_paid=True,max_attempts=2,sleep=waits.append) as p: result=p.generate(self.request)
        self.assertEqual(result.attempts,2);self.assertEqual(waits,[1.0])

    def test_auth_and_timeout_errors_are_not_retried_or_leaked(self):
        for timeout in (True,False):
            calls=[]
            def send(r):
                calls.append(r)
                if timeout: raise httpx.ReadTimeout(self.key,request=r)
                return httpx.Response(401,json={"error":self.key},request=r)
            with self.provider(send,allow_paid=True,max_attempts=2) as p:
                with self.assertRaises(self.n.ProviderError) as error:p.generate(self.request)
            self.assertNotIn(self.key,str(error.exception)); self.assertEqual(len(calls),1)

    def test_invalid_usage_response_and_echoed_secret_rejected(self):
        values=[{"usage":{"prompt_tokens":True,"completion_tokens":0,"total_tokens":1}},
                {"usage":{"prompt_tokens":40,"completion_tokens":10,"total_tokens":41}},
                {"choices":[{"message":{"role":"assistant","content":self.key},"finish_reason":"stop"}]},
                {"choices":[]},{"model":"different"}]
        for value in values:
            with self.subTest(value=value), self.provider(lambda r:self.response(r,**value),allow_paid=True) as p:
                with self.assertRaises(self.n.ProviderError):p.generate(self.request)

    def test_unicode_escaped_credential_in_structured_content_is_rejected(self):
        escaped="".join("\\u%04x"%ord(c) for c in self.key)
        inner='{"kind":"abstain","claims":[],"missing_information":["'+escaped+'"],"clarification_question":null,"reason":"insufficient_evidence"}'
        def send(r):
            response=self.response(r,choices=[{"message":{"role":"assistant","content":inner},"finish_reason":"stop"}])
            self.assertNotIn(self.key,response.text)
            return response
        with self.provider(send,allow_paid=True) as p:
            with self.assertRaises(self.n.ProviderError) as error:p.generate(self.request)
        self.assertNotIn(self.key,str(error.exception))

    def test_endpoint_redirect_and_configuration_safety(self):
        for url in ("http://api.tokenfactory.nebius.com/v1", "https://evil.test/v1", "https://api.tokenfactory.nebius.com/v1?key=x"):
            with self.assertRaises(self.n.ProviderError):self.n.NebiusSettings(self.key,endpoint=url)
        with self.provider(lambda r:httpx.Response(307,headers={"Location":"https://evil.test"},request=r),allow_paid=True) as p:
            with self.assertRaises(self.n.ProviderError):p.generate(self.request)
        self.assertNotIn(self.key,repr(self.settings))

    def test_logs_do_not_contain_credentials_or_provider_body(self):
        records=[]
        class Capture(logging.Handler):
            def emit(self,record):records.append(record.getMessage())
        handler=Capture(); root=logging.getLogger(); root.addHandler(handler)
        previous=root.level; root.setLevel(logging.DEBUG)
        try:
            with self.provider(lambda r:httpx.Response(400,text=self.key,request=r),allow_paid=True) as p:
                with self.assertRaises(self.n.ProviderError):p.generate(self.request)
        finally:root.removeHandler(handler);root.setLevel(previous)
        self.assertNotIn(self.key,"\n".join(records))

    def test_retry_delay_above_budget_stops_without_replaying(self):
        calls=[]
        def send(r):
            calls.append(r)
            return httpx.Response(429,headers={"Retry-After":"60"},request=r)
        with self.provider(send,allow_paid=True,max_attempts=2,sleep=lambda _:self.fail("wait")) as p:
            with self.assertRaises(self.n.ProviderError):p.generate(self.request)
        self.assertEqual(len(calls),1)

    def test_missing_usage_stays_unknown_and_refusal_is_error(self):
        with self.provider(lambda r:self.response(r,usage=None),allow_paid=True) as p:
            self.assertIsNone(p.generate(self.request).usage)
        with self.provider(lambda r:self.response(r,choices=[{"message":{"role":"assistant","content":"{}","refusal":"refused"},"finish_reason":"stop"}]),allow_paid=True) as p:
            with self.assertRaises(self.n.ProviderError):p.generate(self.request)

    def test_private_configuration_rejects_conflicts_duplicates_and_tracked_env(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/".local/tmp") as temporary:
            directory=Path(temporary)
            clean=lambda *a,**k:subprocess.CompletedProcess(a,0,stdout=b"",stderr=b"")
            with patch.object(self.n.subprocess,"run",side_effect=clean):
                (directory/".env").write_text("NEBIUS_API_KEY="+self.key+"\nLLM_MODEL=\n")
                self.assertEqual(self.n.load_settings(directory,environ={}).model,self.l.MODEL_ID)
                with self.assertRaises(self.n.ProviderError):self.n.load_settings(directory,environ={"NEBIUS_API_KEY":"a-different-synthetic-key"})
                (directory/".env").write_text("NEBIUS_API_KEY="+self.key+"\nNEBIUS_API_KEY="+self.key)
                with self.assertRaises(self.n.ProviderError):self.n.load_settings(directory,environ={})
            with patch.object(self.n.subprocess,"run",return_value=subprocess.CompletedProcess([],0,stdout=b".env",stderr=b"")):
                with self.assertRaises(self.n.ProviderError):self.n.load_settings(directory,environ={})
