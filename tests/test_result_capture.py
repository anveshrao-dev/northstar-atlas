"""Private capture tests use synthetic public results, never provider calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from northstar_atlas import result_capture as capture
from northstar_atlas.evidence import CitationReference
from northstar_atlas.grounding import ConditionCoverage
from northstar_atlas.qa import PublicAnswer, PublicError, QADiagnostics, QAFailure, QASuccess


def success(kind='answer'):
    citation = CitationReference('ref-fixture', 'HR-001', '4.2', '1.0',
        'Approved (Synthetic)', 'a'*64, ('fixture-chunk',))
    return QASuccess(PublicAnswer(kind, 'Authored fixture only.', (citation,), ConditionCoverage()),
        QADiagnostics('complete', 'pinecone', 'approved_only', None, 'fixture', 'fixture', provider_calls=1))


class ResultCaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root/'.local/linear-demo-results/once.json'

    def run_once(self, execute):
        return capture.capture_result(execute, root=self.root, invocation_id='once')

    def saved(self):
        return json.loads(self.path.read_bytes())

    def test_public_content_identity_and_private_permissions(self):
        result = success()
        self.assertIs(self.run_once(lambda: result), result)
        saved = self.saved()
        self.assertEqual(saved['answer_text'], result.public.text)
        self.assertEqual(saved['citations'][0]['reference_id'], 'ref-fixture')
        self.assertEqual(saved['citations'][0]['content_hash'], 'a'*64)
        self.assertEqual(saved['citations'][0]['chunk_ids'], ['fixture-chunk'])
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.path.parent.stat().st_mode & 0o777, 0o700)

    def test_clarification_and_abstention_remain_typed_public_results(self):
        for kind in ('clarify', 'abstain'):
            result = success(kind)
            saved_result = capture.capture_result(lambda: result, root=self.root, invocation_id=kind)
            self.assertIs(saved_result, result)
            saved = json.loads((self.path.parent/f'{kind}.json').read_bytes())
            self.assertEqual(saved['answer_kind'], kind)

    def test_duplicate_or_interrupted_claim_never_executes_again(self):
        calls = []
        def execute():
            calls.append(1)
            self.assertEqual(self.saved()['status'], 'in_progress')
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt): self.run_once(execute)
        with self.assertRaises(FileExistsError): self.run_once(execute)
        self.assertEqual(calls, [1])

    def test_exception_propagates_after_sanitized_record(self):
        error = type('private_secret_name', (Exception,), {})('PRIVATE')
        with self.assertRaises(type(error)):
            self.run_once(Mock(side_effect=error))
        self.assertEqual(self.saved(), {'schema_version': 1, 'status': 'outcome_unknown'})

    def test_error_codes_are_allowlisted_and_messages_never_recorded(self):
        error = QAFailure(PublicError('secret_payload', 'PRIVATE'), success().diagnostics)
        self.assertIs(self.run_once(lambda: error), error)
        self.assertEqual(self.saved()['failure_code'], 'unknown')
        self.assertNotIn('PRIVATE', self.path.read_text())
        self.assertNotIn('secret_payload', self.path.read_text())

    def test_claim_write_failure_prevents_execution(self):
        calls = []
        with patch.object(capture.os, 'fsync', side_effect=OSError()), self.assertRaises(OSError):
            self.run_once(lambda: calls.append(1))
        with self.assertRaises(FileExistsError): self.run_once(lambda: calls.append(1))
        self.assertEqual(calls, [])

    def test_failed_final_write_withholds_result_and_prevents_reexecution(self):
        calls = []
        def execute():
            calls.append(1)
            return success()
        with patch.object(capture.os, 'replace', side_effect=OSError()), self.assertRaises(OSError):
            self.run_once(execute)
        self.assertEqual(self.saved()['status'], 'in_progress')
        with self.assertRaises(FileExistsError): self.run_once(execute)
        self.assertEqual(calls, [1])

    def test_symlink_and_unsafe_directory_rejected_without_execution(self):
        calls = []
        target = self.root/'target'; target.mkdir()
        (self.root/'.local').symlink_to(target, target_is_directory=True)
        with self.assertRaises(capture.ResultRecordingError): self.run_once(lambda: calls.append(1))
        (self.root/'.local').unlink()
        self.path.parent.mkdir(parents=True); self.path.parent.chmod(0o755)
        with self.assertRaises(capture.ResultRecordingError): self.run_once(lambda: calls.append(1))
        self.assertEqual(calls, [])

    def test_invalid_invocation_and_oversized_or_unknown_result_fail_closed(self):
        calls = []
        with self.assertRaises(ValueError):
            capture.capture_result(lambda: calls.append(1), root=self.root, invocation_id='../escape')
        self.assertEqual(calls, [])
        with self.assertRaises(capture.ResultRecordingError): self.run_once(lambda: object())
        self.assertEqual(self.saved()['status'], 'outcome_unknown')
        result = success()
        oversized = QASuccess(PublicAnswer('answer', 'x'*8001, (), ConditionCoverage()), result.diagnostics)
        with self.assertRaises(capture.ResultRecordingError):
            capture.capture_result(lambda: oversized, root=self.root, invocation_id='oversized')
        saved = json.loads((self.path.parent/'oversized.json').read_bytes())
        self.assertEqual(saved['status'], 'outcome_unknown')
        self.assertNotIn('answer_text', saved)
