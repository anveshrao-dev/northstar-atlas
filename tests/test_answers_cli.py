"""CLI boundaries use fake retrieval and never contact a service."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from test_context import hit, chunk
from checkout_fixture import checkout


class AnswerCLITests(unittest.TestCase):
    def setUp(self):
        self.root = checkout(self)
        self.environment = patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=str(self.root))
        self.cwd = patch.object(Path, 'cwd', return_value=self.root)
        self.environment.start(); self.cwd.start()
        self.addCleanup(self.environment.stop); self.addCleanup(self.cwd.stop)

    def test_prepare_uses_chroma_without_loading_llm_credentials(self):
        from northstar_atlas import answers
        selected=[]
        class Store:
            def search(self,*args,**kwargs): return (hit(chunk()),)
        @contextmanager
        def backend(root,name):
            selected.append(name); yield Store(),object(),"snapshot"
        output=io.StringIO()
        with patch.object(answers,"open_retriever",backend), patch.object(answers,"load_settings",side_effect=AssertionError("credentials")), \
             patch.dict("os.environ",{'NORTHSTAR_PROJECT_ROOT': str(self.root)},clear=True), redirect_stdout(output):
            result=answers.main(["prepare","Who?","--mode","draft_review"])
        self.assertEqual(result,0);self.assertEqual(selected,["chroma"])
        self.assertEqual(json.loads(output.getvalue())["model"],answers.MODEL_ID)

    def test_ask_without_approval_stops_before_any_io(self):
        from northstar_atlas import answers
        with patch.object(answers,"open_retriever",side_effect=AssertionError("retrieval")), redirect_stderr(io.StringIO()):
            self.assertEqual(answers.main(["ask","Who?"]),1)

    def test_wrong_directory_is_rejected_before_io(self):
        from northstar_atlas import answers
        with patch.object(Path,"cwd",return_value=self.root.parent), redirect_stderr(io.StringIO()):
            self.assertEqual(answers.main(["prepare","Who?"]),1)
