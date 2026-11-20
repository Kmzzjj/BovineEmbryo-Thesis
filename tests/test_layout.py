import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LayoutTest(unittest.TestCase):
    def test_sibling_script_references_exist(self):
        for path in (ROOT / "experiments").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr != "with_name" or not node.args:
                    continue
                name = node.args[0]
                if isinstance(name, ast.Constant) and isinstance(name.value, str) and name.value.endswith(".py"):
                    with self.subTest(script=path.relative_to(ROOT), reference=name.value):
                        self.assertTrue(path.with_name(name.value).is_file())

    def test_readme_links_exist(self):
        for path in (ROOT / "README.md", ROOT / "docs/workflow.md", ROOT / "experiments/README.md"):
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
                if target.startswith(("https://", "http://", "#")):
                    continue
                with self.subTest(document=path.relative_to(ROOT), target=target):
                    self.assertTrue((path.parent / target.split("#")[0]).exists())
