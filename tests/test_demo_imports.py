"""Every name a demo module imports from a sibling module is exported there.

A missing export is a module-link error: the browser runs none of app.js and the
page sticks at its loading text, while every unit test of race.js stays green.
"""

import re
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent / "demo"
IMPORT = re.compile(r'import\s*\{([^}]*)\}\s*from\s*"\./([\w.-]+)"')
EXPORT = re.compile(r"export\s+(?:async\s+)?(?:function\*?|const|let|class)\s+(\w+)")


def test_demo_imports_resolve_to_real_exports():
    missing = []
    for module in sorted(DEMO.glob("*.js")):
        for names, target in IMPORT.findall(module.read_text()):
            exported = set(EXPORT.findall((DEMO / target).read_text()))
            for name in (n.split(" as ")[0].strip() for n in names.split(",")):
                if name and name not in exported:
                    missing.append(f"{module.name} imports {name} from {target}")
    assert not missing, missing
