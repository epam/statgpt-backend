#!/usr/bin/env python3
"""
Check that modules follow architectural boundaries:
- common must not import from admin_portal, eval, or statgpt
- statgpt must only import from common (no imports from admin_portal or eval)
- admin_portal must only import from common (no imports from statgpt or eval)
"""
import ast
import sys
from pathlib import Path

MODULE = "statgpt"
ADMIN_SUBMODULE = "admin"
APP_SUBMODULE = "app"
CLI_SUBMODULE = "cli"
COMMON_SUBMODULE = "common"
ADMIN_MODULE = f"{MODULE}.{ADMIN_SUBMODULE}"
APP_MODULE = f"{MODULE}.{APP_SUBMODULE}"
COMMON_MODULE = f"{MODULE}.{COMMON_SUBMODULE}"
CLI_MODULE = f"{MODULE}.{CLI_SUBMODULE}"


class ImportViolation:
    def __init__(self, file_path: str, line: int, imported_module: str, violating_module: str):
        self._file_path = file_path
        self._line = line
        self._imported_module = imported_module
        self._violating_module = violating_module

    def __str__(self) -> str:
        return f"{self._file_path}:{self._line} - imports '{self._imported_module}' from forbidden module '{self._violating_module}'"


class ImportChecker:
    def __init__(self, src_dir: Path):
        self._src_dir = src_dir
        self._violations: list[ImportViolation] = []

    def check_all(self) -> list[ImportViolation]:
        """Check all modules for import violations."""
        modules = [COMMON_SUBMODULE, APP_SUBMODULE, ADMIN_SUBMODULE]

        for module in modules:
            module_path = self._src_dir / module
            if module_path.exists():
                self._check_module(module)

        return self._violations

    def _check_module(self, module_name: str) -> None:
        """Check a specific module for import violations."""
        module_path = self._src_dir / module_name

        # Define what each module is allowed to import
        forbidden_imports = self._get_forbidden_imports(module_name)

        # Walk through all Python files in the module
        for py_file in module_path.rglob("*.py"):
            self._check_file(py_file, forbidden_imports)

    def _get_forbidden_imports(self, module_name: str) -> set[str]:
        """Get the set of forbidden import prefixes for a module."""
        if module_name == COMMON_SUBMODULE:
            # common cannot import from any other application module
            return {ADMIN_MODULE, APP_MODULE, CLI_MODULE}
        elif module_name == APP_SUBMODULE:
            # statgpt can only import from common
            return {ADMIN_MODULE, CLI_MODULE}
        elif module_name == ADMIN_SUBMODULE:
            # admin_portal can only import from common
            return {APP_MODULE, CLI_MODULE}
        elif module_name == CLI_SUBMODULE:
            # cli can import from any module
            return set()
        return set()

    def _check_file(self, file_path: Path, forbidden_imports: set[str]) -> None:
        """Check a single Python file for forbidden imports."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except SyntaxError:
            # Skip files with syntax errors (they'll be caught by other linters)
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Check: import module_name
                for alias in node.names:
                    self._check_import_name(file_path, node.lineno, alias.name, forbidden_imports)
            elif isinstance(node, ast.ImportFrom):
                # Check: from module_name import ...
                if node.module:
                    self._check_import_name(file_path, node.lineno, node.module, forbidden_imports)

    def _check_import_name(
        self, file_path: Path, line: int, import_name: str, forbidden_imports: set[str]
    ) -> None:
        """Check if an import name violates the rules."""
        # Get the top-level module (first part before the first dot)
        module_names = import_name.split(".")
        if len(module_names) < 2:
            return
        l1 = module_names[0]
        l2 = module_names[1]
        top_level = f"{l1}.{l2}"

        if top_level in forbidden_imports:
            violation = ImportViolation(
                str(file_path.relative_to(self._src_dir.parent)), line, import_name, top_level
            )
            self._violations.append(violation)


def main() -> int:
    """Main entry point."""
    # Assume script is in scripts/ and src/ is a sibling directory
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent / MODULE

    if not src_dir.exists():
        print(f"Error: Source directory not found: {src_dir}", file=sys.stderr)
        return 1

    checker = ImportChecker(src_dir)
    violations = checker.check_all()

    if violations:
        print("Found import violations:", file=sys.stderr)
        print(file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        print(file=sys.stderr)
        print(f"Total: {len(violations)} violation(s)", file=sys.stderr)
        return 1

    print("No import violations found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
