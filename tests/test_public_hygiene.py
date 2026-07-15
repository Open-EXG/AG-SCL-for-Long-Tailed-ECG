from pathlib import Path
import unittest


class PublicHygieneTests(unittest.TestCase):
    def test_no_private_absolute_paths(self) -> None:
        root = Path(__file__).resolve().parents[1]
        extensions = {".py", ".md", ".yaml", ".yml", ".toml", ".cff"}
        forbidden = ("/" + "home/", "/" + "bigdata/", "ECG_" + "bed_single")
        violations = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in extensions and ".git" not in path.parts:
                text = path.read_text(encoding="utf-8")
                for marker in forbidden:
                    if marker in text:
                        violations.append(f"{path.relative_to(root)}: {marker}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
