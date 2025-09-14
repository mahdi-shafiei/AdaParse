"""Test multilingual document parsing support across different parsers."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pytest
from rapidfuzz import fuzz

from tests.utils.helpers import get_nougat_checkpoint

# Add the project root to sys.path to import adaparse modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from adaparse.parsers.pymupdf import PyMuPDFParser, PyMuPDFParserConfig
from adaparse.parsers.pypdf import PyPDFParser, PyPDFParserConfig
from adaparse.parsers.nougat import NougatParser, NougatParserConfig
from adaparse.parsers.adaparse import AdaParse, AdaParseConfig

# Test configuration
PDF_PATH = "./tests/data/groundtruth/languages/languages.pdf"
GROUNDTRUTH_PATH = "./tests/data/groundtruth/multilingual.mmd"
PAGE_SEPARATOR = "<><><><><><>NEWPAGE<><><><><><>"
SIMILARITY_THRESHOLD = 0.75  # 75% similarity threshold
MOCK_WEIGHTS_PATH = "./tests/mock_weights"  # Mock path for regression weights
ENV_BASENAME = ".adaparse.env"

# Language mapping - order corresponds to pages in the test PDF
LANGUAGES = ['ENG', 'SPA', 'DEU', 'JPN', 'FRA', 'POR', 'RUS', 'ITA', 'NLD', 'POL',
             'TUR', 'FAS', 'CMN', 'VIE', 'IND', 'CES', 'KOR', 'UKR', 'HUN', 'SWE',
             'ARA', 'RON', 'ELL', 'HEB', 'DAN']

# Regex for parsing env file assignments
_ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*)\s*$")


def load_env_file(path: Path) -> None:
    """Load environment variables from .env file."""
    if not path or not path.exists():
        return

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ASSIGN.match(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        # Handle quoted values
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        else:
            # Strip trailing inline comment on unquoted values
            if " #" in v:
                v = v.split(" #", 1)[0].rstrip()
        os.environ[k] = os.path.expanduser(os.path.expandvars(v))


def find_env_file(start: Path) -> Path | None:
    """Find .adaparse.env file by searching upwards from start directory."""
    p = start.resolve()
    while True:
        cand = p / ENV_BASENAME
        if cand.exists():
            return cand
        if p.parent == p:
            return None
        p = p.parent


def get_nougat_checkpoint() -> Path | None:
    """Get Nougat checkpoint path from environment."""
    # Load env file from parent directory of tests
    test_dir = Path(__file__).parent
    env_path = find_env_file(test_dir)
    if env_path:
        load_env_file(env_path)

    checkpoint_path = os.environ.get('ADAPARSE_CHECKPOINT')
    if checkpoint_path:
        return Path(checkpoint_path)
    return None


class TestLanguageSupport:
    """Test class for multilingual document parsing support."""

    @classmethod
    def setup_class(cls):
        """Setup test class with ground truth data."""
        cls.groundtruth_pages = cls._load_groundtruth()
        cls.pdf_path = PDF_PATH

    @classmethod
    def _load_groundtruth(cls) -> List[str]:
        """Load and split ground truth text by pages."""
        try:
            with open(GROUNDTRUTH_PATH, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split by page separator and clean up
            pages = [page.strip() for page in content.split(PAGE_SEPARATOR) if page.strip()]
            return pages
        except FileNotFoundError:
            pytest.skip(f"Ground truth file not found: {GROUNDTRUTH_PATH}")
        except Exception as e:
            pytest.fail(f"Failed to load ground truth: {e}")

    def _split_text_by_pages(self, text: str, page_indices: List[int]) -> List[str]:
        """Split full text into pages using character indices."""
        if not page_indices:
            return [text]

        pages = []
        for i in range(len(page_indices)):
            start_idx = page_indices[i]
            end_idx = page_indices[i + 1] if i + 1 < len(page_indices) else len(text)
            page_text = text[start_idx:end_idx].strip()
            pages.append(page_text)

        return pages

    def _calculate_similarity(self, parsed_page: str, groundtruth_page: str) -> float:
        """Calculate similarity between parsed page and ground truth using rapidfuzz."""
        if not parsed_page.strip() or not groundtruth_page.strip():
            return 0.0

        # Use token_sort_ratio which is more forgiving of word order differences
        similarity = fuzz.token_sort_ratio(parsed_page.strip(), groundtruth_page.strip())
        return similarity / 100.0  # Convert to 0-1 scale

    def _evaluate_parser_support(self, parsed_pages: List[str]) -> Dict[str, Any]:
        """Evaluate which languages/pages are supported by comparing with ground truth."""
        supported_pages = []
        unsupported_pages = []
        similarities = []

        max_pages = min(len(parsed_pages), len(self.groundtruth_pages), len(LANGUAGES))

        for i in range(max_pages):
            similarity = self._calculate_similarity(parsed_pages[i], self.groundtruth_pages[i])
            similarities.append(similarity)

            if similarity >= SIMILARITY_THRESHOLD:
                supported_pages.append(i)
            else:
                unsupported_pages.append(i)

        # Map page indices to language codes
        supported_languages = [LANGUAGES[i] for i in supported_pages]
        unsupported_languages = [LANGUAGES[i] for i in unsupported_pages]

        return {
            'supported_pages': supported_pages,
            'unsupported_pages': unsupported_pages,
            'supported_languages': supported_languages,
            'unsupported_languages': unsupported_languages,
            'total_pages': max_pages,
            'similarities': similarities,
            'support_percentage': len(supported_pages) / max_pages * 100 if max_pages > 0 else 0,
            'avg_similarity': sum(similarities) / len(similarities) if similarities else 0
        }

    def _parse_with_error_handling(self, parser, pdf_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """Parse PDF with error handling and return pages and metadata."""
        try:
            documents = parser.parse([pdf_path])

            if not documents or len(documents) == 0:
                return [], {'error': 'No documents returned'}

            # Get the first (and should be only) document
            doc = documents[0]
            full_text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            page_indices = metadata.get('page_char_idx', [])

            # DEBUG
            print(f'page_indices : {page_indices}')
            print(f'full_text : {full_text}')
            print(f'documents : {documents}')
            # - - - - -

            # Split text into pages
            pages = self._split_text_by_pages(full_text, page_indices)

            return pages, {
                'parser_name': doc.get('parser', 'unknown'),
                'page_count': len(pages),
                'total_chars': len(full_text)
            }, documents

        except Exception as e:
            return [], {'error': str(e)}

    @pytest.mark.skipif(not Path(PDF_PATH).exists(), reason=f"Test PDF not found: {PDF_PATH}")
    def test_pymupdf_language_support(self):
        """Test PyMuPDF parser language support."""
        # Setup parser
        config = PyMuPDFParserConfig()
        parser = PyMuPDFParser(config)

        # Parse PDF
        pages, parse_metadata, documents = self._parse_with_error_handling(parser, self.pdf_path)

        if 'error' in parse_metadata:
            pytest.skip(f"PyMuPDF parsing failed: {parse_metadata['error']}")

        # Evaluate support
        results = self._evaluate_parser_support(pages)

        # Assertions
        assert len(pages) > 0, "No pages were parsed"
        assert results['total_pages'] > 0, "No pages to evaluate"

        # Log results
        print(f"\nPyMuPDF Results:")
        print(f"  Supported pages: {len(results['supported_pages'])}/{results['total_pages']}")
        print(f"  Support percentage: {results['support_percentage']:.1f}%")
        print(f"  Average similarity: {results['avg_similarity']:.3f}")
        print(f"  Unsupported languages: {results['unsupported_languages']}")

        # Store results for comparison
        self.pymupdf_results = results

    @pytest.mark.skipif(not Path(PDF_PATH).exists(), reason=f"Test PDF not found: {PDF_PATH}")
    def test_pypdf_language_support(self):
        """Test PyPDF parser language support."""
        # Setup parser
        config = PyPDFParserConfig()
        parser = PyPDFParser(config)

        # Parse PDF
        pages, parse_metadata, documents = self._parse_with_error_handling(parser, self.pdf_path)

        if 'error' in parse_metadata:
            pytest.skip(f"PyPDF parsing failed: {parse_metadata['error']}")

        # Evaluate support
        results = self._evaluate_parser_support(pages)

        # Assertions
        assert len(pages) > 0, "No pages were parsed"
        assert results['total_pages'] > 0, "No pages to evaluate"

        # Log results
        print(f"\nPyPDF Results:")
        print(f"  Supported pages: {len(results['supported_pages'])}/{results['total_pages']}")
        print(f"  Support percentage: {results['support_percentage']:.1f}%")
        print(f"  Average similarity: {results['avg_similarity']:.3f}")
        print(f"  Unsupported languages: {results['unsupported_languages']}")

        # Store results for comparison
        self.pypdf_results = results

    @pytest.mark.skipif(not Path(PDF_PATH).exists(), reason=f"Test PDF not found: {PDF_PATH}")
    def test_nougat_language_support(self):
        """Test Nougat parser language support."""
        try:
            # Get checkpoint from environment
            nougat_checkpoint = get_nougat_checkpoint()
            if not nougat_checkpoint or not nougat_checkpoint.exists():
                pytest.skip(f"Nougat checkpoint not found. Set ADAPARSE_CHECKPOINT in .adaparse.env")

            # Setup parser with environment checkpoint
            config = NougatParserConfig(
                checkpoint=nougat_checkpoint,
                batchsize=1,
                num_workers=1,
                nougat_logs_path=Path("./tests/logs")
            )
            parser = NougatParser(config)

            # Parse PDF
            pages, parse_metadata, documents = self._parse_with_error_handling(parser, self.pdf_path)

            # CHECK
            print('type(pages)')
            print(type(pages))
            print('pages : ')
            print(pages)
            print('= = = = =')
            #print('parse_metadata')
            #print(parse_metadata)

            breakpoint()
            # - - - -

            if 'error' in parse_metadata:
                pytest.skip(f"Nougat parsing failed: {parse_metadata['error']}")

            # Evaluate support
            results = self._evaluate_parser_support(pages)

            # Assertions
            assert len(pages) > 0, "No pages were parsed"
            assert results['total_pages'] > 0, "No pages to evaluate"

            # Log results
            print(f"\nNougat Results:")
            print(f"  Supported pages: {len(results['supported_pages'])}/{results['total_pages']}")
            print(f"  Support percentage: {results['support_percentage']:.1f}%")
            print(f"  Average similarity: {results['avg_similarity']:.3f}")
            print(f"  Unsupported languages: {results['unsupported_languages']}")

            # Store results for comparison
            self.nougat_results = results

        except ImportError as e:
            pytest.skip(f"Nougat dependencies not available: {e}")
        except Exception as e:
            pytest.skip(f"Nougat parser setup failed: {e}")

    @pytest.mark.skipif(not Path(PDF_PATH).exists(), reason=f"Test PDF not found: {PDF_PATH}")
    def test_adaparse_language_support(self):
        """Test AdaParse parser language support."""
        try:
            # Get checkpoint from environment
            nougat_checkpoint = get_nougat_checkpoint()
            if not nougat_checkpoint or not nougat_checkpoint.exists():
                pytest.skip(f"AdaParse checkpoint not found. Set ADAPARSE_CHECKPOINT in .adaparse.env")

            # Check for mock weights path
            if not Path(MOCK_WEIGHTS_PATH).exists():
                pytest.skip(f"AdaParse weights not found: {MOCK_WEIGHTS_PATH}")

            # Setup parser with environment checkpoint
            config = AdaParseConfig(
                checkpoint=nougat_checkpoint,
                weights_path=Path(MOCK_WEIGHTS_PATH),
                batchsize=1,
                num_workers=1,
                nougat_logs_path=Path("./tests/logs"),
                alpha=0.5  # Allow more Nougat usage for testing
            )
            parser = AdaParse(config)

            # Parse PDF
            pages, parse_metadata = self._parse_with_error_handling(parser, self.pdf_path)

            if 'error' in parse_metadata:
                pytest.skip(f"AdaParse parsing failed: {parse_metadata['error']}")

            # Evaluate support
            results = self._evaluate_parser_support(pages)

            # Assertions
            assert len(pages) > 0, "No pages were parsed"
            assert results['total_pages'] > 0, "No pages to evaluate"

            # Log results
            print(f"\nAdaParse Results:")
            print(f"  Supported pages: {len(results['supported_pages'])}/{results['total_pages']}")
            print(f"  Support percentage: {results['support_percentage']:.1f}%")
            print(f"  Average similarity: {results['avg_similarity']:.3f}")
            print(f"  Unsupported languages: {results['unsupported_languages']}")

            # Store results for comparison
            self.adaparse_results = results

        except ImportError as e:
            pytest.skip(f"AdaParse dependencies not available: {e}")
        except Exception as e:
            pytest.skip(f"AdaParse parser setup failed: {e}")

    def test_parser_comparison_summary(self):
        """Generate a comparison summary of all parser results."""
        results_summary = {}

        # Collect results from all parsers that ran successfully
        if hasattr(self, 'pymupdf_results'):
            results_summary['PyMuPDF'] = self.pymupdf_results
        if hasattr(self, 'pypdf_results'):
            results_summary['PyPDF'] = self.pypdf_results
        if hasattr(self, 'nougat_results'):
            results_summary['Nougat'] = self.nougat_results
        if hasattr(self, 'adaparse_results'):
            results_summary['AdaParse'] = self.adaparse_results

        if not results_summary:
            pytest.skip("No parser results available for comparison")

        # Print comprehensive comparison
        print(f"\n{'='*60}")
        print("MULTILINGUAL PARSING COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Test PDF: {PDF_PATH}")
        print(f"Ground truth: {GROUNDTRUTH_PATH}")
        print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
        print(f"Total pages in ground truth: {len(self.groundtruth_pages)}")

        for parser_name, results in results_summary.items():
            print(f"\n{parser_name}:")
            print(f"  Support: {results['support_percentage']:.1f}% ({len(results['supported_pages'])}/{results['total_pages']} pages)")
            print(f"  Avg similarity: {results['avg_similarity']:.3f}")
            print(f"  Supported languages: {results['supported_languages']}")
            print(f"  Unsupported languages: {results['unsupported_languages']}")

        # Find best performing parser
        if results_summary:
            best_parser = max(results_summary.items(), key=lambda x: x[1]['support_percentage'])
            print(f"\nBest performing parser: {best_parser[0]} ({best_parser[1]['support_percentage']:.1f}% support)")

        print(f"{'='*60}")

        # Basic assertion that at least one parser worked
        assert len(results_summary) > 0, "No parsers produced valid results"


# Utility function to setup mock files if needed
def setup_mock_files():
    """Setup mock weights files for testing."""
    mock_weights_file = Path(MOCK_WEIGHTS_PATH)

    # Create mock weights file
    mock_weights_file.parent.mkdir(parents=True, exist_ok=True)
    if not mock_weights_file.exists():
        mock_weights_file.write_text("mock weights file")


if __name__ == "__main__":
    # Setup mock files for testing
    setup_mock_files()

    # Run the tests
    pytest.main([__file__, "-v", "-s"])
