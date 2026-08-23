from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noticelens.phase1 import _notice_aliases, clean_html  # noqa: E402


class CleanHtmlTests(unittest.TestCase):
    def test_notice_aliases_handle_series_and_composite_values(self) -> None:
        self.assertEqual(_notice_aliases("CP2000 series"), {"CP2000"})
        self.assertEqual(_notice_aliases("CP06/CP06A"), {"CP06", "CP06A"})
        self.assertEqual(_notice_aliases("LT11 / Letter 1058"), {"LT11"})

    def test_preserves_article_structure_and_removes_page_chrome(self) -> None:
        html = b"""
        <!doctype html><html><body>
          <nav><p>Search IRS.gov</p><h2>Main navigation</h2></nav>
          <h1>Understanding your CP999 notice</h1>
          <aside><p>Unrelated side navigation</p></aside>
          <main><article>
            <h2>What this notice is about</h2>
            <p>This is a meaningful paragraph about the notice with enough explanatory detail
            to represent the article body rather than global navigation or footer text.</p>
            <h2>Frequently asked questions</h2>
            <h3 class="accordion-heading"><button>What should I do?</button></h3>
            <p>Read the notice and use the official <a href="/payments">payment page</a>.</p>
            <h2>You may want to</h2>
            <ul><li>Keep a copy.</li><li>Respond by the stated date.</li></ul>
            <div>This useful text is directly inside a container rather than a paragraph.</div>
            <table><thead><tr><th>Office</th><th>Address</th></tr></thead>
            <tbody><tr><td>Austin</td><td>100 Main St.<br>Texas</td></tr></tbody></table>
            <p>This additional explanatory paragraph deliberately makes the synthetic article
            long enough to pass the meaningful-content quality threshold used in production.
            It also verifies that useful prose after lists is retained in document order.</p>
          </article></main>
          <footer><p>Footer navigation</p></footer>
        </body></html>
        """
        result = clean_html(
            html,
            doc_id="irs_cp999",
            source_url="https://www.irs.gov/individuals/understanding-your-cp999-notice",
            fallback_title="Fallback title",
        )
        self.assertIn("# Understanding your CP999 notice", result.markdown)
        self.assertIn("## What this notice is about", result.markdown)
        self.assertIn("### What should I do?", result.markdown)
        self.assertIn("- Keep a copy.", result.markdown)
        self.assertIn("This useful text is directly inside a container", result.markdown)
        self.assertIn("| Office | Address |", result.markdown)
        self.assertIn("| Austin | 100 Main St.<br>Texas |", result.markdown)
        self.assertIn("[payment page](https://www.irs.gov/payments)", result.markdown)
        self.assertNotIn("Main navigation", result.markdown)
        self.assertNotIn("Footer navigation", result.markdown)
        self.assertEqual(result.heading_counts["h2"], 3)
        self.assertEqual(result.heading_counts["h3"], 1)
        self.assertEqual(result.faq_questions, 1)
        self.assertGreaterEqual(result.source_text_token_coverage, 0.99)

    def test_preserves_nested_list_items_and_paragraph_breaks(self) -> None:
        html = b"""
        <html><head><link rel="canonical" href="https://www.irs.gov/test"></head><body>
        <h1>CP999 test notice</h1><article>
        <h2>Steps</h2>
        <ul><li>Outer instruction<ul>
          <li><p>Inner address</p><p>400 Second Street</p></li>
          <li>Inner follow-up</li>
        </ul></li></ul>
        <p>This paragraph provides enough meaningful explanatory content for a realistic
        structure test. It is deliberately repeated in concept, but not verbatim, so that
        the cleaner has ordinary prose alongside the nested procedural list and headings.</p>
        <p>Additional official guidance would appear here in a real IRS article and remain
        associated with the Steps heading for later heading-aware chunking experiments.</p>
        </article></body></html>
        """
        result = clean_html(
            html,
            doc_id="irs_cp999",
            source_url="https://www.irs.gov/test",
            fallback_title="CP999 fallback",
            expected_notice_code="CP999",
        )
        self.assertIn("- Outer instruction", result.markdown)
        self.assertIn("  - Inner address<br>400 Second Street", result.markdown)
        self.assertIn("  - Inner follow-up", result.markdown)
        self.assertEqual(result.source_list_item_count, 3)
        self.assertEqual(result.list_item_count, 3)
        self.assertNotIn("preserved 2 of 3 source list items", result.problems)


if __name__ == "__main__":
    unittest.main()
