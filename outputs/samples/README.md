# Mocked export fixtures

These are complete export-format examples, not travel recommendations. They use
existing synthetic venues and https://example.test/ receipts from
`tests/test_failures.py::FailureRun`. No live Tavily, Open-Meteo or Nebius calls
were made. Approval was simulated by the test harness, not given by a traveler.

The real graph produced Lisbon draft v1, then a separate trip was revised from
Lisbon to Paris and produced draft v2. Each exported bundle was validated with
`src.export.export_is_complete` and copied byte-for-byte, including its manifest.
The JSON and Markdown are intentionally unedited so manifest hashes remain valid.
The standard disclaimer in Markdown describes the export template; in these
fixtures source pages and evidence are mocked.

- mock_lisbon_v1: initial checked draft, simulated approval.
- mock_paris_revision_v2: new city applied before research, checked revised draft,
  simulated approval.

Replace these with actual traveler-approved live exports for a real travel demo.
A valid hash manifest proves file completeness/integrity, not factual accuracy.
