# Test fixtures

`analyzer_selftest_v1.sqlite3` is a genuine Crimson Desert Analyzer database (schema 1), produced by the
Analyzer's own pipeline from its **synthetic** test installation (no game data):

```bash
git clone https://github.com/Sokhii/Crimson-Desert-Analyzer && cd Crimson-Desert-Analyzer
pip install -r requirements.txt
CSTUDIO_HOME=/tmp/analyzer_home python CrimsonSoundtrackStudio.py --selftest
cp /tmp/analyzer_home/data/database/studio.sqlite3 analyzer_selftest_v1.sqlite3
```

It keeps the Studio honest against the Analyzer's real output format (WAL-mode file, real field layouts in
`fields_json`, real reference kinds). Regenerate it when the Analyzer's schema changes, and add a new file
for the new schema version rather than replacing this one. Test audio is generated at test time.

`src/soundtrack_studio/testing/analyzer_fakeinstall.zip` + `analyzer_fakeinstall_v1.sqlite3`: the synthetic
installation written by the Analyzer's `testing/builders.make_fake_install` (real PAZ/PAMT archives with LZ4
entries and v150 banks, filler audio) and the database the Analyzer produced by scanning it
(`CSTUDIO_HOME=... python CrimsonSoundtrackStudio.py --scan <folder>`). The compiler tests and `--selftest`
build a real mod from it, so the archive reader and bank patcher are checked against an independent
implementation of the formats.
