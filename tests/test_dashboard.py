"""Run the Streamlit dashboard headlessly: both tabs render and the controls work without errors."""
import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from ml.predict import read_table  # noqa: E402
from ml.split import HOLDOUT_DIR  # noqa: E402


@pytest.fixture(scope="module")
def app(bundle):
    # Force offline mode: an empty key also stops the app's load_dotenv() from reading a local .env file.
    previous = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = ""
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app" / "dashboard.py"), default_timeout=300)
    at.run()
    yield at
    if previous is None:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    else:
        os.environ["ANTHROPIC_API_KEY"] = previous


def test_dashboard_renders_without_errors(app):
    assert not app.exception
    headers = [h.value for h in app.header]
    assert "Hold-out test set (scored once) and time-based validation" in headers
    assert "Model comparison" not in headers   # needs the model ladder, which is not ported yet
    assert any("the time-based validation is the realistic forward-looking estimate" in m.value for m in app.markdown)


def test_sample_upload_and_explain(app):
    app.button(key="use_sample").click().run()
    assert not app.exception
    rows = len(read_table(HOLDOUT_DIR / "holdout_features.xlsx"))
    assert any(f"Scored {rows:,} row(s)" in i.value for i in app.info)
    app.button(key="explain_0").click().run()
    assert not app.exception
    assert any(c.value.startswith("Cat1 · ") and "Verdict:" in c.value for c in app.code)
    assert any(m.value.startswith("**Row 1**") for m in app.markdown)   # inside the pop-up dialog
    assert any("Offline template explanation" in i.value for i in app.info)
