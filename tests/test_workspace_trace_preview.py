from pathlib import Path


WORKSPACE_TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "workspace.html"


def test_ml_curve_prediction_replaces_previous_preview():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")

    assert "window.mlPredictedCurves.push({" not in source
    assert "window.mlPredictedCurves = [{" in source


def test_successful_digitization_clears_transient_ml_preview():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    success_block = source.index("if (data.success) {")
    clear_preview = source.index("window.mlPredictedCurves = [];", success_block)
    accept_traces = source.index("lastCurveTraces = traces;", success_block)

    assert success_block < clear_preview < accept_traces
