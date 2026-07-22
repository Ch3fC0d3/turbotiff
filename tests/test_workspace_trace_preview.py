import re
import shutil
import subprocess
from pathlib import Path

import pytest


WORKSPACE_TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "workspace.html"
WEB_APP = Path(__file__).resolve().parents[1] / "web_app.py"


def _drag_source(source):
    drag_start = source.index("function _doDragUpdate()")
    drag_end = source.index("function getBezierSegments", drag_start)
    return source[drag_start:drag_end]


def _function_source(source, function_name, next_function_name):
    function_start = source.index(f"function {function_name}(")
    function_end = source.index(f"function {next_function_name}(", function_start)
    return source[function_start:function_end]


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


def test_curve_drag_uses_stable_interaction_snapshot_for_proposals():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    drag_source = _drag_source(source)

    assert "originalDepthIndex: neighborDepthIdx" in source
    assert "originalValues: activeValues.slice()" in source
    assert "const sourceDepthIndex = n.originalDepthIndex;" in drag_source
    assert "const originalValues = state.originalValues;" in drag_source
    assert "const proposals = neighbors.map(n =>" in drag_source


def test_curve_drag_resolves_duplicate_destinations_with_stable_priority():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    drag_source = _drag_source(source)
    resolver_source = _function_source(source, "resolveDepthMoveProposals", "applyCurveXShiftForCurve")

    center_priority = drag_source.index("if (a.isCenter !== b.isCenter)")
    weight_priority = drag_source.index("if (a.weight !== b.weight)", center_priority)
    movement_priority = drag_source.index("if (aMovement !== bMovement)", weight_priority)
    point_priority = drag_source.index("return Number(a.pointIndex) - Number(b.pointIndex);", movement_priority)

    assert center_priority < weight_priority < movement_priority < point_priority
    assert "resolveDepthMoveProposals(proposals, originalValues, compareProposalPriority);" in drag_source
    assert "destinationProposals.sort(compareProposalPriority);" in resolver_source
    assert "loser.rejectionReason = 'duplicate-destination';" in resolver_source


def test_curve_drag_propagates_rejection_from_stationary_destinations():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    drag_source = _drag_source(source)
    resolver_source = _function_source(source, "resolveDepthMoveProposals", "applyCurveXShiftForCurve")

    assert "resolveDepthMoveProposals(proposals, originalValues, compareProposalPriority);" in drag_source
    assert "while (rejectedAnotherProposal)" in resolver_source
    assert "stationarySourceDepths.has(proposal.destinationDepthIndex)" in resolver_source
    assert "proposal.rejectionReason = 'destination-not-vacated';" in resolver_source
    assert "destinationHasExternalOccupant" in resolver_source


def test_curve_drag_commits_resolved_moves_atomically_and_preserves_swaps():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    drag_source = _drag_source(source)
    commit_start = source.index("function commitCurveDrag")
    commit_end = source.index("async function finishCurveEditDrag", commit_start)
    commit_source = source[commit_start:commit_end]

    calculate = drag_source.index("const proposals = neighbors.map(n =>")
    resolve = drag_source.index("resolveDepthMoveProposals(proposals, originalValues, compareProposalPriority);", calculate)
    snapshot = drag_source.index("const nextValues = originalValues.slice();", resolve)
    clear_source = drag_source.index("nextValues[proposal.sourceDepthIndex] = null;", snapshot)
    write_destination = drag_source.index("nextValues[proposal.destinationDepthIndex] = proposal.newValue;", clear_source)
    store_preview = drag_source.index("state.previewValues = nextValues;", write_destination)

    assert calculate < resolve < snapshot < clear_source < write_destination < store_preview
    assert "commitCurveLayerValues(" in commit_source
    assert "activeValues[idx] = state.previewValues[idx];" not in commit_source
    assert "activeValues[" not in drag_source
    assert "proposal.neighbor.depthIndex" not in drag_source
    assert "activeValues[previousDepthIndex] = n.originalValue" not in drag_source


def test_curve_drag_keeps_pinned_destination_and_active_layer_protection():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    drag_source = _drag_source(source)

    assert "const activeValues = getActiveLayerValues(entry, state.dragLayer);" in drag_source
    assert "const originalValues = state.originalValues;" in drag_source
    assert "isDepthIndexPinned(state.curveId, destinationDepthIndex)" in drag_source
    assert "proposal.rejectionReason = 'destination-protected';" in drag_source
    assert "const nextValues = originalValues.slice();" in drag_source


def test_add_point_binds_one_layer_values_track_and_trace_context():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    add_source = _function_source(source, "handleCurveAddPoint", "handleCurveEditDelete")

    resolve_layer = add_source.index("const activeLayer = explicitLayer === 'main' || explicitLayer === 'wrap'")
    resolve_values = add_source.index("const activeValues = activeLayer === 'wrap' ? entry.wrapLayer.values : entry.values;")
    resolve_track = add_source.index("const activeTrack = getActiveLayerTrackConfig(entry, targetTrack, activeLayer);", resolve_values)
    resolve_trace = add_source.index("const activeTrace = getActiveCurveTracePoints(editCurveId, activeLayer);", resolve_track)

    assert resolve_layer < resolve_values < resolve_track < resolve_trace
    assert "getActiveLayerValues(entry)" not in add_source
    assert "entry.values[" not in add_source
    assert "activeValues[dIdx] = pointValue;" in add_source


def test_add_point_rejects_locked_or_missing_wrap_layer_without_main_fallback():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    add_source = _function_source(source, "handleCurveAddPoint", "handleCurveEditDelete")

    lock_check = add_source.index("if (isLayerLocked(activeLayer, entry))")
    missing_wrap_check = add_source.index("if (activeLayer === 'wrap' && !hasWrapValues)", lock_check)
    values_selection = add_source.index("const activeValues = activeLayer === 'wrap' ? entry.wrapLayer.values : entry.values;", missing_wrap_check)
    first_value_write = add_source.index("activeValues[dIdx] = pointValue;", values_selection)

    assert lock_check < missing_wrap_check < values_selection < first_value_write
    assert "Cannot add point: wrapped curve values are unavailable." in add_source


def test_add_point_snapshots_selected_layer_and_exact_trace_for_undo():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    add_source = _function_source(source, "handleCurveAddPoint", "handleCurveEditDelete")

    assert "const undoValues = activeValues.map(" in add_source
    assert "dragLayer: activeLayer" in add_source
    assert "traceKey: targetKey" in add_source
    assert "lastCurveTraces[targetKey] = pts;" in add_source
    assert "const originalValue = activeValues[dIdx];" in add_source
    assert "!isMissingDigitizedValue(activeValues[idx])" in add_source
    assert "isDepthIndexPinned(editCurveId, dIdx)" in add_source


def test_add_point_undo_restores_saved_layer_and_trace_only():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    undo_source = _function_source(source, "undoLastCurveEdit", "beginCurveEditInteraction")
    add_undo_start = undo_source.index("if (last.type === 'add_point_crest')")
    add_undo_end = undo_source.index("if (last.type === 'bulk_transform')", add_undo_start)
    add_undo_source = undo_source[add_undo_start:add_undo_end]

    assert "const { curveKey, curveId, traceKey, dragLayer, undoTracePoints, undoValues } = last;" in add_undo_source
    assert "const vals = dragLayer === 'wrap'" in add_undo_source
    assert "entry.wrapLayer.values" in add_undo_source
    assert ": entry.values;" in add_undo_source
    assert "lastCurveTraces[traceKey]" in add_undo_source
    assert "rebuildCurveTraceFromDigitized" not in add_undo_source


def test_add_point_and_undo_are_functionally_layer_isolated():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")

    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    add_source = _function_source(source, "handleCurveAddPoint", "handleCurveEditDelete")
    undo_source = _function_source(source, "undoLastCurveEdit", "beginCurveEditInteraction")
    harness = r"""
const assert = require('assert');
let editMode = true;
let editCurveId = 'GR';
let editCurveDataKey = 'GR';
let lastDigitizedDepth = Array.from({ length: 40 }, (_, i) => i);
let lastDigitizedCurves;
let lastCurveTraces;
let lastDepthConfig = {};
let editUndoStack;
let desiredLayer = 'main';
let lockedLayer = null;
let pinnedDepths = new Set();
const window = { currentDragLayer: 'main', lastActiveLayer: 'main', lastNullValue: -999.25 };
const targetTrack = { leftX: 0, rightX: 200, scaleMin: 0, scaleMax: 200 };

function showStatus() {}
function findTrackByCurveId() { return targetTrack; }
function getTrackCalibrationsSnapshot() { return [targetTrack]; }
function resolveActiveLayer() { return desiredLayer; }
function isLayerLocked(layer) { return layer === lockedLayer; }
function getActiveLayerTrackConfig(entry, track, explicitLayer = null) {
    const layer = explicitLayer || desiredLayer;
    return layer === 'wrap'
        ? { ...track, scaleMin: entry.wrapLayer.left_value, scaleMax: entry.wrapLayer.right_value }
        : track;
}
function getDepthConfigFromInputs() { return {}; }
function pixelToDepthFromConfig(y) { return y; }
function findNearestDepthIndex(depth) { return Math.round(depth); }
function getPixelYForDepthIndex(idx) { return idx; }
function isDepthIndexPinned(curveId, idx) { return pinnedDepths.has(idx); }
function isTrackWrapActive() { return true; }
function isMissingDigitizedValue(value) { return value == null || !Number.isFinite(Number(value)); }
function trackValueToPixelX(value) { return Number(value); }
function unwrapDisplayXNearReference(x) { return x; }
function pixelToTrackValue(x) { return Number(x); }
function getActiveCurveTraceKey() { return 'GR'; }
function getActiveCurveTracePoints(curveId, explicitLayer = null) {
    const traceKey = (explicitLayer || desiredLayer) === 'wrap' ? 'GR_wrap' : 'GR';
    return { traceKey, points: lastCurveTraces[traceKey] };
}
function ensureTracePointsHaveDepthIndex() {}
function isPointLocked() { return false; }
function wrapRawXToTrack(x) { return x; }
function findBestPointIndexForDepth(pts, depthIdx) { return pts.findIndex(pt => pt[2] === depthIdx); }
function findEditablePointIndex() { return { pointIndex: -1 }; }
function renderCurveTraceOverlays() {}
function markCurveLayerTraceRevision() {}
function commitInPlaceCurveLayerMutation(curveId, layer, changed, options = {}) {
    if (changed && options.rebuildTrace !== false && typeof rebuildCurveTraceFromDigitized === 'function') {
        rebuildCurveTraceFromDigitized(curveId, layer);
    }
    return { changed };
}

function reset(layer, options = {}) {
    desiredLayer = layer;
    lockedLayer = options.lockedLayer || null;
    pinnedDepths = new Set(options.pinnedDepths || []);
    window.currentDragLayer = layer;
    const mainValues = Array(40).fill(10);
    const wrapValues = Array(40).fill(110);
    if (options.missingWrapDepth != null) wrapValues[options.missingWrapDepth] = null;
    lastDigitizedCurves = {
        GR: {
            values: mainValues,
            wrapLayer: { values: wrapValues, left_value: 100, right_value: 200 },
        },
    };
    lastCurveTraces = {
        GR: mainValues.map((value, idx) => [value, idx, idx]),
        GR_wrap: wrapValues.map((value, idx) => [value, idx, idx]),
    };
    editUndoStack = [];
}

function snapshot() {
    return JSON.parse(JSON.stringify({
        mainValues: lastDigitizedCurves.GR.values,
        wrapValues: lastDigitizedCurves.GR.wrapLayer.values,
        mainTrace: lastCurveTraces.GR,
        wrapTrace: lastCurveTraces.GR_wrap,
    }));
}

for (const layer of ['main', 'wrap']) {
    reset(layer);
    const before = snapshot();
    handleCurveAddPoint(layer === 'wrap' ? 150 : 50, 20);
    const after = snapshot();
    const record = editUndoStack[editUndoStack.length - 1];
    assert.strictEqual(record.dragLayer, layer);
    assert.strictEqual(record.traceKey, layer === 'wrap' ? 'GR_wrap' : 'GR');
    if (layer === 'wrap') {
        assert.deepStrictEqual(after.mainValues, before.mainValues);
        assert.deepStrictEqual(after.mainTrace, before.mainTrace);
        assert.notDeepStrictEqual(after.wrapValues, before.wrapValues);
        assert.notDeepStrictEqual(after.wrapTrace, before.wrapTrace);
    } else {
        assert.deepStrictEqual(after.wrapValues, before.wrapValues);
        assert.deepStrictEqual(after.wrapTrace, before.wrapTrace);
        assert.notDeepStrictEqual(after.mainValues, before.mainValues);
        assert.notDeepStrictEqual(after.mainTrace, before.mainTrace);
    }
    undoLastCurveEdit();
    assert.deepStrictEqual(snapshot(), before);
}

reset('wrap', { missingWrapDepth: 19 });
const isolatedMain = lastDigitizedCurves.GR.values.slice();
handleCurveAddPoint(150, 20);
assert.strictEqual(lastDigitizedCurves.GR.wrapLayer.values[19], null);
assert.ok(lastDigitizedCurves.GR.wrapLayer.values[18] > 100);
assert.deepStrictEqual(lastDigitizedCurves.GR.values, isolatedMain);

for (const layer of ['main', 'wrap']) {
    reset(layer, { lockedLayer: layer });
    const before = snapshot();
    handleCurveAddPoint(50, 20);
    assert.deepStrictEqual(snapshot(), before);
    assert.strictEqual(editUndoStack.length, 0);
}

reset('wrap', { pinnedDepths: [19] });
const beforePinned = snapshot();
handleCurveAddPoint(150, 20);
assert.strictEqual(lastDigitizedCurves.GR.wrapLayer.values[19], beforePinned.wrapValues[19]);
assert.deepStrictEqual(lastDigitizedCurves.GR.values, beforePinned.mainValues);
assert.notStrictEqual(lastDigitizedCurves.GR.wrapLayer.values[18], beforePinned.wrapValues[18]);
undoLastCurveEdit();
assert.deepStrictEqual(snapshot(), beforePinned);
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((add_source, undo_source, harness)),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_active_layer_helpers_prioritize_explicit_and_interaction_state():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")

    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    trace_helper = _function_source(source, "getActiveCurveTracePoints", "getCurrentPinKey")
    effective_helper = _function_source(source, "_getEffectiveActiveLayer", "isLayerLocked")
    lock_helper = _function_source(source, "isPointLocked", "isPointLockedByRawX")
    raw_lock_helper = _function_source(source, "isPointLockedByRawX", "perturbIfEqual")
    values_helper = _function_source(source, "getActiveLayerValues", "getActiveLayerTrackConfig")
    track_helper = _function_source(source, "getActiveLayerTrackConfig", "pixelToTrackValue")
    harness = r"""
const assert = require('assert');
const window = { currentDragLayer: 'wrap', paintTargetLayer: 'wrap' };
let editDragState = { dragLayer: 'main' };
let drawStrokeActive = false;
let drawStrokeLayer = null;
let paintStrokeActive = false;
let paintStrokeLayer = null;
let smoothStrokeActive = false;
let smoothStrokeLayer = null;
let eraserStart = null;
let eraserStrokeLayer = null;
let lastCurveTraces = { GR: [[1]], GR_wrap: [[2]] };
function getActiveCurveTraceKey() { return 'GR'; }
function isMainLineLocked() { return false; }
function isWrapLineLocked() { return true; }

const entry = {
    values: [10],
    wrapLayer: { values: [110], left_value: 100, right_value: 200 },
};
const track = { scaleMin: 0, scaleMax: 100 };

assert.strictEqual(_getEffectiveActiveLayer(), 'main');
assert.strictEqual(_getEffectiveActiveLayer('wrap'), 'wrap');
assert.strictEqual(getActiveLayerValues(entry, 'main'), entry.values);
assert.strictEqual(getActiveLayerValues(entry, 'wrap'), entry.wrapLayer.values);
assert.strictEqual(getActiveLayerTrackConfig(entry, track, 'main'), track);
assert.deepStrictEqual(
    [getActiveLayerTrackConfig(entry, track, 'wrap').scaleMin, getActiveLayerTrackConfig(entry, track, 'wrap').scaleMax],
    [100, 200],
);
assert.strictEqual(getActiveCurveTracePoints('GR', 'main').traceKey, 'GR');
assert.strictEqual(getActiveCurveTracePoints('GR', 'wrap').traceKey, 'GR_wrap');
assert.strictEqual(isPointLocked(10, track, entry, 'main'), false);
assert.strictEqual(isPointLocked(110, track, entry, 'wrap'), true);
assert.strictEqual(isPointLockedByRawX(50, track, entry, 'main'), false);
assert.strictEqual(isPointLockedByRawX(150, track, entry, 'wrap'), true);
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((
            trace_helper,
            effective_helper,
            lock_helper,
            raw_lock_helper,
            values_helper,
            track_helper,
            harness,
        )),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_edit_interactions_store_pass_and_clear_frozen_layers():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    begin_source = _function_source(source, "beginCurveEditInteraction", "handleCurveEditDragMove")
    drag_source = _drag_source(source)
    pointer_start = source.index("imageHitTarget.addEventListener('pointerdown'")
    pointer_end = source.index("async function handleFile", pointer_start)
    pointer_source = source[pointer_start:pointer_end]

    assert "dragLayer: selectedLayer" in begin_source
    assert "getActiveCurveTracePoints(editCurveId, selectedLayer)" in begin_source
    assert "getActiveLayerValues(entry, selectedLayer)" in begin_source
    assert "getActiveLayerValues(entry, state.dragLayer)" in drag_source
    assert "getActiveLayerTrackConfig(entry, previewTrack, activeLayer)" in drag_source
    assert "drawStrokeLayer = targetLayer;" in pointer_source
    assert "paintStrokeLayer = targetLayer;" in pointer_source
    assert "smoothStrokeLayer = targetLayer;" in pointer_source
    assert "clearEditInteractionLayerState();" in pointer_source
    assert "window.currentDragLayer = null;" in source
    assert "window.paintTargetLayer = null;" in source


def test_layer_specific_rebuild_does_not_persist_the_inactive_trace():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    rebuild_source = _function_source(source, "buildDisplayTracePointsFromDigitized", "rebuildCurveTraceFromDigitized")

    assert "if (explicitLayer !== 'wrap')" in rebuild_source
    assert "lastCurveTraces[traceKey] = points;" in rebuild_source
    assert "if (explicitLayer !== 'main')" in rebuild_source
    assert "lastCurveTraces[traceKey + '_wrap'] = wrapPoints;" in rebuild_source

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
let editCurveId = 'GR';
let editCurveDataKey = 'GR';
let lastDigitizedDepth = [0, 1, 2];
let lastDepthConfig = {};
let lastCurveTraces;
const entry = {
    values: [10, 20, 30],
    wrapLayer: { values: [110, 120, 130], left_value: 100, right_value: 200 },
};
function normalizeCurveKey(value) { return value; }
function getDepthConfigFromInputs() { return {}; }
function findTrackByCurveId() { return { leftX: 0, rightX: 200, scaleMin: 0, scaleMax: 200, wrapped: false }; }
function findDigitizedCurveEntry() { return { key: 'GR', entry }; }
function resolveTraceKeyForCurveId() { return 'GR'; }
function isMissingDigitizedValue(value) { return value == null; }
function trackValueToPixelX(value) { return value; }
function getPixelYForDepthIndex(idx) { return idx; }
function wrapRawXToTrack(value) { return value; }
function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }

lastCurveTraces = { GR: [['old-main']], GR_wrap: [['old-wrap']] };
const wrapBeforeMainRebuild = JSON.stringify(lastCurveTraces.GR_wrap);
buildDisplayTracePointsFromDigitized('GR', { persist: true, explicitLayer: 'main' });
assert.strictEqual(JSON.stringify(lastCurveTraces.GR_wrap), wrapBeforeMainRebuild);
assert.notDeepStrictEqual(lastCurveTraces.GR, [['old-main']]);

lastCurveTraces = { GR: [['old-main']], GR_wrap: [['old-wrap']] };
const mainBeforeWrapRebuild = JSON.stringify(lastCurveTraces.GR);
buildDisplayTracePointsFromDigitized('GR', { persist: true, explicitLayer: 'wrap' });
assert.strictEqual(JSON.stringify(lastCurveTraces.GR), mainBeforeWrapRebuild);
assert.notDeepStrictEqual(lastCurveTraces.GR_wrap, [['old-wrap']]);
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((rebuild_source, harness)),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_curve_hit_testing_selects_nearest_geometry_without_lock_filtering():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    best_source = _function_source(source, "findBestPointIndexForDepth", "findEditablePointIndex")
    hit_source = _function_source(source, "findEditablePointIndex", "resolvePinKey")

    assert "isPointLocked" not in best_source
    assert "isPointLocked" not in hit_source
    assert "isDepthIndexPinned" not in best_source
    assert "isDepthIndexPinned" not in hit_source

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
let lastDigitizedCurves = { C: { values: [10, 20] } };
function ensureTracePointsHaveDepthIndex() {}
function findTrackByCurveId() { return {}; }
function getDepthIndexAtImageY() { return -1; }

let points = [[2, 0, 0], [8, 0, 1]];
let picked = findEditablePointIndex(points, 0, 0, 'C', { maxYDist: 14, maxDist: 24, explicitLayer: 'main' });
assert.strictEqual(points[picked.pointIndex][2], 0);
assert.strictEqual(picked.layer, 'main');

points = [[8, 0, 1], [2, 0, 0]];
picked = findEditablePointIndex(points, 0, 0, 'C', { maxYDist: 14, maxDist: 24, explicitLayer: 'main' });
assert.strictEqual(points[picked.pointIndex][2], 0);

picked = findEditablePointIndex(points, 100, 100, 'C', { maxYDist: 14, maxDist: 24, explicitLayer: 'main' });
assert.strictEqual(picked.pointIndex, -1);
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((best_source, hit_source, harness)),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_blocked_nearest_delete_does_not_fall_through_or_create_undo():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    best_source = _function_source(source, "findBestPointIndexForDepth", "findEditablePointIndex")
    hit_source = _function_source(source, "findEditablePointIndex", "resolvePinKey")
    delete_source = _function_source(source, "handleCurveEditDelete", "applySmoothCurve")

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
let editMode = true;
let editCurveId = 'C';
let editCurveDataKey = 'C';
let lastDigitizedDepth = [100, 101];
let lastDigitizedCurves = {
    C: { values: [11, 22], wrapLayer: { values: [111, 222] } },
};
let lastCurveTraces = {
    C: [[2, 0, 0], [8, 0, 1]],
    C_wrap: [[2, 0, 0], [8, 0, 1]],
};
let editUndoStack = [];
let lastNullValue = null;
let blockedMode = 'locked';
let statuses = [];
function ensureTracePointsHaveDepthIndex() {}
function findTrackByCurveId() { return { leftX: 0, rightX: 100 }; }
function getDepthIndexAtImageY() { return -1; }
function resolveActiveLayer() { return 'main'; }
function getActiveLayerValues(entry, layer) { return layer === 'wrap' ? entry.wrapLayer.values : entry.values; }
function getActiveLayerTrackConfig(entry, track) { return track; }
function getActiveCurveTracePoints(curveId, layer) {
    const traceKey = layer === 'wrap' ? curveId + '_wrap' : curveId;
    return { traceKey, points: lastCurveTraces[traceKey] };
}
function getActiveCurveTraceKey() { return 'C'; }
function isLayerLocked() { return blockedMode === 'layer'; }
function isDepthIndexPinned(curveId, idx) { return blockedMode === 'pinned' && idx === 0; }
function isPointLocked(value) { return blockedMode === 'locked' && (value === 11 || value === 111); }
function showStatus(message) { statuses.push(message); }
function getDepthConfigFromInputs() { return {}; }
function pixelToDepthFromConfig() { return 100; }
function findNearestDepthIndex() { return 0; }
function fillDeletedDepthSpan() {}
function rebuildCurveTraceFromDigitized() {}
function renderCurveTraceOverlays() {}
function commitInPlaceCurveLayerMutation(curveId, layer, changed, options = {}) {
    if (changed && options.rebuildTrace !== false) rebuildCurveTraceFromDigitized(curveId, layer);
    return { changed };
}

const snapshot = () => JSON.stringify({ curves: lastDigitizedCurves, traces: lastCurveTraces });
let before = snapshot();
handleCurveEditDelete(0, 0, 'main');
assert.strictEqual(snapshot(), before);
assert.strictEqual(editUndoStack.length, 0);
assert(statuses.at(-1).includes('locked'));

lastCurveTraces.C.reverse();
before = snapshot();
handleCurveEditDelete(0, 0, 'main');
assert.strictEqual(snapshot(), before);
assert.strictEqual(editUndoStack.length, 0);

blockedMode = 'pinned';
before = snapshot();
handleCurveEditDelete(0, 0, 'main');
assert.strictEqual(snapshot(), before);
assert.strictEqual(editUndoStack.length, 0);
assert(statuses.at(-1).includes('pinned'));

blockedMode = 'layer';
before = snapshot();
handleCurveEditDelete(0, 0, 'main');
assert.strictEqual(snapshot(), before);
assert.strictEqual(editUndoStack.length, 0);

blockedMode = 'locked';
before = snapshot();
handleCurveEditDelete(0, 0, 'wrap');
assert.strictEqual(snapshot(), before);
assert.strictEqual(editUndoStack.length, 0);
assert(statuses.at(-1).includes('locked'));

blockedMode = 'none';
const wrapBefore = JSON.stringify({ values: lastDigitizedCurves.C.wrapLayer.values, trace: lastCurveTraces.C_wrap });
handleCurveEditDelete(8, 0, 'main');
assert.strictEqual(lastDigitizedCurves.C.values[1], null);
assert.strictEqual(JSON.stringify({ values: lastDigitizedCurves.C.wrapLayer.values, trace: lastCurveTraces.C_wrap }), wrapBefore);
assert.strictEqual(editUndoStack.length, 1);
assert.strictEqual(editUndoStack[0].dragLayer, 'main');

const undoCount = editUndoStack.length;
handleCurveEditDelete(100, 100, 'wrap');
assert.strictEqual(editUndoStack.length, undoCount);
assert(statuses.at(-1).includes('No editable curve point'));
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((best_source, hit_source, delete_source, harness)),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_pin_toggle_targets_the_true_nearest_point_even_when_already_pinned():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    toggle_source = _function_source(source, "togglePinAtImageCoords", "restorePinnedPointsAfterRedigitize")

    assert "bestUnpinned" not in toggle_source
    assert "toggleDepthIndexPinned(curveId, depthIdx)" in toggle_source

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
let editMode = true;
let editCurveId = 'C';
let editCurveDataKey = 'C';
let lastDigitizedCurves = { C: { values: [1, 2] } };
let lastCurveTraces = { C: [[1, 0, 0], [5, 0, 1]] };
let toggled = null;
function findTrackByCurveId() { return {}; }
function resolveActiveLayer() { return 'main'; }
function getActiveCurveTracePoints() { return { points: lastCurveTraces.C }; }
function ensureTracePointsHaveDepthIndex() {}
function toggleDepthIndexPinned(curveId, depthIdx) { toggled = [curveId, depthIdx]; return false; }
function renderCurveTraceOverlays() {}
function showStatus() {}
togglePinAtImageCoords(0, 0);
assert.deepStrictEqual(toggled, ['C', 0]);
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((toggle_source, harness)),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_drag_hit_authorization_precedes_trace_mutation_and_undo():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    begin_source = _function_source(source, "beginCurveEditInteraction", "handleCurveEditDragMove")
    regular_source = begin_source[begin_source.index("const activeTrace = getActiveCurveTracePoints") :]

    hit = regular_source.index("const picked = findEditablePointIndex")
    layer_check = regular_source.index("if (isLayerLocked(selectedLayer, entry))", hit)
    pin_check = regular_source.index("isDepthIndexPinned(editCurveId, centerDepthIdx)", layer_check)
    point_check = regular_source.index("isPointLocked(activeValues[centerDepthIdx]", pin_check)
    drag_state = regular_source.index("editDragState = {", point_check)

    assert hit < layer_check < pin_check < point_check < drag_state
    assert "lastCurveTraces[traceKey] = pts;" not in regular_source
    assert "pts = pts.map(pt => Array.isArray(pt) ? pt.slice() : pt);" in regular_source
    assert "No editable curve point found near the cursor." in regular_source


def test_click_bezier_and_eraser_authorize_after_target_selection():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    click_source = _function_source(source, "handleCurveEditClick", "handleDrawSample")
    begin_source = _function_source(source, "beginCurveEditInteraction", "handleCurveEditDragMove")

    click_hit = click_source.index("const picked = findEditablePointIndex")
    click_layer = click_source.index("if (isLayerLocked(activeLayer, entry))", click_hit)
    click_pin = click_source.index("isDepthIndexPinned(editCurveId, idx)", click_layer)
    click_lock = click_source.index("isPointLocked(oldValue, targetTrack, entry, activeLayer)", click_pin)
    click_undo = click_source.index("editUndoStack.push", click_lock)
    assert click_hit < click_layer < click_pin < click_lock < click_undo

    bezier_pick = begin_source.index("if (bestSegIdx >= 0)")
    bezier_lock = begin_source.index("if (isLayerLocked(selectedLayer, entry))", bezier_pick)
    bezier_undo = begin_source.index("editUndoStack.push", bezier_lock)
    assert bezier_pick < bezier_lock < bezier_undo

    assert "if (isDepthIndexPinned(editCurveId, idx))" in source
    eraser_pin = source.index("if (isDepthIndexPinned(editCurveId, idx))", source.index("if (activeTool === 'eraser'"))
    eraser_lock = source.index("isPointLockedByRawX(rawX, track, entry, deleteLayer)", eraser_pin)
    eraser_mutation = source.index("activeDeleteValues[idx] = null", eraser_lock)
    assert eraser_pin < eraser_lock < eraser_mutation


def test_gap_aware_smoothing_excludes_all_missing_representations():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    missing_source = _function_source(source, "isMissingDigitizedValue", "fillDeletedDepthSpan")
    smooth_helper = _function_source(source, "getGapAwareSmoothedValue", "handleSmoothSample")

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
let lastNullValue = -999.25;

assert.strictEqual(getGapAwareSmoothedValue([80, 82, -999.25, 84, 86], 1, 8), 81);
assert.strictEqual(getGapAwareSmoothedValue([80, 82, -999.25, 84, 86], 2, 8), -999.25);
assert.strictEqual(getGapAwareSmoothedValue([80, 82, null, 84, 86], 2, 8), null);
assert.strictEqual(getGapAwareSmoothedValue([80, 81, 82, -999.25, 120, 121, 122], 2, 8), 81);
assert.strictEqual(getGapAwareSmoothedValue([80, 81, 82, -999.25, 120, 121, 122], 4, 8), 121);
assert.strictEqual(getGapAwareSmoothedValue([80, 81, -999.25, -999.25, 120, 121], 1, 20), 80.5);
assert.strictEqual(getGapAwareSmoothedValue([-999.25, -999.25, 75, -999.25], 2, 20), 75);
assert.strictEqual(getGapAwareSmoothedValue([-999.25, -999.25], 0, 4), -999.25);
assert.strictEqual(getGapAwareSmoothedValue([80, 82], 0, 8), 81);
assert(Number.isNaN(getGapAwareSmoothedValue([80, NaN, 82], 1, 8)));

lastNullValue = -9999;
assert.strictEqual(getGapAwareSmoothedValue([80, -9999, 120], 1, 8), -9999);
assert.strictEqual(getGapAwareSmoothedValue([80, -9999, 120], 0, 8), 80);

lastNullValue = -999.25;
const asymmetric = [10, 20, 40, 80, 160];
function smoothPass(order) {
    const stableSource = asymmetric.slice();
    const output = stableSource.slice();
    order.forEach(idx => { output[idx] = getGapAwareSmoothedValue(stableSource, idx, 2); });
    return output;
}
assert.deepStrictEqual(smoothPass([0, 1, 2, 3, 4]), smoothPass([4, 3, 2, 1, 0]));
assert(smoothPass([0, 1, 2, 3, 4]).every(Number.isFinite));
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((missing_source, smooth_helper, harness)),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_smooth_brush_preserves_gaps_pins_locks_and_selected_layer():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    missing_source = _function_source(source, "isMissingDigitizedValue", "fillDeletedDepthSpan")
    smooth_helper = _function_source(source, "getGapAwareSmoothedValue", "handleSmoothSample")
    brush_source = _function_source(source, "handleSmoothSample", "commitSmoothStroke")

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
let lastNullValue = -999.25;
let smoothStrokeActive = true;
let smoothStrokeLayer = 'main';
let smoothLastProcessedIdx = null;
let smoothUndoValuesBefore = [];
let editMode = true;
let editCurveId = 'C';
let editCurveDataKey = 'C';
let lastDepthConfig = {};
let lastDigitizedDepth = [0, 1, 2, 3, 4];
let lastCurveTraces = {};
let lockedLayer = null;
let pinned = new Set();
let entry = {
    values: [80, 82, -999.25, 84, 86],
    wrapLayer: { values: [180, 182, -999.25, 184, 186] },
};
let lastDigitizedCurves = { C: entry };
const document = { getElementById: () => ({ value: '1' }) };
function findTrackByCurveId() { return {}; }
function getDepthConfigFromInputs() { return {}; }
function pixelToDepthFromConfig(y) { return y; }
function findNearestDepthIndex(depth) { return Math.round(depth); }
function getActiveLayerValues(target, layer) { return layer === 'wrap' ? target.wrapLayer.values : target.values; }
function isLayerLocked(layer) { return layer === lockedLayer; }
function isDepthIndexPinned(curveId, idx) { return pinned.has(idx); }
function isPointLocked() { return false; }
function perturbIfEqual() {}
function rebuildCurveTraceFromDigitized() {}
function renderCurveTraceOverlays() {}

const wrapBefore = JSON.stringify(entry.wrapLayer.values);
handleSmoothSample(0, 2, true);
assert.strictEqual(entry.values[2], -999.25);
assert.strictEqual(smoothUndoValuesBefore.length, 0);

handleSmoothSample(0, 1, true);
assert(entry.values[1] >= 80 && entry.values[1] <= 82);
assert.strictEqual(entry.values[2], -999.25);
assert.strictEqual(JSON.stringify(entry.wrapLayer.values), wrapBefore);
assert.deepStrictEqual(smoothUndoValuesBefore, [{ idx: 1, oldValue: 82 }]);

const pinnedBefore = entry.values[0];
pinned.add(0);
handleSmoothSample(0, 0, true);
assert.strictEqual(entry.values[0], pinnedBefore);

smoothUndoValuesBefore = [];
entry.values = [-999.25, -999.25, 75, -999.25, -999.25];
handleSmoothSample(0, 2, true);
assert.strictEqual(entry.values[2], 75);
assert.strictEqual(smoothUndoValuesBefore.length, 0);

entry.values = [80, 82, null, 84, 86];
handleSmoothSample(0, 2, true);
assert.strictEqual(entry.values[2], null);

lockedLayer = 'main';
smoothUndoValuesBefore = [];
const lockedBefore = JSON.stringify(entry.values);
handleSmoothSample(0, 1, true);
assert.strictEqual(JSON.stringify(entry.values), lockedBefore);
assert.strictEqual(smoothUndoValuesBefore.length, 0);

lockedLayer = null;
smoothStrokeLayer = 'wrap';
smoothUndoValuesBefore = [];
const mainBeforeWrap = JSON.stringify(entry.values);
handleSmoothSample(0, 1, true);
assert.strictEqual(JSON.stringify(entry.values), mainBeforeWrap);
assert(entry.wrapLayer.values[1] >= 180 && entry.wrapLayer.values[1] <= 182);
assert.strictEqual(entry.wrapLayer.values[2], -999.25);
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((missing_source, smooth_helper, brush_source, harness)),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_apply_smooth_curve_uses_stable_values_and_exact_layer_undo():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    missing_source = _function_source(source, "isMissingDigitizedValue", "fillDeletedDepthSpan")
    smooth_helper = _function_source(source, "getGapAwareSmoothedValue", "handleSmoothSample")
    command_source = _function_source(source, "applySmoothCurve", "applyCenterCurve")

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
let lastNullValue = -999.25;
let editMode = true;
let editCurveId = 'C';
let lastMouseImgX = 10;
let lastMouseImgY = 2;
let selectedLayer = 'main';
let lockedLayer = null;
let pinned = new Set();
let editUndoStack = [];
let statuses = [];
const entry = {
    values: [80, 82, -999.25, 84, 86],
    wrapLayer: { values: [180, 182, -999.25, 184, 186] },
};
let traces = {
    C: [[80, 0, 0], [82, 1, 1], [84, 3, 3], [86, 4, 4]],
    C_wrap: [[180, 0, 0], [182, 1, 1], [184, 3, 3], [186, 4, 4]],
};
let lastCurveTraces = traces;
const document = { getElementById: () => ({ value: '8' }) };
function findDigitizedCurveEntry() { return { key: 'C', entry }; }
function findTrackByCurveId() { return {}; }
function resolveActiveLayer() { return selectedLayer; }
function isLayerLocked(layer) { return layer === lockedLayer; }
function getActiveCurveTracePoints(curveId, layer) {
    const traceKey = layer === 'wrap' ? 'C_wrap' : 'C';
    return { traceKey, points: traces[traceKey] };
}
function ensureTracePointsHaveDepthIndex() {}
function getActiveLayerValues(target, layer) { return layer === 'wrap' ? target.wrapLayer.values : target.values; }
function isDepthIndexPinned(curveId, idx) { return pinned.has(idx); }
function isPointLocked() { return false; }
function perturbIfEqual() {}
function rebuildCurveTraceFromDigitized() {}
function renderCurveTraceOverlays() {}
function showStatus(message) { statuses.push(message); }
function commitInPlaceCurveLayerMutation(curveId, layer, changed, options = {}) {
    if (changed && options.rebuildTrace !== false) rebuildCurveTraceFromDigitized(curveId, layer);
    return { changed };
}

const mainBefore = entry.values.slice();
const wrapBefore = JSON.stringify({ values: entry.wrapLayer.values, trace: traces.C_wrap });
applySmoothCurve();
assert.strictEqual(entry.values[2], -999.25);
assert(entry.values.filter(v => v !== -999.25).every(v => Number.isFinite(v) && v >= 80 && v <= 86));
assert.strictEqual(JSON.stringify({ values: entry.wrapLayer.values, trace: traces.C_wrap }), wrapBefore);
assert.strictEqual(editUndoStack.length, 1);
assert.strictEqual(editUndoStack[0].dragLayer, 'main');
assert.deepStrictEqual(editUndoStack[0].undoValues.map(item => item.oldValue), mainBefore);
editUndoStack[0].undoValues.forEach(item => { entry.values[item.idx] = item.oldValue; });
assert.deepStrictEqual(entry.values, mainBefore);

editUndoStack = [];
pinned = new Set([0, 2]);
applySmoothCurve();
assert.strictEqual(entry.values[0], 80);
assert.strictEqual(entry.values[2], -999.25);

editUndoStack = [];
selectedLayer = 'wrap';
lockedLayer = 'wrap';
const lockedBefore = JSON.stringify({ main: entry.values, wrap: entry.wrapLayer.values, traces });
applySmoothCurve();
assert.strictEqual(JSON.stringify({ main: entry.values, wrap: entry.wrapLayer.values, traces }), lockedBefore);
assert.strictEqual(editUndoStack.length, 0);

lockedLayer = null;
pinned = new Set();
const mainBeforeWrap = JSON.stringify({ values: entry.values, trace: traces.C });
applySmoothCurve();
assert.strictEqual(JSON.stringify({ values: entry.values, trace: traces.C }), mainBeforeWrap);
assert.strictEqual(entry.wrapLayer.values[2], -999.25);
assert(entry.wrapLayer.values.filter(v => v !== -999.25).every(v => Number.isFinite(v) && v >= 180 && v <= 186));
assert.strictEqual(editUndoStack.length, 1);
assert.strictEqual(editUndoStack[0].dragLayer, 'wrap');
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((missing_source, smooth_helper, command_source, harness)),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_x_shift_is_atomic_layer_safe_pin_safe_and_exactly_undoable():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    missing_source = _function_source(source, "isMissingDigitizedValue", "fillDeletedDepthSpan")
    context_source = _function_source(source, "createCurveShiftContext", "applyCurveXShiftForCurve")
    x_source = _function_source(source, "applyCurveXShiftForCurve", "applyCurveYShiftForCurve")
    undo_source = _function_source(source, "undoLastCurveEdit", "beginCurveEditInteraction")

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
const window = {};
let lastNullValue = -999.25;
let selectedLayer = 'main';
let lockedLayer = null;
let pinned = new Set([4]);
let editUndoStack = [];
let statuses = [];
const entry = {
    values: [10, null, -999.25, 30, 40],
    wrapLayer: { values: [110, null, -999.25, 130, 140], left_value: 100, right_value: 300 },
};
let lastDigitizedCurves = { C: entry };
let lastCurveTraces = {
    C: [[10, 0, 0], [30, 3, 3], [40, 4, 4]],
    C_wrap: [[5, 0, 0], [15, 3, 3], [20, 4, 4]],
};
let shiftValue = 10;
const document = { getElementById: id => id.startsWith('curveShift') ? { value: String(shiftValue) } : null };
function getTrackCalibrationsSnapshot() { return [{ index: 0, id: 'C', leftX: 0, rightX: 100, scaleMin: 0, scaleMax: 100 }]; }
function findDigitizedCurveEntry() { return { key: 'C', entry }; }
function _getEffectiveActiveLayer() { return selectedLayer; }
function getActiveLayerValues(target, layer) { return layer === 'wrap' ? target.wrapLayer.values : target.values; }
function getCurveLayerContext(curveId, layer) { return { values: getActiveLayerValues(entry, layer) }; }
function commitCurveLayerValues(curveId, layer, nextValues, options = {}) {
    const values = getActiveLayerValues(entry, layer);
    values.splice(0, values.length, ...nextValues);
    if (options.rebuildTrace !== false) rebuildCurveTraceFromDigitized(curveId, layer);
    return { changed: true, values };
}
function markCurveLayerTraceRevision() {}
function getActiveLayerTrackConfig(target, track, layer) {
    return layer === 'wrap' ? { ...track, scaleMin: 100, scaleMax: 300 } : track;
}
function getActiveCurveTracePoints(curveId, layer) {
    const traceKey = layer === 'wrap' ? 'C_wrap' : 'C';
    return { traceKey, points: lastCurveTraces[traceKey] };
}
function isLayerLocked(layer) { return layer === lockedLayer; }
function isDepthIndexPinned(curveId, idx) { return pinned.has(idx); }
function isPointLocked(value) { return value === 30 || value === 130; }
function trackValueToPixelX(value, track) { return (value - track.scaleMin) * 100 / (track.scaleMax - track.scaleMin); }
function pixelToTrackValue(x, track) { return track.scaleMin + x * (track.scaleMax - track.scaleMin) / 100; }
function rebuildCurveTraceFromDigitized(curveId, layer) {
    const traceKey = layer === 'wrap' ? 'C_wrap' : 'C';
    const values = getActiveLayerValues(entry, layer);
    lastCurveTraces[traceKey] = values.flatMap((value, idx) => isMissingDigitizedValue(value) ? [] : [[value, idx, idx]]);
}
function renderCurveTraceOverlays() {}
function showStatus(message) { statuses.push(message); }
function logCorrectionEvent() {}

const mainBefore = entry.values.slice();
const mainTraceBefore = JSON.stringify(lastCurveTraces.C);
const wrapBefore = JSON.stringify({ values: entry.wrapLayer.values, trace: lastCurveTraces.C_wrap });
applyCurveXShiftForCurve(0);
assert.deepStrictEqual(entry.values, [20, null, -999.25, 30, 40]);
assert.strictEqual(JSON.stringify({ values: entry.wrapLayer.values, trace: lastCurveTraces.C_wrap }), wrapBefore);
assert.strictEqual(editUndoStack.length, 1);
assert.strictEqual(editUndoStack[0].label, 'shift_x');
assert.strictEqual(editUndoStack[0].dragLayer, 'main');
undoLastCurveEdit();
assert.deepStrictEqual(entry.values, mainBefore);
assert.strictEqual(JSON.stringify(lastCurveTraces.C), mainTraceBefore);
assert.strictEqual(JSON.stringify({ values: entry.wrapLayer.values, trace: lastCurveTraces.C_wrap }), wrapBefore);

selectedLayer = 'wrap';
pinned = new Set([4]);
const mainBeforeWrap = JSON.stringify({ values: entry.values, trace: lastCurveTraces.C });
applyCurveXShiftForCurve(0);
assert.deepStrictEqual(entry.wrapLayer.values, [130, null, -999.25, 130, 140]);
assert.strictEqual(JSON.stringify({ values: entry.values, trace: lastCurveTraces.C }), mainBeforeWrap);
undoLastCurveEdit();
assert.deepStrictEqual(entry.wrapLayer.values, [110, null, -999.25, 130, 140]);

lockedLayer = 'wrap';
const undoCount = editUndoStack.length;
const lockedBefore = JSON.stringify({ values: entry.wrapLayer.values, trace: lastCurveTraces.C_wrap });
applyCurveXShiftForCurve(0);
assert.strictEqual(JSON.stringify({ values: entry.wrapLayer.values, trace: lastCurveTraces.C_wrap }), lockedBefore);
assert.strictEqual(editUndoStack.length, undoCount);

lockedLayer = null;
shiftValue = 0;
applyCurveXShiftForCurve(0);
assert.strictEqual(editUndoStack.length, undoCount);
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((missing_source, context_source, x_source, undo_source, harness)),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_shared_depth_collision_resolver_is_deterministic_and_propagates_blockage():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    missing_source = _function_source(source, "isMissingDigitizedValue", "fillDeletedDepthSpan")
    context_source = _function_source(source, "createCurveShiftContext", "applyCurveXShiftForCurve")

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
let lastNullValue = -999.25;
function make(source, destination, roundingError, movementDistance = 1) {
    return { sourceDepthIndex: source, destinationDepthIndex: destination, roundingError, movementDistance, accepted: true, rejectionReason: null };
}
function duplicateWinner(order) {
    const proposals = order.map(source => source === 0 ? make(0, 2, 0.2) : make(1, 2, 0.1));
    resolveDepthMoveProposals(proposals, [10, 20, -999.25], compareYShiftProposalPriority);
    return proposals.find(proposal => proposal.accepted).sourceDepthIndex;
}
assert.strictEqual(duplicateWinner([0, 1]), 1);
assert.strictEqual(duplicateWinner([1, 0]), 1);

const blocked = [make(0, 1, 0), make(1, 2, 0), { ...make(2, 2, 0, 0), accepted: false, rejectionReason: 'stationary' }];
resolveDepthMoveProposals(blocked, [10, 20, 30], compareYShiftProposalPriority);
assert(blocked.every(proposal => !proposal.accepted));
assert.strictEqual(blocked[1].rejectionReason, 'destination-not-vacated');
assert.strictEqual(blocked[0].rejectionReason, 'destination-not-vacated');

const chain = [make(0, 1, 0), make(1, 2, 0)];
resolveDepthMoveProposals(chain, [10, 20, -999.25], compareYShiftProposalPriority);
assert(chain.every(proposal => proposal.accepted));

const external = [make(0, 1, 0)];
resolveDepthMoveProposals(external, [10, 99], compareYShiftProposalPriority);
assert.strictEqual(external[0].accepted, false);
assert.strictEqual(external[0].rejectionReason, 'destination-not-vacated');
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((missing_source, context_source, harness)),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_y_shift_moves_sparse_chains_without_loss_and_preserves_layers_and_undo():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    missing_source = _function_source(source, "isMissingDigitizedValue", "fillDeletedDepthSpan")
    context_source = _function_source(source, "createCurveShiftContext", "applyCurveXShiftForCurve")
    y_source = _function_source(source, "applyCurveYShiftForCurve", "toggleCurveEditMode")
    undo_source = _function_source(source, "undoLastCurveEdit", "beginCurveEditInteraction")

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
const window = {};
let lastNullValue = -999.25;
let lastDepthConfig = {};
let lastDigitizedDepth = [0, 1, 2, 3, 4];
let selectedLayer = 'main';
let lockedLayer = null;
let pinned = new Set();
let editUndoStack = [];
let statuses = [];
let shiftValue = 1;
const entry = {
    values: [10, -999.25, 20, -999.25, -999.25],
    wrapLayer: { values: [110, -999.25, 120, -999.25, -999.25] },
};
let lastDigitizedCurves = { C: entry };
let lastCurveTraces = {
    C: [[10, 0, 0], [20, 2, 2]],
    C_wrap: [[110, 0, 0], [120, 2, 2]],
};
const document = { getElementById: id => id.startsWith('curveShiftY') ? { value: String(shiftValue) } : null };
function getTrackCalibrationsSnapshot() { return [{ index: 0, id: 'C' }]; }
function findDigitizedCurveEntry() { return { key: 'C', entry }; }
function _getEffectiveActiveLayer() { return selectedLayer; }
function getActiveLayerValues(target, layer) { return layer === 'wrap' ? target.wrapLayer.values : target.values; }
function getCurveLayerContext(curveId, layer) { return { values: getActiveLayerValues(entry, layer) }; }
function commitCurveLayerValues(curveId, layer, nextValues, options = {}) {
    const values = getActiveLayerValues(entry, layer);
    values.splice(0, values.length, ...nextValues);
    if (options.rebuildTrace !== false) rebuildCurveTraceFromDigitized(curveId, layer);
    return { changed: true, values };
}
function markCurveLayerTraceRevision() {}
function getActiveLayerTrackConfig(target, track) { return track; }
function getActiveCurveTracePoints(curveId, layer) {
    const traceKey = layer === 'wrap' ? 'C_wrap' : 'C';
    return { traceKey, points: lastCurveTraces[traceKey] };
}
function isLayerLocked(layer) { return layer === lockedLayer; }
function isDepthIndexPinned(curveId, idx) { return pinned.has(idx); }
function isPointLocked() { return false; }
function getDepthConfigFromInputs() { return {}; }
function getPixelYForDepthIndex(idx) { return idx; }
function pixelToDepthFromConfig(y) { return y; }
function findNearestDepthIndex(depth) {
    let best = 0;
    for (let i = 1; i < lastDigitizedDepth.length; i++) {
        if (Math.abs(lastDigitizedDepth[i] - depth) < Math.abs(lastDigitizedDepth[best] - depth)) best = i;
    }
    return best;
}
function rebuildCurveTraceFromDigitized(curveId, layer) {
    const traceKey = layer === 'wrap' ? 'C_wrap' : 'C';
    const values = getActiveLayerValues(entry, layer);
    lastCurveTraces[traceKey] = values.flatMap((value, idx) => isMissingDigitizedValue(value) ? [] : [[value, idx, idx]]);
}
function renderCurveTraceOverlays() {}
function showStatus(message) { statuses.push(message); }
function logCorrectionEvent() {}

const mainBefore = entry.values.slice();
const mainTraceBefore = JSON.stringify(lastCurveTraces.C);
const wrapBefore = JSON.stringify({ values: entry.wrapLayer.values, trace: lastCurveTraces.C_wrap });
applyCurveYShiftForCurve(0);
assert.deepStrictEqual(entry.values, [-999.25, 10, -999.25, 20, -999.25]);
assert.strictEqual(JSON.stringify({ values: entry.wrapLayer.values, trace: lastCurveTraces.C_wrap }), wrapBefore);
assert.strictEqual(editUndoStack.length, 1);
assert.strictEqual(editUndoStack[0].label, 'shift_y');
undoLastCurveEdit();
assert.deepStrictEqual(entry.values, mainBefore);
assert.strictEqual(JSON.stringify(lastCurveTraces.C), mainTraceBefore);

entry.values = [10, 20, -999.25, -999.25, -999.25];
lastCurveTraces.C = [[10, 0, 0], [20, 1, 1]];
applyCurveYShiftForCurve(0);
assert.deepStrictEqual(entry.values, [-999.25, 10, 20, -999.25, -999.25]);

editUndoStack = [];
entry.values = [10, 20, 30, 40, 50];
lastCurveTraces.C = entry.values.map((value, idx) => [value, idx, idx]);
const blockedBefore = JSON.stringify({ values: entry.values, trace: lastCurveTraces.C });
applyCurveYShiftForCurve(0);
assert.strictEqual(JSON.stringify({ values: entry.values, trace: lastCurveTraces.C }), blockedBefore);
assert.strictEqual(editUndoStack.length, 0);

entry.values = [10, 20, -999.25, -999.25, -999.25];
lastCurveTraces.C = [[10, 0, 0], [20, 1, 1]];
pinned = new Set([1]);
const pinnedBefore = JSON.stringify({ values: entry.values, trace: lastCurveTraces.C });
applyCurveYShiftForCurve(0);
assert.strictEqual(JSON.stringify({ values: entry.values, trace: lastCurveTraces.C }), pinnedBefore);
assert.strictEqual(editUndoStack.length, 0);

pinned = new Set();
selectedLayer = 'wrap';
entry.wrapLayer.values = [110, -999.25, 120, -999.25, -999.25];
lastCurveTraces.C_wrap = [[110, 0, 0], [120, 2, 2]];
const mainBeforeWrap = JSON.stringify({ values: entry.values, trace: lastCurveTraces.C });
applyCurveYShiftForCurve(0);
assert.deepStrictEqual(entry.wrapLayer.values, [-999.25, 110, -999.25, 120, -999.25]);
assert.strictEqual(JSON.stringify({ values: entry.values, trace: lastCurveTraces.C }), mainBeforeWrap);
undoLastCurveEdit();
assert.deepStrictEqual(entry.wrapLayer.values, [110, -999.25, 120, -999.25, -999.25]);

lockedLayer = 'wrap';
const undoCount = editUndoStack.length;
const lockedBefore = JSON.stringify({ values: entry.wrapLayer.values, trace: lastCurveTraces.C_wrap });
applyCurveYShiftForCurve(0);
assert.strictEqual(JSON.stringify({ values: entry.wrapLayer.values, trace: lastCurveTraces.C_wrap }), lockedBefore);
assert.strictEqual(editUndoStack.length, undoCount);

lockedLayer = null;
selectedLayer = 'main';
entry.values = [-999.25, -999.25, -999.25, -999.25, 50];
lastCurveTraces.C = [[50, 4, 4]];
const boundaryBefore = JSON.stringify({ values: entry.values, trace: lastCurveTraces.C });
applyCurveYShiftForCurve(0);
assert.strictEqual(JSON.stringify({ values: entry.values, trace: lastCurveTraces.C }), boundaryBefore);
assert.strictEqual(editUndoStack.length, undoCount);
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((missing_source, context_source, y_source, undo_source, harness)),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_drag_preview_isolated_until_single_commit_and_cancel_is_mutation_free():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    resolver_source = _function_source(source, "resolveDepthMoveProposals", "applyCurveXShiftForCurve")
    lifecycle_source = _function_source(source, "cancelCurveDragAnimationFrame", "_computeLocalDepthSpacing")
    spacing_source = _function_source(source, "_computeLocalDepthSpacing", "_doDragUpdate")
    drag_source = _drag_source(source)
    undo_source = _function_source(source, "undoLastCurveEdit", "beginCurveEditInteraction")

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
const window = { currentDragLayer: null, paintTargetLayer: null, bezierCurveControls: {} };
let editMode = true;
let editUndoStack = [];
let lastDigitizedDepth = [0, 1, 2];
let lastDepthConfig = {};
let lastNullValue = -999.25;
_dragHasMoved = true;
_lastDragRenderMs = 0;
let renderCalls = 0;
let rebuildCalls = 0;
let wrapActivations = 0;
let nextFrameId = 1;
const cancelledFrames = [];
const document = { getElementById: () => ({ checked: false }) };
const performance = { now: () => 1000 };
function cancelAnimationFrame(id) { cancelledFrames.push(id); }
function getDepthConfigFromInputs() { return {}; }
function pixelToDepthFromConfig(y) { return y; }
function findNearestDepthIndex(depth) { return Math.max(0, Math.min(2, Math.round(depth))); }
function getActiveLayerValues(entry, layer) { return layer === 'wrap' ? entry.wrapLayer.values : entry.values; }
function getActiveLayerTrackConfig(entry, track, layer) {
    return layer === 'wrap' ? { ...track, scaleMin: 100, scaleMax: 200 } : track;
}
function isLayerLocked() { return false; }
function isDepthIndexPinned(curveId, idx) { return pinned.has(idx); }
function isPointLocked() { return false; }
function trackValueToPixelX(value) { return value; }
function pixelToTrackValue(x) { return x; }
function perturbIfEqual() {}
function snapToCurveColor(x, y) { return { x, y }; }
function activateTrackWrapForEdit() { wrapActivations++; }
function buildDisplayTracePointsFromDigitized(curveId, options) {
    return { traceKey: options.explicitLayer === 'wrap' ? 'C_wrap' : 'C', points: options.valuesSource.map((value, idx) => [value, idx, idx]) };
}
function renderCurveTraceOverlays() { renderCalls++; }
function rebuildCurveTraceFromDigitized() { rebuildCalls++; }
function getCurveLayerContext(curveId, layer) { return { values: getActiveLayerValues(entry, layer) }; }
function markCurveLayerTraceRevision() {}
function commitCurveLayerValues(curveId, layer, nextValues) {
    const values = getActiveLayerValues(entry, layer);
    values.splice(0, values.length, ...nextValues);
    rebuildCurveTraceFromDigitized(curveId, layer, { preserveBezierCache: true });
    return { changed: true, values };
}
function showStatus() {}
function applyBezierToSegmentValues() {}
async function applySnapToCurve() {}

const targetTrack = { leftX: 0, rightX: 100, scaleMin: 0, scaleMax: 100, wrapped: false };
const entry = { values: [10, 20, 30], wrapLayer: { values: [110, 120, 130] } };
let lastDigitizedCurves = { C: entry };
let lastCurveTraces = { C: [[10, 0, 0], [20, 1, 1], [30, 2, 2]], C_wrap: [[110, 0, 0], [120, 1, 1], [130, 2, 2]] };
let pinned = new Set();
function makeState(layer = 'main') {
    const values = getActiveLayerValues(entry, layer);
    const traceKey = layer === 'wrap' ? 'C_wrap' : 'C';
    const centerValue = values[1];
    return {
        interactionId: 1,
        curveId: 'C', curveKey: 'C', traceKey, dragLayer: layer,
        entry, targetTrack, centerIndex: 1, pointIndex: 1, depthIndex: 1,
        neighbors: [{ pointIndex: 1, depthIndex: 1, originalDepthIndex: 1, originalX: centerValue, originalY: 1, originalValue: centerValue, originalRawX: centerValue }],
        startX: centerValue, startY: 1, currentX: centerValue + 10, currentY: 1,
        originalValues: values.slice(), previewValues: values.slice(),
        originalTracePoints: lastCurveTraces[traceKey].map(point => point.slice()),
        previewTracePoints: lastCurveTraces[traceKey].map(point => point.slice()),
        committed: false, hasPreviewChange: false, animationFrameId: null,
    };
}

let editDragState = makeState('main');
const mainBefore = entry.values.slice();
const wrapBefore = JSON.stringify({ values: entry.wrapLayer.values, trace: lastCurveTraces.C_wrap });
const mainTraceBefore = JSON.stringify(lastCurveTraces.C);
_doDragUpdate();
assert.deepStrictEqual(entry.values, mainBefore);
assert.strictEqual(JSON.stringify(lastCurveTraces.C), mainTraceBefore);
assert.strictEqual(JSON.stringify({ values: entry.wrapLayer.values, trace: lastCurveTraces.C_wrap }), wrapBefore);
assert.strictEqual(editUndoStack.length, 0);
assert.strictEqual(editDragState.previewValues[1], 30);
const previewAtA = editDragState.previewValues.slice();

editDragState.currentX = 40;
_doDragUpdate();
editDragState.currentX = 30;
_doDragUpdate();
assert.deepStrictEqual(editDragState.previewValues, previewAtA);
assert.deepStrictEqual(entry.values, mainBefore);

const previewBeforeCommit = editDragState.previewValues.slice();
assert.strictEqual(commitCurveDrag(editDragState), true);
assert.deepStrictEqual(entry.values, previewBeforeCommit);
assert.strictEqual(editUndoStack.length, 1);
assert.strictEqual(rebuildCalls, 1);
assert.strictEqual(JSON.stringify({ values: entry.wrapLayer.values, trace: lastCurveTraces.C_wrap }), wrapBefore);
assert.deepStrictEqual(editUndoStack[0].values.map(item => item.oldValue), mainBefore);
undoLastCurveEdit();
assert.deepStrictEqual(entry.values, mainBefore);
assert.strictEqual(JSON.stringify(lastCurveTraces.C), mainTraceBefore);
assert.strictEqual(editUndoStack.length, 0);

editDragState = makeState('main');
editDragState.currentX = 35;
_doDragUpdate();
cancelCurveDrag('test-cancel');
assert.deepStrictEqual(entry.values, mainBefore);
assert.strictEqual(JSON.stringify(lastCurveTraces.C), mainTraceBefore);
assert.strictEqual(editUndoStack.length, 0);
assert.strictEqual(editDragState, null);

editDragState = makeState('wrap');
const mainBeforeWrap = JSON.stringify({ values: entry.values, trace: lastCurveTraces.C });
_doDragUpdate();
assert.deepStrictEqual(entry.wrapLayer.values, [110, 120, 130]);
assert.strictEqual(editDragState.previewValues[1], 130);
assert.strictEqual(JSON.stringify({ values: entry.values, trace: lastCurveTraces.C }), mainBeforeWrap);
cancelCurveDrag('wrap-cancel');

pinned = new Set([2]);
editDragState = makeState('main');
editDragState.currentY = 2;
_doDragUpdate();
assert.deepStrictEqual(editDragState.previewValues, mainBefore);
assert.strictEqual(editDragState.hasPreviewChange, false);
assert.strictEqual(commitCurveDrag(editDragState), false);
assert.strictEqual(editUndoStack.length, 0);
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((resolver_source, lifecycle_source, spacing_source, drag_source, undo_source, harness)).encode("utf-8"),
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def test_cancelled_drag_invalidates_late_animation_frame_and_noop_release():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    lifecycle_source = _function_source(source, "cancelCurveDragAnimationFrame", "_computeLocalDepthSpacing")

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
const window = { currentDragLayer: null, paintTargetLayer: null };
let editMode = true;
_dragHasMoved = false;
let editUndoStack = [];
let callbacks = new Map();
let nextId = 1;
let updates = 0;
let renders = 0;
let lastCurveTraces = { C: [[10, 0, 0]] };
const entry = { values: [10] };
function requestAnimationFrame(callback) { const id = nextId++; callbacks.set(id, callback); return id; }
function cancelAnimationFrame(id) { /* Intentionally retain callback to simulate a late browser delivery. */ }
function renderCurveTraceOverlays() { renders++; }
function getActiveLayerValues() { return entry.values; }
function buildDisplayTracePointsFromDigitized() { return { points: [[20, 0, 0]] }; }
function rebuildCurveTraceFromDigitized() {}
function applyBezierToSegmentValues() {}
async function applySnapToCurve() {}
function _doDragUpdate() { updates++; }
const document = { getElementById: () => ({ checked: false }) };

let editDragState = {
    interactionId: 77, curveId: 'C', curveKey: 'C', traceKey: 'C', dragLayer: 'main', entry,
    startX: 10, startY: 0, currentX: 10, currentY: 0,
    originalValues: [10], previewValues: [10], originalTracePoints: [[10, 0, 0]],
    committed: false, hasPreviewChange: false, animationFrameId: null,
};
handleCurveEditDragMove(20, 0);
const queued = callbacks.get(editDragState.animationFrameId);
assert.strictEqual(typeof queued, 'function');
cancelCurveDrag('pointercancel');
queued();
assert.strictEqual(updates, 0);
assert.deepStrictEqual(entry.values, [10]);
assert.strictEqual(editUndoStack.length, 0);
assert.strictEqual(editDragState, null);

editDragState = {
    interactionId: 78, curveId: 'C', curveKey: 'C', traceKey: 'C', dragLayer: 'main', entry,
    startX: 10, startY: 0, currentX: 10, currentY: 0,
    originalValues: [10], previewValues: [10], originalTracePoints: [[10, 0, 0]],
    committed: false, hasPreviewChange: false, animationFrameId: null,
};
finishCurveEditDrag();
assert.deepStrictEqual(entry.values, [10]);
assert.strictEqual(editUndoStack.length, 0);
assert.strictEqual(editDragState, null);
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((lifecycle_source, harness)),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_snap_is_frozen_into_preview_and_commit_copies_preview_exactly():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    begin_source = _function_source(source, "beginCurveEditInteraction", "cancelCurveDragAnimationFrame")
    resolver_source = _function_source(source, "resolveDepthMoveProposals", "applyCurveXShiftForCurve")
    lifecycle_source = _function_source(source, "cancelCurveDragAnimationFrame", "_computeLocalDepthSpacing")
    spacing_source = _function_source(source, "_computeLocalDepthSpacing", "_doDragUpdate")
    drag_source = _drag_source(source)
    commit_source = _function_source(source, "commitCurveDrag", "finishCurveEditDrag")

    assert "snapEnabled" in begin_source
    assert "snapTrackMode" in begin_source
    assert "snapOptions" in begin_source
    assert "magnetEditToggle" not in drag_source
    assert "snapToCurveToggle" not in drag_source
    assert drag_source.index("snapToCurveColor(") < drag_source.index("if (state.isBezierControl)")
    assert "snapToCurveColor" not in commit_source

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
const window = { currentDragLayer: null, paintTargetLayer: null, bezierCurveControls: {} };
let editMode = true;
let editUndoStack = [];
let lastDigitizedDepth = [0, 1, 2];
let lastDepthConfig = {};
let lastNullValue = -999.25;
_dragHasMoved = true;
_lastDragRenderMs = 0;
let snapCalls = 0;
let invalidSnap = false;
let rebuildCalls = 0;
let pinned = new Set();
const document = { getElementById: () => ({ checked: false }) };
const performance = { now: () => 1000 };
function cancelAnimationFrame() {}
function getDepthConfigFromInputs() { return {}; }
function pixelToDepthFromConfig(y) { return y; }
function findNearestDepthIndex(depth) { return Math.max(0, Math.min(2, Math.round(depth))); }
function getActiveLayerValues(entry, layer) { return layer === 'wrap' ? entry.wrapLayer.values : entry.values; }
function getActiveLayerTrackConfig(entry, track) { return track; }
function isLayerLocked() { return false; }
function isDepthIndexPinned(curveId, depthIndex) { return pinned.has(depthIndex); }
function isPointLocked() { return false; }
function isMissingDigitizedValue(value) { return value == null || !Number.isFinite(value) || value === lastNullValue; }
function trackValueToPixelX(value) { return value; }
function pixelToTrackValue(x) { return x; }
function perturbIfEqual() {}
function snapToCurveColor(x, y) { snapCalls++; return invalidSnap ? { x: NaN, y: Infinity } : { x: x + 5, y }; }
function activateTrackWrapForEdit() {}
function buildDisplayTracePointsFromDigitized(curveId, options) { return { points: options.valuesSource.map((value, idx) => [value, idx, idx]) }; }
function renderCurveTraceOverlays() {}
function rebuildCurveTraceFromDigitized() { rebuildCalls++; }
function commitCurveLayerValues(curveId, layer, nextValues) {
    const values = getActiveLayerValues(entry, layer);
    values.splice(0, values.length, ...nextValues);
    rebuildCurveTraceFromDigitized(curveId, layer, { preserveBezierCache: true });
    return { changed: true, values };
}
function applyBezierToSegmentValues() {}
async function applySnapToCurve() { throw new Error('release-only snap must not run'); }

const track = { leftX: 0, rightX: 100, scaleMin: 0, scaleMax: 100, wrapped: false };
const entry = { values: [10, 20, 30], wrapLayer: { values: [110, 120, 130] } };
let lastDigitizedCurves = { C: entry };
let lastCurveTraces = { C: [[10, 0, 0], [20, 1, 1], [30, 2, 2]], C_wrap: [[110, 0, 0], [120, 1, 1], [130, 2, 2]] };
function makeState(snapEnabled, dragLayer = 'main') {
    const isWrap = dragLayer === 'wrap';
    const originalValues = isWrap ? [110, 120, 130] : [10, 20, 30];
    const originalX = originalValues[1];
    return {
        interactionId: 1, curveId: 'C', curveKey: 'C', traceKey: isWrap ? 'C_wrap' : 'C', dragLayer, entry,
        targetTrack: track, centerIndex: 1, pointIndex: 1, depthIndex: 1,
        neighbors: [{ pointIndex: 1, depthIndex: 1, originalDepthIndex: 1, originalX, originalY: 1, originalValue: originalX, originalRawX: originalX }],
        startX: originalX, startY: 1, currentX: 30, currentY: 1,
        originalValues, previewValues: originalValues.slice(),
        originalTracePoints: lastCurveTraces[isWrap ? 'C_wrap' : 'C'].map(point => point.slice()),
        committed: false, hasPreviewChange: false, animationFrameId: null,
        snapEnabled, snapTrackMode: 'green', snapOptions: { mode: 'green' },
    };
}

let editDragState = makeState(true);
const authoritativeBefore = entry.values.slice();
_doDragUpdate();
assert.deepStrictEqual(entry.values, authoritativeBefore);
assert.strictEqual(editDragState.lastRawPointer.x, 30);
assert.strictEqual(editDragState.lastEffectivePointer.x, 35);
assert.deepStrictEqual(editDragState.lastSnapResult, { enabled: true, available: true, x: 35, y: 1, changed: true });
assert.strictEqual(editDragState.previewValues[1], 35);
const finalPreview = editDragState.previewValues.slice();
assert.strictEqual(snapCalls, 1);
editDragState.currentX = 40;
_doDragUpdate();
assert.strictEqual(editDragState.previewValues[1], 45);
editDragState.currentX = 30;
_doDragUpdate();
assert.deepStrictEqual(editDragState.previewValues, finalPreview);
assert.deepStrictEqual(entry.values, authoritativeBefore);
assert.strictEqual(snapCalls, 3);
assert.strictEqual(commitCurveDrag(editDragState), true);
assert.strictEqual(snapCalls, 3);
assert.deepStrictEqual(entry.values, finalPreview);
assert.strictEqual(editDragState.previewMatchesCommit, true);
assert.strictEqual(editUndoStack.length, 1);

entry.values.splice(0, entry.values.length, 10, 20, 30);
editUndoStack = [];
snapCalls = 0;
editDragState = makeState(false);
_doDragUpdate();
assert.strictEqual(snapCalls, 0);
assert.strictEqual(editDragState.previewValues[1], 30);
assert.deepStrictEqual(editDragState.lastSnapResult, { enabled: false, available: false, x: 30, y: 1, changed: false });

invalidSnap = true;
snapCalls = 0;
editDragState = makeState(true);
_doDragUpdate();
assert.strictEqual(snapCalls, 1);
assert.strictEqual(editDragState.lastEffectivePointer.x, 30);
assert.strictEqual(editDragState.lastSnapResult.available, false);
assert.strictEqual(editDragState.previewValues[1], 30);

cancelCurveDrag('snap-cancel');
assert.deepStrictEqual(entry.values, [10, 20, 30]);
assert.strictEqual(editUndoStack.length, 0);
assert.strictEqual(editDragState, null);

invalidSnap = false;
pinned = new Set([2]);
editDragState = makeState(true);
editDragState.currentY = 2;
_doDragUpdate();
assert.deepStrictEqual(editDragState.previewValues, [10, 20, 30]);
cancelCurveDrag('pinned-snap-cancel');

pinned = new Set();
const mainBeforeWrapPreview = entry.values.slice();
editDragState = makeState(true, 'wrap');
_doDragUpdate();
assert.strictEqual(editDragState.previewValues[1], 35);
assert.deepStrictEqual(entry.values, mainBeforeWrapPreview);
assert.deepStrictEqual(entry.wrapLayer.values, [110, 120, 130]);
cancelCurveDrag('wrap-snap-cancel');
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((resolver_source, lifecycle_source, spacing_source, drag_source, harness)).encode("utf-8"),
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def test_release_coordinates_run_one_final_snapped_preview_before_commit():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    lifecycle_source = _function_source(source, "cancelCurveDragAnimationFrame", "_computeLocalDepthSpacing")

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
const window = { currentDragLayer: null, paintTargetLayer: null };
_dragHasMoved = true;
let editUndoStack = [];
let previewCalls = [];
let lastCurveTraces = { C: [[10, 0, 0]] };
const entry = { values: [10] };
const document = { getElementById: () => ({ checked: false }) };
function cancelAnimationFrame() {}
function getActiveLayerValues() { return entry.values; }
function renderCurveTraceOverlays() {}
function renderCurveDragPreview() {}
function rebuildCurveTraceFromDigitized() {}
function valuesExactlyEqual(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
function commitCurveLayerValues(curveId, layer, nextValues) {
    entry.values.splice(0, entry.values.length, ...nextValues);
    return { changed: true, values: entry.values };
}
function activateTrackWrapForEdit() {}
function _doDragUpdate() {
    previewCalls.push([editDragState.currentX, editDragState.currentY]);
    editDragState.previewValues = [editDragState.currentX + 5];
    editDragState.lastRawPointer = { x: editDragState.currentX, y: editDragState.currentY };
    editDragState.lastEffectivePointer = { x: editDragState.currentX + 5, y: editDragState.currentY };
    editDragState.hasPreviewChange = true;
}

let editDragState = {
    interactionId: 9, curveId: 'C', curveKey: 'C', traceKey: 'C', dragLayer: 'main', entry,
    targetTrack: {}, currentX: 20, currentY: 1,
    originalValues: [10], originalTracePoints: [[10, 0, 0]], previewValues: [25],
    hasPreviewChange: true, committed: false, animationFrameId: 44,
};

(async () => {
    const committed = await finishCurveEditDrag(40, 2);
    assert.strictEqual(committed, true);
    assert.deepStrictEqual(previewCalls, [[40, 2]]);
    assert.deepStrictEqual(entry.values, [45]);
    assert.strictEqual(editUndoStack.length, 1);
    assert.strictEqual(editDragState, null);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((lifecycle_source, harness)),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_bezier_cache_is_layer_keyed_deep_cloned_and_missing_wrap_does_not_fallback():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    helper_source = _function_source(source, "getBezierCacheKey", "getBezierSegments")
    build_source = _function_source(source, "getBezierSegments", "applyBezierToSegmentValues")

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
const window = { bezierCurveControls: {} };
function normalizeCurveKey(value) { return String(value || '').trim().toUpperCase(); }
function getActiveCurveTraceKey(curveId) { return String(curveId || '').replace(/_wrap$/i, ''); }
function isDepthIndexPinned() { return false; }
const traces = {
    GR: [[10, 0, 0], [20, 1, 1], [30, 2, 2], [40, 3, 3]],
    GR_wrap: [[110, 0, 0], [130, 1, 1], [150, 2, 2], [170, 3, 3]],
    LOCKED: [[10, 0, 0], [20, 1, 1], [30, 2, 2], [40, 3, 3]],
};
function getActiveCurveTracePoints(curveId, layer) {
    const key = String(curveId).replace(/_wrap$/i, '').toUpperCase() + (layer === 'wrap' ? '_wrap' : '');
    return { traceKey: key, points: traces[key] || null };
}

const main = syncBezierControlsForCurve('gr', 'main');
const wrap = syncBezierControlsForCurve('gr', 'wrap');
assert.strictEqual(getBezierCacheKey('GR', 'main'), 'GR::main');
assert.strictEqual(getBezierCacheKey('GR_wrap', 'wrap'), 'GR::wrap');
assert.ok(Array.isArray(main) && Array.isArray(wrap));
assert.notStrictEqual(main, wrap);
assert.notStrictEqual(main[0], wrap[0]);
assert.notStrictEqual(main[0].p1, wrap[0].p1);
assert.notDeepStrictEqual(main[0].p1, wrap[0].p1);

const wrapBefore = cloneBezierSegments(wrap);
main[0].p1[0] += 999;
assert.deepStrictEqual(wrap, wrapBefore);
assert.strictEqual(syncBezierControlsForCurve('GR', 'main'), main);
assert.strictEqual(syncBezierControlsForCurve('GR', 'wrap'), wrap);

const replacement = cloneBezierSegments(main);
setBezierSegments('GR', 'main', replacement);
replacement[0].p2[0] += 500;
assert.notStrictEqual(getCachedBezierSegments('GR', 'main')[0].p2[0], replacement[0].p2[0]);

assert.strictEqual(deleteBezierSegments('GR', 'main'), true);
assert.strictEqual(getCachedBezierSegments('GR', 'main'), null);
assert.deepStrictEqual(getCachedBezierSegments('GR', 'wrap'), wrapBefore);
assert.strictEqual(deleteAllBezierSegmentsForCurve('GR'), true);
assert.strictEqual(getCachedBezierSegments('GR', 'wrap'), null);

const beforeMissing = JSON.stringify(window.bezierCurveControls);
assert.strictEqual(syncBezierControlsForCurve('MISSING', 'wrap'), null);
assert.strictEqual(JSON.stringify(window.bezierCurveControls), beforeMissing);
const lockedPreview = syncBezierControlsForCurve('LOCKED', 'main', { persist: false });
assert.ok(Array.isArray(lockedPreview));
assert.strictEqual(getCachedBezierSegments('LOCKED', 'main'), null);
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((helper_source, build_source, harness)).encode("utf-8"),
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def test_bezier_preview_is_transactional_and_uses_frozen_layer_segments():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    helper_source = _function_source(source, "getBezierCacheKey", "getBezierSegments")
    apply_source = _function_source(source, "applyBezierToSegmentValues", "renderBezierControls")
    drag_source = _drag_source(source)

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
const window = { bezierCurveControls: {} };
let editMode = true;
let _dragHasMoved = true;
let _lastDragRenderMs = 0;
let lastDigitizedCurves = null;
let lastDigitizedDepth = [0, 1, 2];
let lastDepthConfig = {};
let lastNullValue = -999.25;
const performance = { now: () => 1000 };
const document = { getElementById: () => ({ checked: false }) };
function normalizeCurveKey(value) { return String(value || '').trim().toUpperCase(); }
function getActiveCurveTraceKey(curveId) { return String(curveId || '').toUpperCase(); }
function getActiveLayerValues(entry, layer) { return layer === 'wrap' ? entry.wrapLayer.values : entry.values; }
function getActiveLayerTrackConfig(entry, track, layer) { return layer === 'wrap' ? { ...track, scaleMin: 100, scaleMax: 200 } : track; }
function getActiveCurveTracePoints(curveId, layer) { return { points: layer === 'wrap' ? traces.C_wrap : traces.C }; }
function findTrackByCurveId() { return track; }
function isDepthIndexPinned() { return false; }
function isPointLocked() { return false; }
function pixelToTrackValue(x) { return x; }
function perturbIfEqual() {}
function snapToCurveColor(x, y) { return { x, y }; }
function valuesExactlyEqual(a, b) { return a.length === b.length && a.every((v, i) => Object.is(v, b[i])); }
function renderCurveDragPreview() { renders++; }

const track = { leftX: 0, rightX: 200, scaleMin: 0, scaleMax: 200 };
const entry = { values: [10, 20, 30], wrapLayer: { values: [110, 120, 130] } };
const traces = { C: [[10, 0, 0], [20, 1, 1], [30, 2, 2]], C_wrap: [[110, 0, 0], [120, 1, 1], [130, 2, 2]] };
const mainSegments = [{ startIdx: 0, endIdx: 2, p0: [10, 0], p1: [15, 0.66], p2: [25, 1.33], p3: [30, 2] }];
const wrapSegments = [{ startIdx: 0, endIdx: 2, p0: [110, 0], p1: [115, 0.66], p2: [125, 1.33], p3: [130, 2] }];
setBezierSegments('C', 'main', mainSegments);
setBezierSegments('C', 'wrap', wrapSegments);
const mainCacheBefore = cloneBezierSegments(getCachedBezierSegments('C', 'main'));
const wrapCacheBefore = cloneBezierSegments(getCachedBezierSegments('C', 'wrap'));
const mainValuesBefore = entry.values.slice();
const wrapValuesBefore = entry.wrapLayer.values.slice();
let renders = 0;

let editDragState = {
    curveId: 'C', curveKey: 'C', traceKey: 'C_wrap', dragLayer: 'wrap', isBezierControl: true,
    entry, targetTrack: track, segmentIndex: 0, controlIndex: 1,
    startX: 115, startY: 0.66, currentX: 135, currentY: 0.66,
    originalControlX: 115, originalControlY: 0.66,
    originalValues: wrapValuesBefore.slice(), originalTracePoints: traces.C_wrap.map(p => p.slice()),
    originalBezierSegments: cloneBezierSegments(wrapSegments),
    previewBezierSegments: cloneBezierSegments(wrapSegments), previewValues: wrapValuesBefore.slice(),
    snapEnabled: false,
};
_doDragUpdate();
assert.deepStrictEqual(entry.values, mainValuesBefore);
assert.deepStrictEqual(entry.wrapLayer.values, wrapValuesBefore);
assert.deepStrictEqual(getCachedBezierSegments('C', 'main'), mainCacheBefore);
assert.deepStrictEqual(getCachedBezierSegments('C', 'wrap'), wrapCacheBefore);
assert.strictEqual(editDragState.previewBezierSegments[0].p1[0], 135);
assert.notDeepStrictEqual(editDragState.previewValues, wrapValuesBefore);
assert.strictEqual(editDragState.hasPreviewChange, true);
assert.strictEqual(renders, 1);
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((helper_source, apply_source, drag_source, harness)).encode("utf-8"),
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def test_bezier_commit_cancel_and_undo_are_layer_safe():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    helper_source = _function_source(source, "getBezierCacheKey", "getBezierSegments")
    lifecycle_source = _function_source(source, "cancelCurveDragAnimationFrame", "_computeLocalDepthSpacing")
    undo_source = _function_source(source, "undoLastCurveEdit", "beginCurveEditInteraction")

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
const window = { bezierCurveControls: {}, currentDragLayer: null, paintTargetLayer: null };
function normalizeCurveKey(value) { return String(value || '').trim().toUpperCase(); }
function getActiveCurveTraceKey(curveId) { return String(curveId || '').replace(/_wrap$/i, ''); }
function getActiveLayerValues(entry, layer) { return layer === 'wrap' ? entry.wrapLayer.values : entry.values; }
function cancelAnimationFrame() {}
function renderCurveTraceOverlays() {}
function rebuildCurveTraceFromDigitized(curveId, layer, options) { rebuilds.push([curveId, layer, options]); }
function getCurveLayerContext(curveId, layer) { return { values: getActiveLayerValues(entry, layer) }; }
function commitCurveLayerValues(curveId, layer, nextValues, options = {}) {
    const values = getActiveLayerValues(entry, layer);
    values.splice(0, values.length, ...nextValues);
    if (options.rebuildTrace !== false) rebuildCurveTraceFromDigitized(curveId, layer, { preserveBezierCache: true });
    return { changed: true, values };
}
function markCurveLayerTraceRevision() {}
function showStatus() {}

let editUndoStack = [];
let rebuilds = [];
const entry = { values: [10, 20, 30], wrapLayer: { values: [110, 120, 130] } };
let lastDigitizedCurves = { C: entry };
let lastCurveTraces = { C: [[10, 0, 0], [20, 1, 1], [30, 2, 2]], C_wrap: [[110, 0, 0], [120, 1, 1], [130, 2, 2]] };
const mainOriginal = [{ startIdx: 0, endIdx: 2, p0: [10, 0], p1: [15, 1], p2: [25, 1], p3: [30, 2] }];
const mainPreview = [{ startIdx: 0, endIdx: 2, p0: [10, 0], p1: [22, 1], p2: [25, 1], p3: [30, 2] }];
const wrapOriginal = [{ startIdx: 0, endIdx: 2, p0: [110, 0], p1: [115, 1], p2: [125, 1], p3: [130, 2] }];
setBezierSegments('C', 'main', mainOriginal);
setBezierSegments('C', 'wrap', wrapOriginal);
const wrapEverythingBefore = JSON.stringify({ values: entry.wrapLayer.values, trace: lastCurveTraces.C_wrap, cache: getCachedBezierSegments('C', 'wrap') });

let editDragState = {
    curveId: 'C', curveKey: 'C', traceKey: 'C', dragLayer: 'main', isBezierControl: true,
    bezierCacheKey: 'C::main', entry, segmentIndex: 0, controlIndex: 1, targetTrack: {},
    originalControlX: 15, originalControlY: 1,
    originalValues: [10, 20, 30], previewValues: [10, 22, 30],
    originalTracePoints: lastCurveTraces.C.map(p => p.slice()),
    originalBezierSegments: cloneBezierSegments(mainOriginal),
    previewBezierSegments: cloneBezierSegments(mainPreview),
    bezierUndoValues: [10, 20, 30].map((oldValue, idx) => ({ idx, oldValue })),
    hasPreviewChange: true, committed: false, animationFrameId: null,
};
assert.strictEqual(commitCurveDrag(editDragState), true);
assert.deepStrictEqual(entry.values, [10, 22, 30]);
assert.deepStrictEqual(getCachedBezierSegments('C', 'main'), mainPreview);
assert.strictEqual(JSON.stringify({ values: entry.wrapLayer.values, trace: lastCurveTraces.C_wrap, cache: getCachedBezierSegments('C', 'wrap') }), wrapEverythingBefore);
assert.strictEqual(editUndoStack.length, 1);
assert.strictEqual(editUndoStack[0].type, 'bezier_transform');
assert.strictEqual(editUndoStack[0].bezierCacheKey, 'C::main');
assert.deepStrictEqual(rebuilds, [['C', 'main', { preserveBezierCache: true }]]);

undoLastCurveEdit();
assert.deepStrictEqual(entry.values, [10, 20, 30]);
assert.deepStrictEqual(lastCurveTraces.C, [[10, 0, 0], [20, 1, 1], [30, 2, 2]]);
assert.deepStrictEqual(getCachedBezierSegments('C', 'main'), mainOriginal);
assert.strictEqual(JSON.stringify({ values: entry.wrapLayer.values, trace: lastCurveTraces.C_wrap, cache: getCachedBezierSegments('C', 'wrap') }), wrapEverythingBefore);
assert.strictEqual(editUndoStack.length, 0);

const cacheBeforeCancel = JSON.stringify(window.bezierCurveControls);
editDragState = {
    curveId: 'C', curveKey: 'C', traceKey: 'C_wrap', dragLayer: 'wrap', isBezierControl: true,
    entry, originalTracePoints: lastCurveTraces.C_wrap.map(p => p.slice()),
    originalValues: entry.wrapLayer.values.slice(), previewValues: [110, 140, 130],
    originalBezierSegments: cloneBezierSegments(wrapOriginal),
    previewBezierSegments: [{ ...wrapOriginal[0], p1: [145, 1] }],
    hasPreviewChange: true, committed: false, animationFrameId: null,
};
cancelCurveDrag('test-cancel');
assert.strictEqual(JSON.stringify(window.bezierCurveControls), cacheBeforeCancel);
assert.deepStrictEqual(entry.wrapLayer.values, [110, 120, 130]);
assert.strictEqual(editUndoStack.length, 0);
assert.strictEqual(editDragState, null);
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((helper_source, lifecycle_source, undo_source, harness)).encode("utf-8"),
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def test_curve_layer_revisions_replacement_aliasing_and_stale_drag_protection():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    revision_source = _function_source(source, "getCurveLayerContext", "getActiveLayerValues")
    commit_start = source.index("function commitCurveDrag")
    commit_end = source.index("async function finishCurveEditDrag", commit_start)
    commit_source = source[commit_start:commit_end]

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
const window = { bezierCurveControls: {}, currentDragLayer: null, paintTargetLayer: null };
const curveLayerRevisions = new Map();
const curveLayerTraceRevisions = new Map();
let committedDatasetGeneration = 0;
let workspaceHasUnsavedCommittedChanges = false;
let curveWrapMarkers = { C: [{ depthIndex: 1 }] };
let editUndoStack = [{ type: 'old_dataset_action' }];
let lastCurveTraces = { C: [['old-main']], C_wrap: [['old-wrap']] };
let lastDigitizedCurves = {
    C: { values: [10, 20, 30], wrapLayer: { values: [110, 120, 130], left_value: 100, right_value: 200 } },
};
let editCurveId = 'C';
let editDragState = null;
let drawStrokeActive = false, drawStrokeLayer = null, drawStrokePoints = [], drawStrokeSourceRevision = null;
let paintStrokeActive = false, paintStrokeLayer = null, paintStrokeSourceRevision = null;
let paintUndoValuesBefore = [], paintUndoPtsBefore = [];
let smoothStrokeActive = false, smoothStrokeLayer = null, smoothStrokeSourceRevision = null;
let smoothUndoValuesBefore = [];
let eraserStart = null, eraserStrokeLayer = null, eraserStrokeSourceRevision = null, eraserBox = null;
let clearedBezier = [];
let rebuilds = [];
let cancellations = [];
function normalizeCurveKey(value) { return String(value || '').trim().toUpperCase(); }
function resolveTraceKeyForCurveId(curveId) { return normalizeCurveKey(curveId).replace(/_WRAP$/i, ''); }
function findDigitizedCurveEntry(curveId) {
    const normalized = normalizeCurveKey(curveId).replace(/_WRAP$/i, '');
    const key = Object.keys(lastDigitizedCurves || {}).find(candidate => normalizeCurveKey(candidate) === normalized);
    return key ? { key, entry: lastDigitizedCurves[key] } : null;
}
function valuesExactlyEqual(a, b) { return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => Object.is(v, b[i])); }
function deleteBezierSegments(curveId, layer) { clearedBezier.push(`${normalizeCurveKey(curveId)}::${layer}`); return true; }
function clearAllBezierSegments() { window.bezierCurveControls = {}; }
function clearEditInteractionLayerState() {}
function rebuildCurveTraceFromDigitized(curveId, layer) {
    rebuilds.push(`${normalizeCurveKey(curveId)}::${layer}`);
    const context = getCurveLayerContext(curveId, layer);
    if (context) lastCurveTraces[context.traceKey] = context.values.map((value, idx) => [value, idx, idx]);
}
function cancelCurveDrag(reason, state) { cancellations.push(reason); if (editDragState === state) editDragState = null; return true; }

assert.strictEqual(getCurveLayerRevision('C', 'main'), 0);
assert.strictEqual(getCurveLayerRevision('C', 'wrap'), 0);
const wrapSnapshot = JSON.stringify({ values: lastDigitizedCurves.C.wrapLayer.values, trace: lastCurveTraces.C_wrap });
const replacement = [40, 50, 60];
const mainResult = replaceCurveLayerValues('C', 'main', replacement, { reason: 'external_main' });
assert.strictEqual(mainResult.changed, true);
assert.strictEqual(mainResult.revision, 1);
assert.strictEqual(getCurveLayerRevision('C', 'main'), 1);
assert.strictEqual(getCurveLayerRevision('C', 'wrap'), 0);
assert.strictEqual(JSON.stringify({ values: lastDigitizedCurves.C.wrapLayer.values, trace: lastCurveTraces.C_wrap }), wrapSnapshot);
replacement[0] = 999;
assert.strictEqual(lastDigitizedCurves.C.values[0], 40);
assert.strictEqual(getCurveLayerRevision('C', 'main'), 1);
assert.ok(clearedBezier.includes('C::main'));
assert.ok(!clearedBezier.includes('C::wrap'));
assert.deepStrictEqual(rebuilds, ['C::main']);

const noOp = commitCurveLayerValues('C', 'main', [40, 50, 60]);
assert.strictEqual(noOp.changed, false);
assert.strictEqual(getCurveLayerRevision('C', 'main'), 1);
assert.deepStrictEqual(rebuilds, ['C::main']);

const mainSnapshot = JSON.stringify({ values: lastDigitizedCurves.C.values, trace: lastCurveTraces.C });
const wrapReplacement = [140, 150, 160];
const wrapResult = replaceCurveLayerValues('C', 'wrap', wrapReplacement, { reason: 'external_wrap' });
assert.strictEqual(wrapResult.revision, 1);
assert.strictEqual(getCurveLayerRevision('C', 'main'), 1);
assert.strictEqual(getCurveLayerRevision('C', 'wrap'), 1);
assert.strictEqual(JSON.stringify({ values: lastDigitizedCurves.C.values, trace: lastCurveTraces.C }), mainSnapshot);
wrapReplacement[0] = 999;
assert.strictEqual(lastDigitizedCurves.C.wrapLayer.values[0], 140);

const staleState = {
    curveId: 'C', curveKey: 'C', dragLayer: 'main', sourceRevision: 1,
    entry: lastDigitizedCurves.C, committed: false, previewValues: [70, 80, 90], hasPreviewChange: true,
};
editDragState = staleState;
replaceCurveLayerValues('C', 'main', [41, 51, 61], { reason: 'external_again' });
assert.strictEqual(editDragState, null);
assert.ok(cancellations.includes('external_again'));
const undoCount = editUndoStack.length;
assert.strictEqual(commitCurveDrag(staleState), false);
assert.deepStrictEqual(lastDigitizedCurves.C.values, [41, 51, 61]);
assert.strictEqual(editUndoStack.length, undoCount);
assert.ok(cancellations.includes('source_data_changed'));
const undoLikeRestore = commitCurveLayerValues('C', 'main', [40, 50, 60], { reason: 'undo_test' });
assert.strictEqual(undoLikeRestore.revision, 3);
assert.deepStrictEqual(lastDigitizedCurves.C.values, [40, 50, 60]);
assert.strictEqual(getCurveLayerRevision('C', 'main'), 3);

const staleBezierState = {
    curveId: 'C', curveKey: 'C', dragLayer: 'main', mode: 'bezier', sourceRevision: 3,
    entry: lastDigitizedCurves.C, committed: false, previewValues: [45, 55, 65], hasPreviewChange: true,
};
editDragState = staleBezierState;
replaceCurveLayerValues('C', 'main', [42, 52, 62], { reason: 'external_during_bezier' });
assert.strictEqual(editDragState, null);
const bezierUndoCount = editUndoStack.length;
assert.strictEqual(commitCurveDrag(staleBezierState), false);
assert.deepStrictEqual(lastDigitizedCurves.C.values, [42, 52, 62]);
assert.strictEqual(editUndoStack.length, bezierUndoCount);
assert.ok(cancellations.includes('source_data_changed'));

const datasetInput = { D: { values: [1, 2], wrapLayer: { values: [101, 102], color: '#0ff' } } };
replaceAllDigitizedCurves(datasetInput, { reason: 'project_load', clearTraces: true });
assert.strictEqual(getCurveLayerRevision('D', 'main'), 1);
assert.strictEqual(getCurveLayerRevision('D', 'wrap'), 1);
assert.strictEqual(editUndoStack.length, 0);
datasetInput.D.values[0] = 999;
datasetInput.D.wrapLayer.values[0] = 999;
assert.deepStrictEqual(lastDigitizedCurves.D.values, [1, 2]);
assert.deepStrictEqual(lastDigitizedCurves.D.wrapLayer.values, [101, 102]);
const dMainRevision = getCurveLayerRevision('D', 'main');
assert.strictEqual(removeCurveWrapLayer('D', { reason: 'test_remove_wrap' }).revision, 2);
assert.strictEqual(lastDigitizedCurves.D.wrapLayer, undefined);
assert.strictEqual(getCurveLayerRevision('D', 'main'), dMainRevision);
const createdWrapValues = [201, 202];
assert.strictEqual(createOrReplaceCurveWrapLayer('D', createdWrapValues, { color: '#0ff' }, { reason: 'test_create_wrap' }).revision, 3);
createdWrapValues[0] = 999;
assert.deepStrictEqual(lastDigitizedCurves.D.wrapLayer.values, [201, 202]);
assert.strictEqual(getCurveLayerRevision('D', 'main'), dMainRevision);
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((revision_source, commit_source, harness)).encode("utf-8"),
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def test_bezier_cache_revision_mismatch_is_never_returned():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    revision_source = _function_source(source, "getCurveLayerContext", "getActiveLayerValues")
    bezier_helpers = _function_source(source, "getBezierCacheKey", "getBezierSegments")

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
const window = { bezierCurveControls: {} };
const curveLayerRevisions = new Map();
const curveLayerTraceRevisions = new Map();
let curveWrapMarkers = {};
let editUndoStack = [];
let lastCurveTraces = { C: [] };
let lastDigitizedCurves = { C: { values: [10, 20, 30] } };
let editDragState = null;
let drawStrokeActive = false, paintStrokeActive = false, smoothStrokeActive = false, eraserStart = null;
function normalizeCurveKey(value) { return String(value || '').trim().toUpperCase(); }
function resolveTraceKeyForCurveId(curveId) { return normalizeCurveKey(curveId); }
function findDigitizedCurveEntry(curveId) { return { key: 'C', entry: lastDigitizedCurves.C }; }
function valuesExactlyEqual(a, b) { return a.length === b.length && a.every((v, i) => Object.is(v, b[i])); }
function rebuildCurveTraceFromDigitized() {}
function cancelCurveDrag() {}

bumpCurveLayerRevision('C', 'main', 'initialize');
const segments = [{ startIdx: 0, endIdx: 2, p0: [10, 0], p1: [15, 1], p2: [25, 1], p3: [30, 2] }];
setBezierSegments('C', 'main', segments);
assert.deepStrictEqual(getCachedBezierSegments('C', 'main'), segments);
const cacheKey = getBezierCacheKey('C', 'main');
assert.strictEqual(window.bezierCurveControls[cacheKey].sourceRevision, 1);
bumpCurveLayerRevision('C', 'main', 'untracked_test_advance');
assert.strictEqual(getCachedBezierSegments('C', 'main'), null);
assert.strictEqual(window.bezierCurveControls[cacheKey], undefined);
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((revision_source, bezier_helpers, harness)).encode("utf-8"),
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def test_authoritative_array_replacement_is_confined_to_canonical_helpers():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    replace_region = _function_source(source, "replaceCurveLayerValues", "commitCurveLayerValues")
    dataset_region = _function_source(source, "replaceAllDigitizedCurves", "invalidateAllLayersForCurve")

    direct_dataset_assignments = [
        line for line in source.splitlines()
        if re.search(r"\blastDigitizedCurves\s*=(?!=)", line)
        and "let lastDigitizedCurves" not in line
    ]
    assert direct_dataset_assignments == [
        line for line in dataset_region.splitlines() if re.search(r"\blastDigitizedCurves\s*=(?!=)", line)
    ]

    authoritative_value_assignments = [
        line for line in source.splitlines()
        if re.search(r"(?:context|found)\.entry(?:\.wrapLayer)?\.values\s*=", line)
    ]
    assert authoritative_value_assignments
    assert authoritative_value_assignments == [
        line for line in replace_region.splitlines()
        if re.search(r"(?:context|found)\.entry(?:\.wrapLayer)?\.values\s*=", line)
    ]


def test_committed_snapshot_isolated_preview_free_merge_safe_and_repeatable():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    snapshot_source = _function_source(source, "clonePersistenceValue", "getActiveLayerValues")
    las_source = _function_source(source, "buildLasFromDigitized", "renderDigitizationSummary")
    canonical_source = _function_source(source, "getCanonicalHeaderMetadata", "mergeHeaderMetadataValues")

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the workspace JavaScript regression harness")
    harness = r"""
const assert = require('assert');
const HEADER_METADATA_KEY_GROUPS = [
    ['well'], ['comp', 'company'], ['api'], ['date'], ['fld', 'field'],
    ['loc', 'location'], ['county'], ['state'], ['prov', 'province'],
    ['srvc', 'service', 'service_company'], ['uwi'],
];
const window = {};
const curveLayerRevisions = new Map([['GR::main', 1], ['GR::wrap', 1]]);
let committedDatasetGeneration = 7;
let committedSnapshotSequence = 0;
let metadataStateRevision = 3;
let latestSaveOperationId = 0;
let latestExportOperationId = 0;
let lastSavedSnapshotDescriptor = null;
let workspaceHasUnsavedCommittedChanges = true;
let lastMetadataFingerprint = null;
let lastDigitizedDepth = [1000, 1001, 1002, 1003];
let lastDigitizedCurves = {
    GR: {
        unit: 'API',
        values: [10, 20, 30, 40],
        wrapLayer: { values: [-999.25, 120, null, 140], left_value: 100, right_value: 200, color: '#0ff' },
    },
};
let headerMetadata = { well: 'ALPHA', nested: { owner: 'original' } };
let lastDepthConfig = { unit: 'FT', top_depth: 1000, bottom_depth: 1003 };
let lastNullValue = -999.25;
let activeLogId = 55;
let uploadedImagePath = '/api/images/image-a';
let originalUploadedImagePath = '/api/images/image-a';
let lastLasFilename = 'alpha.las';
let curveWrapMarkers = { GR: [{ depthIndex: 1, cycle: 1 }] };
let primaryRegion = { left_px: 1, right_px: 10 };
let headerRegion = { left_px: 2, right_px: 9 };
let pinnedDepthIndices = new Map([['GR', new Set([1, 3])]]);
let editDragState = { previewValues: [999, 999, 999, 999], dragLayer: 'main' };
let drawStrokeActive = false, paintStrokeActive = false, smoothStrokeActive = false, eraserStart = null;
function normalizeCurveKey(value) { return String(value || '').trim().toUpperCase(); }
function getCurveLayerRevision(curveId, layer) { return curveLayerRevisions.get(`${normalizeCurveKey(curveId)}::${layer}`) || 0; }
function syncHeaderMetadataFromInputs() { return headerMetadata; }
function buildDigitizeConfigFromInputs() {
    return { depth: { top_depth: 1000, bottom_depth: 1003 }, curves: [{ las_mnemonic: 'GR', scale_min: 0, scale_max: 100 }] };
}
function getDepthConfigFromInputs() { return lastDepthConfig; }
function isMissingDigitizedValue(value, sentinel) {
    return value == null || !Number.isFinite(value) || (Number.isFinite(sentinel) && value === sentinel);
}
function showStatus() {}

const snapshot = createCommittedDatasetSnapshot({ purpose: 'test_save' });
assert(Object.isFrozen(snapshot));
assert(Object.isFrozen(snapshot.curves.GR.values));
assert.strictEqual(snapshot.hadActivePreview, true);
assert.deepStrictEqual(snapshot.curves.GR.values, [10, 20, 30, 40]);
assert.deepStrictEqual(snapshot.revisionManifest, { 'GR::main': 1, 'GR::wrap': 1 });
assert.deepStrictEqual(snapshot.pinnedDepthIndices, { GR: [1, 3] });
assert.strictEqual(markCommittedSnapshotSaved(snapshot), true);
assert.strictEqual(workspaceHasUnsavedCommittedChanges, false);

lastDigitizedDepth[0] = 9000;
lastDigitizedCurves.GR.values[0] = 999;
lastDigitizedCurves.GR.wrapLayer.values[1] = 999;
headerMetadata.well = 'CHANGED';
headerMetadata.nested.owner = 'changed';
assert.deepStrictEqual(snapshot.depths, [1000, 1001, 1002, 1003]);
assert.deepStrictEqual(snapshot.curves.GR.values, [10, 20, 30, 40]);
assert.deepStrictEqual(snapshot.curves.GR.wrapLayer.values, [-999.25, 120, null, 140]);
assert.strictEqual(snapshot.metadata.well, 'ALPHA');
assert.strictEqual(snapshot.metadata.nested.owner, 'original');
try { snapshot.curves.GR.values[0] = 777; } catch (_) {}
assert.strictEqual(lastDigitizedCurves.GR.values[0], 999);

assert.deepStrictEqual(buildMergedCurveValuesForExport(snapshot.curves.GR, snapshot.nullValue), [10, 120, 30, 140]);
const mergeInput = {
    values: [10, 20, 30, 40],
    wrapLayer: { values: [-999.25, null, NaN, 140] },
};
const mergeBefore = stablePersistenceStringify(mergeInput);
assert.deepStrictEqual(buildMergedCurveValuesForExport(mergeInput, -999.25), [10, 20, 30, 140]);
assert.strictEqual(stablePersistenceStringify(mergeInput), mergeBefore);
assert.throws(() => buildMergedCurveValuesForExport({ values: [1, 2], wrapLayer: { values: [3] } }, -999.25));

const lasA = buildLasFromCommittedSnapshot(snapshot);
lastDigitizedCurves.GR.values.fill(555);
headerMetadata.well = 'LATEST';
const lasB = buildLasFromCommittedSnapshot(snapshot);
assert.strictEqual(lasA, lasB);
const dataLines = lasA.split(/\r?\n/).filter(line => /^\s*100[0-3]\.0000/.test(line));
assert.strictEqual(dataLines.length, 4);
assert(dataLines[0].includes('10.0000'));
assert(dataLines[1].includes('120.0000'));
assert(dataLines[2].includes('30.0000'));
assert(dataLines[3].includes('140.0000'));

const stateChunks = [];
let inState = false;
for (const line of lasA.split(/\r?\n/)) {
    if (line.includes('TURBOTIFF_STATE_START')) { inState = true; continue; }
    if (line.includes('TURBOTIFF_STATE_END')) { inState = false; continue; }
    if (inState && line.startsWith('# ')) stateChunks.push(line.slice(2).trim());
}
const state = JSON.parse(decodeURIComponent(escape(Buffer.from(stateChunks.join(''), 'base64').toString('binary'))));
assert.deepStrictEqual(state.digitizedDepth, snapshot.depths);
assert.deepStrictEqual(state.digitizedCurves.GR.values, snapshot.curves.GR.values);
assert.deepStrictEqual(state.digitizedCurves.GR.wrapLayer.values, snapshot.curves.GR.wrapLayer.values);
assert.deepStrictEqual(state.pinnedDepthIndices, { GR: [1, 3] });
assert.strictEqual(Object.prototype.hasOwnProperty.call(state, 'curveTraces'), false);
assert.strictEqual(Object.prototype.hasOwnProperty.call(state, 'bezierCurveControls'), false);

lastDigitizedDepth = [1000, 1001, 1002, 1003];
lastDigitizedCurves = clonePersistenceValue(snapshot.curves);
headerMetadata = clonePersistenceValue(snapshot.metadata);
curveLayerRevisions.set('GR::main', 2);
lastDigitizedCurves.GR.values[0] = 11;
const committedSnapshot = createCommittedDatasetSnapshot({ purpose: 'after_commit' });
assert.strictEqual(committedSnapshot.curves.GR.values[0], 11);
assert.strictEqual(committedSnapshot.revisionManifest['GR::main'], 2);
assert.strictEqual(markCommittedSnapshotSaved(snapshot), false);
assert.strictEqual(workspaceHasUnsavedCommittedChanges, true);
assert.strictEqual(markCommittedSnapshotSaved(committedSnapshot), true);
assert.strictEqual(workspaceHasUnsavedCommittedChanges, false);

lastDigitizedCurves.GR.wrapLayer.values = [1, 2];
assert.throws(
    () => createCommittedDatasetSnapshot({ purpose: 'bad_lengths' }),
    error => error && error.code === 'invalid_committed_dataset' && error.validation.errors.some(item => item.code === 'wrap_length_mismatch')
);
"""
    result = subprocess.run(
        [node, "-"],
        input="\n".join((snapshot_source, canonical_source, las_source, harness)).encode("utf-8"),
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def test_persistence_paths_capture_once_and_guard_async_save_ordering():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    zip_source = _function_source(source, "downloadLasZip", "autoSaveLogToAccount")
    autosave_source = _function_source(source, "autoSaveLogToAccount", "downloadLastLas")
    manual_source = _function_source(source, "saveLogToAccount", "showStatus")
    download_source = _function_source(source, "downloadLastLas", "saveLogToAccount")

    assert "createCommittedDatasetSnapshot" in zip_source
    assert "buildExportCurvesFromSnapshot(snapshot)" in zip_source
    assert "null_value: snapshot.nullValue" in zip_source
    assert "latestExportOperationId" in zip_source
    assert "lastDigitizedCurves" not in zip_source
    assert "lastDigitizedDepth" not in zip_source

    for function_source in (autosave_source, manual_source):
        assert function_source.count("createCommittedDatasetSnapshot") == 1
        assert "latestSaveOperationId" in function_source
        assert "operationId !== latestSaveOperationId" in function_source
        assert "isPersistenceSnapshotDatasetActive(snapshot)" in function_source
        assert "markCommittedSnapshotSaved(snapshot)" in function_source
        assert "buildLasFromCommittedSnapshot(snapshot)" in function_source

    assert download_source.count("createCommittedDatasetSnapshot") == 1
    assert "buildLasFromCommittedSnapshot(snapshot)" in download_source
    assert "lastDigitizedCurves" not in download_source
    assert "lastDigitizedDepth" not in download_source


def test_zip_export_endpoint_rejects_dimension_and_numeric_corruption():
    source = WEB_APP.read_text(encoding="utf-8")
    start = source.index("def download_las_zip():")
    end = source.index("@app.route('/api/ml_predict_curve_trace'", start)
    endpoint = source[start:end]

    assert "Depths must be strictly monotonic" in endpoint
    assert "values must match the depth count" in endpoint
    assert "contains infinite values" in endpoint
    assert "null_value = data.get('null_value', -999.25)" in endpoint
    assert "write_las_simple(depth_arr, single_curve_data, depth_unit, header_metadata, null_value)" in endpoint
    assert "min_len = min(" not in endpoint


def test_optional_curve_pipeline_imports_cannot_prevent_core_app_boot():
    source = WEB_APP.read_text(encoding="utf-8")

    assert "try:\n    from curve_model.integration" in source
    assert "except ImportError as optional_curve_pipeline_error:" in source
    assert "OPTIONAL_CURVE_PIPELINES_AVAILABLE = False" in source
    assert "def build_phase1_probability(roi, classic_mask, **_kwargs):" in source


def test_trace_continuity_gate_keeps_missing_evidence_as_gaps_not_bridges():
    source = WEB_APP.read_text(encoding="utf-8")
    gate_start = source.index("def enforce_local_trace_continuity(")
    gate_end = source.index("def should_preserve_black_trace_detail(", gate_start)
    gate_source = source[gate_start:gate_end]
    digitize_start = source.index("# Final evidence/continuity gate.")
    display_start = source.index("# Preserve trace gaps in the display")
    display_end = source.index("curve_traces[name] = trace_points", display_start)

    assert "vertical_support = cv2.blur(prob, (3, 5))" in gate_source
    assert "gap_rows += 1" in gate_source
    assert "result[y] = candidate" in gate_source
    assert "xs = enforce_local_trace_continuity(" in source[digitize_start:display_start]
    assert ".interpolate(" not in source[display_start:display_end]


def test_overlay_breaks_on_any_missing_trace_row():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    overlay_source = _function_source(source, "renderCurveTraceOverlays", "clearDepthOverlays")

    assert "const jumpBreakMaxRowGap = 1.0;" in overlay_source


def test_project_load_validates_and_restores_authoritative_layers_pins_and_clean_state():
    source = WORKSPACE_TEMPLATE.read_text(encoding="utf-8")
    load_source = _function_source(source, "ensureDigitizedFromLas", "buildLasFromDigitized")

    validation_pos = load_source.index("validateCommittedDatasetSnapshot")
    replacement_pos = load_source.index("replaceAllDigitizedCurves(stateBlob.digitizedCurves")
    assert validation_pos < replacement_pos
    assert "rebuildTraces: true" in load_source
    assert "pinnedDepthIndices.set(normalizeCurveKey(curveKey), new Set(validIndices))" in load_source
    assert "curveWrapMarkers = clonePersistenceValue(stateBlob.curveWrapMarkers)" in load_source
    assert "markCurrentCommittedStateClean('project_loaded')" in load_source
