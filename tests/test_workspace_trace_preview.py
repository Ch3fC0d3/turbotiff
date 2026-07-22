import shutil
import subprocess
from pathlib import Path

import pytest


WORKSPACE_TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "workspace.html"


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

    calculate = drag_source.index("const proposals = neighbors.map(n =>")
    resolve = drag_source.index("resolveDepthMoveProposals(proposals, originalValues, compareProposalPriority);", calculate)
    snapshot = drag_source.index("const nextValues = originalValues.slice();", resolve)
    clear_source = drag_source.index("nextValues[proposal.sourceDepthIndex] = null;", snapshot)
    write_destination = drag_source.index("nextValues[proposal.destinationDepthIndex] = proposal.newValue;", clear_source)
    commit_values = drag_source.index("activeValues[i] = nextValues[i];", write_destination)
    commit_depth = drag_source.index("proposal.neighbor.depthIndex = proposal.accepted", commit_values)

    assert calculate < resolve < snapshot < clear_source < write_destination < commit_values < commit_depth
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
    assert "getActiveLayerTrackConfig(entry, targetTrack, activeLayer)" in drag_source
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
    trace_write = regular_source.index("lastCurveTraces[traceKey] = pts;", point_check)
    drag_state = regular_source.index("editDragState = {", trace_write)

    assert hit < layer_check < pin_check < point_check < trace_write < drag_state
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
