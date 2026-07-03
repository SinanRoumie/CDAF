"""
CDAF Graph Builder
==================

An interactive web app for constructing and visualizing CDAF graphs -- a
node-and-edge representation of a competitive policy debate round.

Spatial organization (a 2.5D / isometric view of the round):
  * X axis  -- speech progression, 1AC -> 2AR, left to right (one column each).
  * Y axis  -- vertical position within a layer (infinitely scrollable).
  * Z axis  -- layer depth, rendered as an isometric offset:
               content (front) -> framework (middle) -> ballot (back).
Each speech is a colored column: green border for AFF speeches, red for NEG.

The live cytoscape instance is enhanced clientside (see assets/cdaf.js) for
custom selection, drag/pan clamping, and floating column headers.

Run with:
    python cdaf_app.py
then open http://127.0.0.1:8050 in your browser.
"""

import base64
import copy
import json
import re

import dash_cytoscape as cyto
from dash import (Dash, Input, Output, State, ctx, dcc, html, no_update)
from dash.dependencies import ClientsideFunction

# Domain schema lives in model/ (pure Python, no UI deps) -- single source of truth.
from model import (NODE_CLASSES, EDGE_CLASSES, NODE_TYPE_NAMES, EDGE_TYPE_NAMES, Position)
from model import serialize as mser

# Deterministic judge + RFD live in judge/ (pure Python: imports only model/ and
# the stdlib, NEVER anything app-side). The app depends on judge; not vice versa.
from judge import judge as run_judge
from judge import rfd as judge_rfd

# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

LAYERS = ["content", "framework", "ballot"]
LAYER_DEPTH = {"content": 0, "framework": 1, "ballot": 2}

# UI metadata per node type. The type NAMES + order come from model/ (one
# source of truth for the schema); layer/color are app-side presentation only.
_NODE_LAYER = {
    "Uniqueness": "content", "Link": "content", "Impact": "content", "Advocacy": "content",
    "Framework": "framework", "Weighing": "framework", "BallotDirective": "ballot",
}
_NODE_COLOR = {
    "Uniqueness": "#4C9F70", "Link": "#3A7CA5", "Impact": "#D1495B", "Advocacy": "#9B5DE5",
    "Framework": "#E08E45", "Weighing": "#C9A227", "BallotDirective": "#5C6672",
}
# (name, layer, color)
NODE_TYPES = [(name, _NODE_LAYER[name], _NODE_COLOR[name]) for name in NODE_TYPE_NAMES]
NODE_LAYER = {name: layer for name, layer, _ in NODE_TYPES}
NODE_COLOR = {name: color for name, _, color in NODE_TYPES}

LAYER_SHAPE = {"content": "round-rectangle", "framework": "hexagon", "ballot": "diamond"}

# Fixed, immutable speech order. "2NC/1NR" is one combined speech.
SPEECHES = ["1AC", "1NC", "2AC", "2NC/1NR", "1AR", "2NR", "2AR"]
SPEECH_INDEX = {s: i for i, s in enumerate(SPEECHES)}
SPEECH_SIDE = {
    "1AC": "AFF", "1NC": "NEG", "2AC": "AFF", "2NC/1NR": "NEG",
    "1AR": "AFF", "2NR": "NEG", "2AR": "AFF",
}

# UI style per edge type, keyed by the on-disk etype string (names + order from model/).
_EDGE_STYLE = {
    "SupportEdge": ("#2A9D8F", "solid", "triangle", 3, False),
    "ExtensionEdge": ("#457B9D", "dashed", "triangle", 3, False),
    "DefensiveAttackEdge": ("#E76F51", "dotted", "tee", 3, False),
    "OffensiveAttackEdge": ("#D62828", "solid", "triangle", 5, False),
    "ComparisonEdge": ("#6A4C93", "dashed", "diamond", 3, True),
}
# (name, color, line-style, target-arrow, width, comparison?)
EDGE_TYPES = [(name, *_EDGE_STYLE[name]) for name in EDGE_TYPE_NAMES]

# AFF vs NEG node border color (green / red).
SIDE_BORDER = {"AFF": "#2E8B57", "NEG": "#C0392B"}

# Light per-column tint (columns no longer carry a border).
SIDE_COLUMN = {"AFF": {"tint": "#e8f5ee"}, "NEG": {"tint": "#fbe9e9"}}

# Isometric layout geometry (kept in sync with assets/cdaf.js).
LEFT = 120
COL_W = 280            # columns are adjacent (no gap): band width == COL_W
DEPTH_DX = 46          # horizontal isometric offset per layer (content/framework/ballot)
NODE_W = 170
ROW_H = 88             # vertical stacking step
TOP_Y = 80             # first node row of every column
WORLD_TOP = 20         # top of the graph (panning can't go above this)
BG_HEIGHT = 12000
BG_CENTER_Y = WORLD_TOP + BG_HEIGHT // 2

VISIBLE = {"display": "block"}
HIDDEN = {"display": "none"}
MODAL_VISIBLE = {"display": "flex"}
MODAL_HIDDEN = {"display": "none"}
POPUP_VISIBLE = {"display": "block"}   # non-blocking floating card
POPUP_HIDDEN = {"display": "none"}
EMPTY_SELECTION = {"nodes": [], "edge": None}


# ---------------------------------------------------------------------------
# Model helpers (cytoscape `elements` is the source of truth)
# ---------------------------------------------------------------------------

def band_center(col):
    return LEFT + col * COL_W + COL_W / 2


def is_bg(el):
    return str(el["data"].get("id", "")).startswith("__")


def is_node(el):
    return "source" not in el["data"]


def model_elements(elements):
    """The user's nodes/edges -- excludes the column background/header nodes."""
    return [el for el in (elements or []) if not is_bg(el)]


def model_nodes(elements):
    return [el for el in model_elements(elements) if is_node(el)]


def model_edges(elements):
    return [el for el in model_elements(elements) if not is_node(el)]


def infer_side(speech):
    return SPEECH_SIDE.get(speech, "AFF")


def next_id(elements, prefix):
    nums = [0]
    for el in model_elements(elements):
        m = re.fullmatch(prefix + r"(\d+)", str(el["data"].get("id", "")))
        if m:
            nums.append(int(m.group(1)))
    return f"{prefix}{max(nums) + 1}"


def cell_position(col, depth, slot):
    # Stack vertically from the top of the column; layer shifts x (isometric depth).
    return {"x": band_center(col) + (depth - 1) * DEPTH_DX,
            "y": TOP_Y + slot * ROW_H}


def derive_position(elements, speech, layer):
    # First node of a speech is at the top; each later node stacks directly below,
    # regardless of layer -> slot counts every node already in that speech.
    slot = sum(1 for n in model_nodes(elements) if n["data"]["speech"] == speech)
    return cell_position(SPEECH_INDEX.get(speech, 0), LAYER_DEPTH[layer], slot)


def relayout_positions(nodes):
    """Re-derive positions: one vertical stack per speech, in insertion order."""
    counters = {}
    for n in nodes:
        speech = n["data"]["speech"]
        layer = NODE_LAYER[n["data"]["ntype"]]
        slot = counters.get(speech, 0)
        counters[speech] = slot + 1
        n["position"] = cell_position(SPEECH_INDEX.get(speech, 0), LAYER_DEPTH[layer], slot)
    return nodes


def ensure_positions(model):
    """Give every node a position. No-op for app-saved rounds (which always carry
    positions); fills in positions for hand-authored rounds that omit them."""
    nodes = [n for n in model if is_node(n)]
    if any(("position" not in n) or (n.get("position") is None) for n in nodes):
        relayout_positions(nodes)


def find(elements, el_id):
    for el in model_elements(elements):
        if el["data"]["id"] == el_id:
            return el
    return None


def edge_exists_between(elements, a, b):
    for e in model_edges(elements):
        s, t = e["data"]["source"], e["data"]["target"]
        if {s, t} == {a, b}:
            return True
    return False


def make_node(node_id, label, ntype, speech, position):
    node = NODE_CLASSES[ntype](id=node_id, label=label, side=infer_side(speech),
                               speech=speech, position=Position(position["x"], position["y"]))
    return mser.node_to_element(node)


def make_edge(edge_id, source, target, etype):
    edge = EDGE_CLASSES[etype](id=edge_id, source=source, target=target)
    return mser.edge_to_element(edge)


def background_elements():
    """Tinted, adjacent column rectangles (regenerated each render). Headers are a
    separate fixed HTML overlay so they never pan/scroll into the page body."""
    els = []
    for s in SPEECHES:
        c = SPEECH_INDEX[s]
        els.append({
            "data": {"id": f"__col_{c}", "bg": "1", "col": c,
                     "tint": SIDE_COLUMN[SPEECH_SIDE[s]]["tint"]},
            "position": {"x": band_center(c), "y": BG_CENTER_Y},
            "selectable": False, "grabbable": False,
            "classes": "colbg",
        })
    return els


def render_elements(model):
    return background_elements() + model


def initial_elements():
    return background_elements()


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

def build_stylesheet():
    sheet = [
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "text-wrap": "wrap",
                "text-max-width": "150px",
                "text-valign": "center",
                "text-halign": "center",
                "color": "#ffffff",
                "font-size": "12px",
                "font-weight": "bold",
                "text-outline-width": 1,
                "text-outline-color": "#00000055",
                "width": f"{NODE_W}px",
                "height": "62px",
                "border-width": 4,
                "padding": "6px",
                "z-index": 10,
            },
        },
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier",
                "width": 3,
                "line-color": "#999",
                "target-arrow-color": "#999",
                "target-arrow-shape": "triangle",
                "z-index": 9,
            },
        },
        # Column backgrounds: adjacent, tinted, no border, non-interactive.
        {
            "selector": ".colbg",
            "style": {
                "shape": "rectangle",
                "width": COL_W,
                "height": BG_HEIGHT,
                "background-color": "data(tint)",
                "background-opacity": 0.55,
                "border-width": 0,
                "events": "no",
                "z-index": 0,
            },
        },
        {"selector": "node:selected",
         "style": {"overlay-color": "#FFD400", "overlay-opacity": 0.35, "overlay-padding": 8}},
        {"selector": "edge:selected",
         "style": {"overlay-color": "#FFD400", "overlay-opacity": 0.4, "overlay-padding": 6}},
    ]

    # Node color by type / shape by layer.
    for name, layer, color in NODE_TYPES:
        sheet.append({"selector": f'node[ntype="{name}"]',
                      "style": {"background-color": color, "shape": LAYER_SHAPE[layer]}})

    # Node border by side.
    for side, border in SIDE_BORDER.items():
        sheet.append({"selector": f'node[side="{side}"]', "style": {"border-color": border}})

    # Edge style by type.
    for name, color, line_style, arrow, width, comparison in EDGE_TYPES:
        style = {
            "line-color": color, "line-style": line_style, "width": width,
            "target-arrow-color": color, "target-arrow-shape": arrow, "source-arrow-color": color,
        }
        if comparison:
            style["source-arrow-shape"] = arrow
        sheet.append({"selector": f'edge[etype="{name}"]', "style": style})

    return sheet


# ---------------------------------------------------------------------------
# Legends
# ---------------------------------------------------------------------------

def swatch(color, shape="square"):
    base = {"display": "inline-block", "width": "16px", "height": "16px",
            "backgroundColor": color, "marginRight": "8px", "verticalAlign": "middle",
            "border": "1px solid #00000033"}
    if shape == "round":
        base.update({"borderRadius": "4px"})
    if shape == "diamond":
        base.update({"transform": "rotate(45deg)", "width": "12px", "height": "12px"})
    return html.Span(style=base)


def line_swatch(color, line_style):
    return html.Span(style={"display": "inline-block", "width": "26px",
                            "borderTop": f"3px {line_style} {color}", "marginRight": "8px",
                            "verticalAlign": "middle"})


def node_legend():
    rows = []
    for name, layer, color in NODE_TYPES:
        shape = "diamond" if layer == "ballot" else ("round" if layer == "content" else "square")
        rows.append(html.Div(
            [swatch(color, shape), html.Span(name, style={"fontSize": "13px"}),
             html.Span(f"  ·  {layer}", style={"fontSize": "11px", "color": "#888"})],
            style={"marginBottom": "5px"}))
    side_rows = [html.Div(
        [html.Span(style={"display": "inline-block", "width": "14px", "height": "14px",
                          "marginRight": "8px", "verticalAlign": "middle", "backgroundColor": "#ddd",
                          "border": f"3px solid {border}"}),
         html.Span(f"{side} node border", style={"fontSize": "13px"})],
        style={"marginBottom": "5px"}) for side, border in SIDE_BORDER.items()]
    return html.Div(rows
                    + [html.Div("Side (node border)",
                                style={"fontWeight": "bold", "margin": "8px 0 5px"})]
                    + side_rows)


def edge_legend():
    rows = []
    for name, color, line_style, arrow, width, comparison in EDGE_TYPES:
        suffix = " (both ends)" if comparison else ""
        rows.append(html.Div(
            [line_swatch(color, line_style), html.Span(f"{name}{suffix}", style={"fontSize": "13px"})],
            style={"marginBottom": "5px"}))
    return html.Div(rows)


# ---------------------------------------------------------------------------
# UI building blocks
# ---------------------------------------------------------------------------

def section(title, children):
    return html.Div([html.Div(title, className="section-title"), html.Div(children)],
                    className="panel-section")


def labeled(label, component):
    return html.Div([html.Label(label, className="field-label"), component],
                    style={"marginBottom": "8px"})


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "CDAF Graph Builder"

node_type_options = [{"label": n, "value": n} for n, _, _ in NODE_TYPES]
edge_type_options = [{"label": n, "value": n} for n, *_ in EDGE_TYPES]
speech_options = [{"label": s, "value": s} for s in SPEECHES]


# ---- Left control panel ----------------------------------------------------

add_node_panel = section("Add node", [
    labeled("Type", dcc.Dropdown(id="nt-type", options=node_type_options,
                                 value="Link", clearable=False)),
    labeled("Claim / label", dcc.Textarea(id="nt-label", value="",
                                           placeholder="text of the claim…",
                                           style={"width": "100%", "height": "52px"})),
    labeled("Speech (side is inferred)",
            dcc.Dropdown(id="nt-speech", options=speech_options, value="1AC", clearable=False)),
    html.Div(id="nt-side-hint", className="hint small"),
    html.Button("+ Add node", id="add-node-btn", className="btn primary"),
])

add_edge_panel = section("Add edge", [
    html.Div("Pick source + target + type, or hold ⌘ and click two nodes.",
             className="hint small"),
    labeled("Source node", dcc.Dropdown(id="edge-source", options=[], placeholder="from…")),
    labeled("Target node", dcc.Dropdown(id="edge-target", options=[], placeholder="to…")),
    labeled("Edge type", dcc.Dropdown(id="edge-type", options=edge_type_options,
                                      value="SupportEdge", clearable=False)),
    html.Button("+ Add edge", id="add-edge-btn", className="btn primary"),
    html.Div(id="edge-msg", className="msg"),
])

def make_upload():
    """A fresh dcc.Upload. Re-created after each load so its underlying
    <input type=file> is reset and the same file can be uploaded again."""
    return dcc.Upload(id="upload-json", children=html.Div(["⤒ Load JSON (drag or click)"]),
                      className="upload-box", multiple=False)


file_panel = section("Save / load", [
    labeled("File name", dcc.Input(id="save-name", type="text", value="",
                                   placeholder="cdaf_graph", style={"width": "100%"})),
    html.Button("⤓ Save JSON", id="save-btn", className="btn"),
    dcc.Download(id="download-json"),
    html.Div(style={"height": "8px"}),
    html.Div(id="upload-wrapper", children=make_upload()),
    html.Div(style={"height": "8px"}),
    html.Button("Re-run layout", id="relayout-btn", className="btn"),
    html.Button("Clear graph", id="clear-btn", className="btn danger"),
])

judge_panel = section("Judge round", [
    html.Div("Evaluates the current graph with the deterministic judge. "
             "Runs only when you press the button.", className="hint small"),
    html.Button("⚖ Judge Round", id="judge-btn", className="btn primary"),
    html.Div(id="judge-output",
             children=html.Div("Press “Judge Round” to evaluate the current graph.",
                               className="hint small")),
])

left_panel = html.Div([
    html.H2("CDAF Builder", className="app-title"),
    add_node_panel, add_edge_panel, file_panel, judge_panel,
], className="left-panel")


# ---- Inspector (right panel) ----------------------------------------------

node_editor = html.Div(id="node-editor", style=HIDDEN, children=[
    html.Div("Node", className="section-title"),
    html.Div(id="edit-node-meta", className="meta"),
    labeled("Type", dcc.Dropdown(id="edit-node-type", options=node_type_options,
                                 clearable=False)),
    labeled("Claim / label", dcc.Textarea(id="edit-node-label",
                                           style={"width": "100%", "height": "70px"})),
    labeled("Speech (side is inferred)",
            dcc.Dropdown(id="edit-node-speech", options=speech_options, clearable=False)),
    html.Button("Save changes", id="edit-node-save", className="btn primary"),
    html.Button("Delete node", id="edit-node-delete", className="btn danger"),
])

edge_editor = html.Div(id="edge-editor", style=HIDDEN, children=[
    html.Div("Edge", className="section-title"),
    html.Div(id="edit-edge-meta", className="meta"),
    html.Button("⇄ Reverse direction", id="edit-edge-reverse", className="btn"),
    labeled("Edge type", dcc.Dropdown(id="edit-edge-type", options=edge_type_options,
                                      clearable=False)),
    html.Button("Save changes", id="edit-edge-save", className="btn primary"),
    html.Button("Delete edge", id="edit-edge-delete", className="btn danger"),
])

inspector = html.Div([
    html.H3("Inspector", className="panel-heading"),
    node_editor, edge_editor,
    html.Hr(),
    html.Details([html.Summary("Node legend"), node_legend()], open=True, className="legend"),
    html.Details([html.Summary("Edge legend"), edge_legend()], open=True, className="legend"),
], className="right-panel")


# ---- Modals ----------------------------------------------------------------

# Non-blocking floating card: leaves the graph clickable so a third click can
# replace the oldest selection (FIFO) while two nodes are selected.
choose_modal = html.Div(id="choose-modal", style=POPUP_HIDDEN, className="popup-card", children=[
    html.Div("Two nodes selected", className="section-title"),
    html.Div(id="choose-info", className="meta"),
    html.Button("Create Edge", id="choose-create-edge", className="btn primary"),
    html.Button("Create Weighing Node", id="choose-weighing", className="btn primary"),
    html.Div(style={"height": "6px"}),
    html.Button("Cancel", id="choose-cancel", className="btn"),
])

edge_dialog = html.Div(id="edge-dialog", style=MODAL_HIDDEN, className="modal-overlay", children=[
    html.Div(className="modal-box", children=[
        html.Div("Create edge", className="section-title"),
        html.Div(id="dialog-info", className="meta"),
        labeled("Edge type", dcc.Dropdown(id="dialog-edge-type", options=edge_type_options,
                                          value="SupportEdge", clearable=False)),
        labeled("Direction", dcc.RadioItems(id="dialog-direction", options=[], value="ab")),
        html.Div(id="dialog-msg", className="msg"),
        html.Button("Create", id="dialog-create", className="btn primary"),
        html.Button("Cancel", id="dialog-cancel", className="btn"),
    ]),
])

weighing_dialog = html.Div(id="weighing-dialog", style=MODAL_HIDDEN, className="modal-overlay",
                           children=[html.Div(className="modal-box", children=[
    html.Div("Create weighing node", className="section-title"),
    html.Div(id="weighing-info", className="meta"),
    labeled("Label", dcc.Input(id="weighing-label", type="text", value="Weighing",
                               style={"width": "100%"})),
    labeled("Speech (side is inferred)",
            dcc.Dropdown(id="weighing-speech", options=speech_options, value="1AC", clearable=False)),
    html.Button("Create", id="weighing-create", className="btn primary"),
    html.Button("Cancel", id="weighing-cancel", className="btn"),
])])


# ---- Graph canvas ----------------------------------------------------------

# Sticky column headers: a fixed HTML strip over the canvas. Their horizontal
# position is kept aligned to the columns by the clientside refresh handler;
# their vertical position never changes, so they can't drift into the page body.
header_bar = html.Div(
    id="cdaf-header-bar", className="header-bar",
    children=[html.Div(s, id=f"cdaf-hdr-{SPEECH_INDEX[s]}",
                       className=f"col-header col-header-{SPEECH_SIDE[s]}")
              for s in SPEECHES],
)

graph_area = html.Div([
    header_bar,
    cyto.Cytoscape(
        id="cytoscape",
        elements=initial_elements(),
        layout={"name": "preset", "fit": False},
        stylesheet=build_stylesheet(),
        style={"width": "100%", "height": "92vh"},
        autoRefreshLayout=False,   # never auto-relayout on element changes (preserves view)
        minZoom=0.25, maxZoom=2.2,
        boxSelectionEnabled=False,
        autoungrabify=False,
    ),
], className="graph-area")


# ---- Assemble --------------------------------------------------------------

app.layout = html.Div([
    dcc.Store(id="selection-store", data=EMPTY_SELECTION),
    dcc.Store(id="apply-positions", data={}),
    dcc.Store(id="applysel-dummy", data=""),
    dcc.Store(id="applypos-dummy", data=""),
    left_panel, graph_area, inspector, choose_modal, edge_dialog, weighing_dialog,
], className="app-root")


# ---------------------------------------------------------------------------
# Clientside callbacks (mirror selection highlight; apply Dash-driven positions)
# cy event handlers are attached by a self-contained poller in assets/cdaf.js.
# prevent_initial_call avoids invoking the namespace before the asset registers.
# ---------------------------------------------------------------------------

app.clientside_callback(
    ClientsideFunction(namespace="cdaf", function_name="applySelection"),
    Output("applysel-dummy", "data"),
    Input("selection-store", "data"),
    prevent_initial_call=True,
)

app.clientside_callback(
    ClientsideFunction(namespace="cdaf", function_name="applyPositions"),
    Output("applypos-dummy", "data"),
    Input("apply-positions", "data"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Add-node side hint
# ---------------------------------------------------------------------------

@app.callback(Output("nt-side-hint", "children"), Input("nt-speech", "value"))
def side_hint(speech):
    return f"→ this node will be {infer_side(speech)}"


# ---------------------------------------------------------------------------
# Inspector + action popup driven by the selection
# ---------------------------------------------------------------------------

@app.callback(
    Output("node-editor", "style"),
    Output("edge-editor", "style"),
    Output("choose-modal", "style"),
    Output("edge-dialog", "style"),
    Output("weighing-dialog", "style"),
    Output("edit-node-type", "value"),
    Output("edit-node-label", "value"),
    Output("edit-node-speech", "value"),
    Output("edit-node-meta", "children"),
    Output("edit-edge-type", "value"),
    Output("edit-edge-meta", "children"),
    Output("choose-info", "children"),
    Input("selection-store", "data"),
    State("cytoscape", "elements"),
)
def reflect_selection(sel, elements):
    sel = sel or EMPTY_SELECTION
    nodes = sel.get("nodes", [])
    edge_id = sel.get("edge")

    if edge_id:
        e = find(elements, edge_id)
        src = find(elements, e["data"]["source"]) if e else None
        tgt = find(elements, e["data"]["target"]) if e else None
        meta = html.Span(
            f"{(src['data']['label'][:22] if src else '?')}  →  "
            f"{(tgt['data']['label'][:22] if tgt else '?')}")
        return (HIDDEN, VISIBLE, POPUP_HIDDEN, MODAL_HIDDEN, MODAL_HIDDEN,
                no_update, no_update, no_update, no_update,
                (e["data"]["etype"] if e else "SupportEdge"), meta, no_update)

    if len(nodes) == 1:
        n = find(elements, nodes[0])
        if n:
            meta = html.Span(f"{n['data']['ntype']}  ·  {n['data']['side']}  ·  {n['data']['speech']}")
            return (VISIBLE, HIDDEN, POPUP_HIDDEN, MODAL_HIDDEN, MODAL_HIDDEN,
                    n["data"]["ntype"], n["data"]["label"], n["data"]["speech"], meta,
                    no_update, no_update, no_update)

    if len(nodes) == 2:
        a, b = find(elements, nodes[0]), find(elements, nodes[1])
        info = html.Span(f"{(a['data']['label'][:22] if a else nodes[0])}  ↔  "
                         f"{(b['data']['label'][:22] if b else nodes[1])}")
        return (HIDDEN, HIDDEN, POPUP_VISIBLE, MODAL_HIDDEN, MODAL_HIDDEN,
                no_update, no_update, no_update, no_update, no_update, no_update, info)

    # nothing selected -> blank inspector, no modals
    return (HIDDEN, HIDDEN, POPUP_HIDDEN, MODAL_HIDDEN, MODAL_HIDDEN,
            no_update, no_update, no_update, no_update, no_update, no_update, no_update)


# ---------------------------------------------------------------------------
# "Create Edge" -> open the edge dialog (with direction options)
# ---------------------------------------------------------------------------

@app.callback(
    Output("choose-modal", "style", allow_duplicate=True),
    Output("edge-dialog", "style", allow_duplicate=True),
    Output("dialog-direction", "options"),
    Output("dialog-direction", "value"),
    Output("dialog-edge-type", "value"),
    Output("dialog-info", "children"),
    Output("dialog-msg", "children"),
    Input("choose-create-edge", "n_clicks"),
    State("selection-store", "data"),
    State("cytoscape", "elements"),
    prevent_initial_call=True,
)
def open_edge_dialog(_n, sel, elements):
    nodes = (sel or EMPTY_SELECTION).get("nodes", [])
    if len(nodes) != 2:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update
    a, b = find(elements, nodes[0]), find(elements, nodes[1])
    la = a["data"]["label"][:20] if a else nodes[0]
    lb = b["data"]["label"][:20] if b else nodes[1]
    options = [{"label": f"{la}  →  {lb}", "value": "ab"},
               {"label": f"{lb}  →  {la}", "value": "ba"}]
    info = html.Span(f"Between “{la}” and “{lb}”")
    return POPUP_HIDDEN, MODAL_VISIBLE, options, "ab", "SupportEdge", info, ""


# ---------------------------------------------------------------------------
# "Create Weighing Node" -> open the weighing dialog (label + speech)
# ---------------------------------------------------------------------------

@app.callback(
    Output("choose-modal", "style", allow_duplicate=True),
    Output("weighing-dialog", "style", allow_duplicate=True),
    Output("weighing-info", "children"),
    Output("weighing-label", "value"),
    Output("weighing-speech", "value"),
    Input("choose-weighing", "n_clicks"),
    State("selection-store", "data"),
    State("cytoscape", "elements"),
    prevent_initial_call=True,
)
def open_weighing_dialog(_n, sel, elements):
    nodes = (sel or EMPTY_SELECTION).get("nodes", [])
    if len(nodes) != 2:
        return no_update, no_update, no_update, no_update, no_update
    a, b = find(elements, nodes[0]), find(elements, nodes[1])
    la = a["data"]["label"][:20] if a else nodes[0]
    lb = b["data"]["label"][:20] if b else nodes[1]
    default_speech = b["data"]["speech"] if b else (a["data"]["speech"] if a else "1AC")
    info = html.Span(f"Weighing “{la}” and “{lb}”")
    return POPUP_HIDDEN, MODAL_VISIBLE, info, "Weighing", default_speech


# ---------------------------------------------------------------------------
# Cancel buttons -> clear selection (which hides the modals + blanks inspector)
# ---------------------------------------------------------------------------

@app.callback(
    Output("selection-store", "data", allow_duplicate=True),
    Input("choose-cancel", "n_clicks"),
    Input("dialog-cancel", "n_clicks"),
    Input("weighing-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_modals(_c, _d, _w):
    return EMPTY_SELECTION


# ---------------------------------------------------------------------------
# Mutations -> rebuild cytoscape.elements
# ---------------------------------------------------------------------------

@app.callback(
    Output("cytoscape", "elements"),
    Output("apply-positions", "data"),
    Output("edge-msg", "children"),
    Output("nt-label", "value"),
    Output("edge-source", "value"),
    Output("edge-target", "value"),
    Output("selection-store", "data", allow_duplicate=True),
    Output("dialog-msg", "children", allow_duplicate=True),
    Output("edge-dialog", "style", allow_duplicate=True),
    Output("edit-edge-meta", "children", allow_duplicate=True),
    Output("upload-wrapper", "children", allow_duplicate=True),
    Input("add-node-btn", "n_clicks"),
    Input("add-edge-btn", "n_clicks"),
    Input("dialog-create", "n_clicks"),
    Input("weighing-create", "n_clicks"),
    Input("edit-node-save", "n_clicks"),
    Input("edit-node-delete", "n_clicks"),
    Input("edit-edge-save", "n_clicks"),
    Input("edit-edge-reverse", "n_clicks"),
    Input("edit-edge-delete", "n_clicks"),
    Input("relayout-btn", "n_clicks"),
    Input("clear-btn", "n_clicks"),
    Input("upload-json", "contents"),
    State("cytoscape", "elements"),
    State("selection-store", "data"),
    State("nt-type", "value"),
    State("nt-label", "value"),
    State("nt-speech", "value"),
    State("edge-source", "value"),
    State("edge-target", "value"),
    State("edge-type", "value"),
    State("edit-node-type", "value"),
    State("edit-node-label", "value"),
    State("edit-node-speech", "value"),
    State("edit-edge-type", "value"),
    State("dialog-edge-type", "value"),
    State("dialog-direction", "value"),
    State("weighing-label", "value"),
    State("weighing-speech", "value"),
    prevent_initial_call=True,
)
def mutate(_an, _ae, _dc, _wc, _ens, _end, _ees, _eer, _eed, _rl, _clr, upload_contents,
           elements, sel, nt_type, nt_label, nt_speech,
           edge_source, edge_target, edge_type,
           e_ntype, e_label, e_speech, e_etype, dlg_etype, dlg_dir,
           wgt_label, wgt_speech):
    trigger = ctx.triggered_id

    # `dcc.Upload.contents` is reset to None after each load (see below). That
    # reset re-fires this callback with empty contents -- ignore that echo so a
    # later upload of the *same* file still registers as a real change.
    if trigger == "upload-json" and not upload_contents:
        return (no_update,) * 11

    model = copy.deepcopy(model_elements(elements))
    sel = sel or EMPTY_SELECTION
    sel_nodes = sel.get("nodes", [])
    sel_edge = sel.get("edge")

    # defaults
    msg = ""
    nt_label_out = no_update
    edge_source_out = no_update
    edge_target_out = no_update
    selection_out = no_update
    dialog_msg = no_update
    dialog_style = no_update
    edge_meta_out = no_update
    upload_out = no_update

    def repack():
        return render_elements(model)

    if trigger == "add-node-btn":
        layer = NODE_LAYER[nt_type]
        label = (nt_label or "").strip() or nt_type
        node = make_node(next_id(model, "n"), label, nt_type, nt_speech,
                         derive_position(model, nt_speech, layer))
        model.append(node)
        nt_label_out = ""

    elif trigger == "add-edge-btn":
        if not edge_source or not edge_target:
            msg = "Pick a source and a target node."
        elif edge_source == edge_target:
            msg = "Source and target must differ."
        elif edge_exists_between(model, edge_source, edge_target):
            msg = "An edge already exists between these nodes."
        else:
            model.append(make_edge(next_id(model, "e"), edge_source, edge_target, edge_type))
            edge_source_out, edge_target_out = None, None

    elif trigger == "dialog-create":
        if len(sel_nodes) != 2:
            return (no_update,) * 11
        a, b = sel_nodes[0], sel_nodes[1]
        src, tgt = (a, b) if dlg_dir == "ab" else (b, a)
        if edge_exists_between(model, a, b):
            # keep the dialog open and report the violation
            return (no_update, no_update, no_update, no_update, no_update, no_update,
                    no_update, "An edge already exists between these nodes.", no_update,
                    no_update, no_update)
        model.append(make_edge(next_id(model, "e"), src, tgt, dlg_etype))
        selection_out = EMPTY_SELECTION
        dialog_msg, dialog_style = "", MODAL_HIDDEN

    elif trigger == "weighing-create":
        if len(sel_nodes) == 2:
            a, b = sel_nodes[0], sel_nodes[1]
            speech = wgt_speech or "1AC"
            label = (wgt_label or "").strip() or "Weighing"
            wid = next_id(model, "n")
            model.append(make_node(wid, label, "Weighing", speech,
                                   derive_position(model, speech, "framework")))
            # connect both selected nodes to the weighing node (comparison)
            model.append(make_edge(next_id(model, "e"), wid, a, "ComparisonEdge"))
            model.append(make_edge(next_id(model, "e"), wid, b, "ComparisonEdge"))
        selection_out = EMPTY_SELECTION

    elif trigger == "edit-node-save" and len(sel_nodes) == 1:
        n = next((x for x in model if x["data"]["id"] == sel_nodes[0]), None)
        if n:
            if e_ntype in NODE_LAYER:
                n["data"]["ntype"] = e_ntype
            n["data"]["label"] = (e_label or "").strip() or n["data"]["ntype"]
            n["data"]["speech"] = e_speech
            n["data"]["side"] = infer_side(e_speech)
            # Keep the node exactly where it is -- only properties change, not position.

    elif trigger == "edit-node-delete" and len(sel_nodes) == 1:
        nid = sel_nodes[0]
        model = [el for el in model if el["data"]["id"] != nid
                 and el["data"].get("source") != nid and el["data"].get("target") != nid]
        selection_out = EMPTY_SELECTION

    elif trigger == "edit-edge-save" and sel_edge:
        e = next((x for x in model if x["data"]["id"] == sel_edge), None)
        if e:
            e["data"]["etype"] = e_etype

    elif trigger == "edit-edge-reverse" and sel_edge:
        e = next((x for x in model if x["data"]["id"] == sel_edge), None)
        if e:
            e["data"]["source"], e["data"]["target"] = e["data"]["target"], e["data"]["source"]
            src = find(model, e["data"]["source"])
            tgt = find(model, e["data"]["target"])
            edge_meta_out = html.Span(
                f"{(src['data']['label'][:22] if src else '?')}  →  "
                f"{(tgt['data']['label'][:22] if tgt else '?')}")
        # keep the edge selected so the inspector stays open

    elif trigger == "edit-edge-delete" and sel_edge:
        model = [el for el in model if el["data"]["id"] != sel_edge]
        selection_out = EMPTY_SELECTION

    elif trigger == "relayout-btn":
        relayout_positions([el for el in model if is_node(el)])

    elif trigger == "clear-btn":
        model = []
        selection_out = EMPTY_SELECTION

    elif trigger == "upload-json" and upload_contents:
        # Remount the Upload (fresh <input>) so the same file can be re-uploaded.
        upload_out = make_upload()
        try:
            _, content_string = upload_contents.split(",", 1)
            raw = json.loads(base64.b64decode(content_string))
            # drop any stray background elements, then parse via the model schema
            raw["elements"] = [el for el in raw.get("elements", []) if not is_bg(el)]
            model = mser.elements_from_round(mser.from_dict(raw))
            ensure_positions(model)
            selection_out = EMPTY_SELECTION
        except Exception as exc:  # noqa: BLE001
            return (no_update, no_update, f"Could not load file: {exc}",
                    no_update, no_update, no_update, no_update, no_update, no_update,
                    no_update, make_upload())

    posmap = {n["data"]["id"]: n["position"] for n in model if is_node(n)}
    return (repack(), posmap, msg, nt_label_out, edge_source_out, edge_target_out,
            selection_out, dialog_msg, dialog_style, edge_meta_out, upload_out)


# ---------------------------------------------------------------------------
# Node dropdown options for the manual "Add edge" panel
# ---------------------------------------------------------------------------

@app.callback(
    Output("edge-source", "options"),
    Output("edge-target", "options"),
    Input("cytoscape", "elements"),
)
def edge_node_options(elements):
    opts = [{"label": f"{n['data']['label'][:28]}  ·  {n['data']['ntype']}",
             "value": n["data"]["id"]} for n in model_nodes(elements)]
    return opts, opts


# ---------------------------------------------------------------------------
# Save graph to JSON
# ---------------------------------------------------------------------------

def save_filename(name):
    """Turn the user's text into a safe, bare `<name>.json` filename."""
    name = (name or "").strip()
    name = name.replace("\\", "/").split("/")[-1]        # no directory components
    if name.lower().endswith(".json"):
        name = name[:-5]
    name = "".join(c for c in name if c.isalnum() or c in " -_.").strip()
    return f"{name or 'cdaf_graph'}.json"


@app.callback(
    Output("download-json", "data"),
    Input("save-btn", "n_clicks"),
    State("cytoscape", "elements"),
    State("save-name", "value"),
    prevent_initial_call=True,
)
def save_graph(_n, elements, name):
    rnd = mser.from_dict({"elements": model_elements(elements)})
    return dict(content=mser.dumps(rnd), filename=save_filename(name))


# ---------------------------------------------------------------------------
# Judge the current round  (verdict + RFD panel; click-only)
# ---------------------------------------------------------------------------

_NODE_ID_RE = re.compile(r"\bn\d+\b")


def label_display_map(round_elements):
    """id -> display label for the current round. Duplicate labels get the id
    appended so they disambiguate. App-side display only -- rfd.py stays pure
    and id-based; this substitution happens entirely here."""
    labels = {el["data"]["id"]: (el["data"].get("label") or el["data"]["id"])
              for el in round_elements if is_node(el)}
    freq = {}
    for lab in labels.values():
        freq[lab] = freq.get(lab, 0) + 1
    return {nid: (f"{lab} ({nid})" if freq[lab] > 1 else lab) for nid, lab in labels.items()}


def labelize(text, disp):
    """Swap node ids in the rendered RFD for human labels (display only)."""
    text = _NODE_ID_RE.sub(lambda m: disp.get(m.group(0), m.group(0)), text)
    return text.replace("chain:", "")


def verdict_panel(ballot, rfd_text):
    color = "#2E8B57" if ballot == "AFF" else "#C0392B"
    return html.Div([
        html.Div(f"{ballot} wins", style={
            "fontSize": "20px", "fontWeight": "bold", "color": color,
            "padding": "4px 0", "borderBottom": f"2px solid {color}", "marginBottom": "8px"}),
        html.Pre(rfd_text, style={
            "whiteSpace": "pre-wrap", "fontSize": "12px", "lineHeight": "1.45",
            "maxHeight": "340px", "overflowY": "auto", "margin": 0, "fontFamily": "inherit"}),
    ])


@app.callback(
    Output("judge-output", "children"),
    Input("judge-btn", "n_clicks"),
    State("cytoscape", "elements"),
    prevent_initial_call=True,
)
def judge_round(_n, elements):
    """Read the CURRENT graph once and judge it. Fires ONLY on button click
    (`elements` is a State, not an Input) -- it never re-judges on edits. The
    conversion is the SAME serialize path that Save uses (elements -> Round), so
    judging the live graph is identical to saving it and judging that JSON."""
    try:
        round_elements = model_elements(elements)
        rnd = mser.from_dict({"elements": round_elements})
        ballot, trace = run_judge(rnd)
        rfd_text = labelize(judge_rfd.render(ballot, trace),
                            label_display_map(round_elements))
        return verdict_panel(ballot, rfd_text)
    except Exception as exc:  # noqa: BLE001  -- never crash the app on a bad graph
        return html.Div(f"Could not judge this graph: {exc}",
                        style={"color": "#C0392B", "fontSize": "12px",
                               "whiteSpace": "pre-wrap"})


if __name__ == "__main__":
    app.run(debug=True, port=8050)
