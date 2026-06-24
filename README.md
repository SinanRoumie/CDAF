# CDAF Graph Builder

An interactive local web app for constructing and visualizing **CDAF graphs** — a
node-and-edge representation of a competitive policy debate round. Built with Python,
[Dash](https://dash.plotly.com/), and [dash-cytoscape](https://dash.plotly.com/cytoscape).

It is a working tool, not a mockup: you build graphs by hand, edit them, lay them out,
and save/load them as JSON.

## Spatial organization

The round is laid out as a 2.5D / isometric scene:

- **X axis — speech progression.** One adjacent column per speech (no gaps), left → right,
  in the fixed, immutable order `1AC, 1NC, 2AC, 2NC/1NR, 1AR, 2NR, 2AR` (`2NC/1NR` is one
  combined speech). Columns are lightly tinted by side. **Sticky headers** stay pinned to
  the top of the screen at all times, tracking their column horizontally. Panning/zoom is
  bounded so you can't see past the 1AC left edge, the 2AR right edge, or above the top of
  the graph; vertical scrolling downward is unbounded (columns are effectively infinite).
- **Y axis — vertical stacking.** The first node in a speech sits at the top of its column;
  each later node stacks directly below. Dragging a node is constrained to its own column.
- **Z axis — layer depth,** rendered as a horizontal isometric offset (plus shape):
  **content (front) → framework (middle) → ballot (back).**

**Side** is shown by **node border color: green for AFF, red for NEG.**

A node's position is derived automatically from its speech (column) and type (layer/depth).
The view (zoom/pan and your dragged positions) is preserved when you add or edit nodes/edges.

## Domain model

**Three ordered layers** (front → back) and their node types:

| Layer | Node types |
|-----------|------------|
| content   | Uniqueness, Link, Impact, Advocacy |
| framework | Framework, Weighing |
| ballot    | BallotDirective |

Every node carries exactly three neutral fields:

- a **text label / claim**
- the **speech** it was introduced in — chosen from the seven fixed speeches above
- a **side** — `AFF` or `NEG`, **inferred automatically from the speech**
  (1AC/2AC/1AR/2AR → AFF; 1NC/2NC·1NR/2NR → NEG)

**Node type** is shown by **color** (one per type) and **shape** (one per layer:
content = rounded rectangle, framework = hexagon, ballot = diamond).
**Side** is shown by **node border color** (AFF = green, NEG = red).

**Five edge types**, each visually distinct by color, line style, and arrowhead:

| Edge type | Color | Line | Arrow |
|-----------|-------|------|-------|
| SupportEdge         | teal   | solid  | → |
| ExtensionEdge       | blue   | dashed | → |
| DefensiveAttackEdge | orange | dotted | ⊣ (tee) |
| OffensiveAttackEdge | red    | solid (thick) | → |
| ComparisonEdge      | purple | dashed | ◆◆ (both ends, non-directional) |

Legends for both node types and edge types are shown in the right-hand panel.

## Setup

Requires Python 3.9+.

```bash
cd /Users/sinan/Desktop/CDAF

# create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Run

```bash
python cdaf_app.py
```

Then open **http://127.0.0.1:8050** in your browser.

## Using the app

- **Add a node** — left panel: pick a type, write the claim, pick the speech (the side is
  shown as it will be inferred), then *+ Add node*. It appears in the right speech column
  and layer depth, and the claim field clears so you can enter the next one.
- **Select nodes** — click a node to select it. **Hold ⌘ and click a second node** to
  select two at once (max two; clicking a third drops the oldest). The two stay selected
  until you choose an action.
- **Create an edge** — two ways:
  - with two nodes selected, a floating card offers **Create Edge** → a dialog lets you
    pick the **edge type** and **direction** (which node is source/target); or
  - use the left panel: pick a source, target, and edge type, then *+ Add edge*.
  Two nodes can have **at most one edge between them** (either direction); a second is
  rejected with *"An edge already exists between these nodes."*
- **Create a Weighing node** — with two nodes selected, the card's **Create Weighing Node**
  opens a dialog for the node's **label** and **speech**; confirming drops a Weighing node
  (framework layer) connected to both selected nodes with ComparisonEdges, then clears the
  selection.
- **Inspect / edit** — the Inspector is blank until you select something. A selected node
  or edge shows its details for editing (claim/speech, or edge type), then *Save changes*
  or *Delete*. Deleting a node also removes its connected edges.
- **Layout** — positions are derived from each node's speech + layer. Drag a node to
  reposition it (kept inside its column); press **Re-run layout** to re-derive the tidy
  isometric arrangement. Adding/editing never resets your zoom, pan, or dragged positions.
- **Save / load** — *Save JSON* downloads the current graph as `cdaf_graph.json`;
  *Load JSON (drag or click)* restores a saved graph so your work persists between sessions.
- **Clear graph** — removes all nodes and edges.

## Files

```
cdaf_app.py        # the entire app: domain model, stylesheet, layout, callbacks
assets/style.css   # styling (auto-served by Dash)
requirements.txt   # dependencies
```

## Saved-file format

A saved graph is a JSON document:

```json
{
  "elements": [
    {"data": {"id": "n1", "label": "…", "ntype": "Link",
              "side": "AFF", "speech": "1AC"}, "position": {"x": 210, "y": 640}},
    {"data": {"id": "e2", "source": "n1", "target": "n3", "etype": "SupportEdge"}}
  ],
  "counter": 4
}
```

This is the same shape Dash Cytoscape consumes, so the file is both the save format
and the live graph state.
