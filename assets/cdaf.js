/* CDAF Builder -- clientside cytoscape enhancements.
 *
 * dash-cytoscape doesn't expose drag/pan events or the cy instance to Dash
 * callbacks, so we reach the live cytoscape instance via its container's
 * `_cyreg` registry and wire up the behaviors Dash can't:
 *   - custom 2-node selection (max two, FIFO) for edge / weighing creation
 *   - clamp node dragging inside its speech column
 *   - clamp panning so you can't scroll past the 1AC / 2AR column edges
 *   - float the column header labels at the top of the viewport
 *
 * The handler attachment is driven by a self-contained poller (below), NOT a
 * Dash clientside callback -- that avoids a startup race where Dash could call
 * into this namespace before the asset has registered it.
 *
 * Geometry constants MUST match the Python side (see cdaf_app.py).
 */

window.dash_clientside = window.dash_clientside || {};

const CDAF = {
    LEFT: 120,
    COL_W: 280,        // columns are adjacent (no gap)
    NODE_W: 170,
    WORLD_TOP: 20,     // panning cannot go above this graph-top
    SPEECHES: ["1AC", "1NC", "2AC", "2NC/1NR", "1AR", "2NR", "2AR"],
};
CDAF.bandCenter = function (c) { return CDAF.LEFT + c * CDAF.COL_W + CDAF.COL_W / 2; };
CDAF.colIndex = function (sp) { const i = CDAF.SPEECHES.indexOf(sp); return i < 0 ? 0 : i; };
CDAF.worldX = function () {
    return {
        left: CDAF.LEFT,
        right: CDAF.LEFT + CDAF.SPEECHES.length * CDAF.COL_W,
    };
};

function cdafGetCy() {
    const root = document.getElementById("cytoscape");
    if (!root) return null;
    const stack = [root];
    while (stack.length) {
        const n = stack.pop();
        if (n && n._cyreg && n._cyreg.cy) return n._cyreg.cy;
        if (n && n.children) for (let i = 0; i < n.children.length; i++) stack.push(n.children[i]);
    }
    return null;
}

function cdafClampNodeX(node) {
    const sp = node.data("speech");
    if (sp === undefined || sp === null) return;
    const center = CDAF.bandCenter(CDAF.colIndex(sp));
    const half = (CDAF.COL_W - CDAF.NODE_W) / 2;
    const x = node.position("x");
    if (x < center - half) node.position("x", center - half);
    else if (x > center + half) node.position("x", center + half);
}

function cdafAttach(cy) {
    cy.__cdafInit = true;
    cy.__sel = [];

    const setSel = function (nodes, edge) {
        cy.__sel = nodes.slice();
        if (window.dash_clientside && window.dash_clientside.set_props) {
            window.dash_clientside.set_props("selection-store", {
                data: { nodes: nodes, edge: edge || null },
            });
        }
    };

    // selection: plain click = single; cmd/ctrl+click = add 2nd;
    // a click while two are selected replaces the oldest (FIFO).
    cy.on("tap", "node", function (evt) {
        const t = evt.target;
        if (t.data("bg") || t.data("hdr")) return;
        const id = t.id();
        const cmd = evt.originalEvent && (evt.originalEvent.metaKey || evt.originalEvent.ctrlKey);
        let sel = (cy.__sel || []).slice();
        if (sel.length >= 2) sel = [sel[1], id];
        else if (cmd) { if (sel.indexOf(id) < 0) sel.push(id); }
        else sel = [id];
        sel = sel.filter(function (v, i) { return sel.indexOf(v) === i; }).slice(-2);
        setSel(sel, null);
    });

    cy.on("tap", "edge", function (evt) { setSel([], evt.target.id()); });
    cy.on("tap", function (evt) { if (evt.target === cy) setSel([], null); });

    // drag stays inside the node's speech column
    cy.on("drag", "node", function (evt) {
        const t = evt.target;
        if (t.data("bg") || t.data("hdr")) return;
        cdafClampNodeX(t);
    });

    // zoom/pan clamps + sticky HTML headers
    let guard = false;
    const refresh = function () {
        if (guard) return;
        guard = true;
        const wx = CDAF.worldX();
        const worldW = wx.right - wx.left;
        const w = cy.width();
        // can't zoom out far enough to see past the left/right column edges
        const minZ = w > 0 ? w / worldW : 0.25;
        if (Math.abs(cy.minZoom() - minZ) > 1e-4) cy.minZoom(minZ);
        let z = cy.zoom();
        if (z < minZ) { cy.zoom(minZ); z = cy.zoom(); }
        const pan = cy.pan();
        // horizontal: keep within [1AC left edge, 2AR right edge]
        const lower = w - wx.right * z;
        const upper = -wx.left * z;
        let px = pan.x;
        if (lower > upper) px = (lower + upper) / 2;
        else px = Math.min(Math.max(px, lower), upper);
        // vertical: cannot pan above the top of the graph (downward is unbounded)
        const topLimit = -CDAF.WORLD_TOP * z;
        let py = pan.y > topLimit ? topLimit : pan.y;
        if (Math.abs(px - pan.x) > 0.5 || Math.abs(py - pan.y) > 0.5) cy.pan({ x: px, y: py });
        // sticky headers: follow columns horizontally; never move vertically
        const z2 = cy.zoom(), p2 = cy.pan();
        for (let c = 0; c < CDAF.SPEECHES.length; c++) {
            const el = document.getElementById("cdaf-hdr-" + c);
            if (el) el.style.left = (z2 * CDAF.bandCenter(c) + p2.x) + "px";
        }
        guard = false;
    };
    cy.on("viewport add remove", refresh);
    refresh();
}

// Self-contained poller: attach as soon as the cy instance exists. Re-attaches
// if cytoscape is ever recreated. No dependency on Dash callback timing.
(function () {
    const tick = function () {
        const cy = cdafGetCy();
        if (cy && !cy.__cdafInit) cdafAttach(cy);
    };
    setInterval(tick, 400);
    if (document.readyState !== "loading") tick();
    else document.addEventListener("DOMContentLoaded", tick);
})();

window.dash_clientside.cdaf = {

    // Apply node positions from Dash (relayout / speech-move / add) onto cy.
    // dash-cytoscape doesn't re-apply the `layout` prop on change, so we set
    // positions directly to guarantee position-only updates take effect.
    applyPositions: function (posmap) {
        const cy = cdafGetCy();
        if (!cy || !posmap) return window.dash_clientside.no_update;
        cy.batch(function () {
            Object.keys(posmap).forEach(function (id) {
                const n = cy.getElementById(id);
                if (n && n.length && !n.data("bg") && !n.data("hdr")) n.position(posmap[id]);
            });
        });
        return "";
    },

    // Reflect the Dash selection-store into the cytoscape selection highlight.
    applySelection: function (sel) {
        const cy = cdafGetCy();
        if (!cy) return window.dash_clientside.no_update;
        const nodes = (sel && sel.nodes) || [];
        cy.batch(function () {
            cy.elements().unselect();
            nodes.forEach(function (id) { const n = cy.getElementById(id); if (n) n.select(); });
            if (sel && sel.edge) { const e = cy.getElementById(sel.edge); if (e) e.select(); }
        });
        cy.__sel = nodes.slice();
        return "";
    },
};
