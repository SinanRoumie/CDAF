"""E1 gate -- model liveness field, serializer v2, and the §5 converter.

Pure model/ work: no dash/plotly/flask; the converter is a lossless v1 -> v2
re-encoding of the ExtensionEdge structure into per-node liveness.
"""

import os

from model import (
    Round, Link, Advocacy, BallotDirective,
    Support, DefensiveAttack, OffensiveAttack,
    CONTESTED, CONCEDED, SCHEMA_VERSION, convert, serialize,
)


def _no_extension_edges(rnd):
    return not any(e.etype == "ExtensionEdge" for e in rnd.edges)

ORACLE = os.path.join(os.path.dirname(__file__), "oracle", "NSDA24Finals.json")


# --- model field --------------------------------------------------------------

def test_node_liveness_always_includes_intro_speech():
    n = Link(id="n1", label="x", side="AFF", speech="1AC")
    assert n.speech == "1AC"
    assert "1AC" in n.liveness            # introduction stamped automatically


def test_liveness_is_ordered_by_speech_order():
    n = Link(id="n1", label="x", side="AFF", speech="1AC",
             liveness={"2AR": CONCEDED, "1AC": CONTESTED, "2AC": CONCEDED})
    assert list(n.liveness.keys()) == ["1AC", "2AC", "2AR"]   # not insertion order


def test_schema_version_is_2():
    assert SCHEMA_VERSION == 2


# --- serializer round-trip ----------------------------------------------------

def test_v2_round_trips_byte_stable():
    rnd = Round(elements=[
        Link(id="n1", label="L", side="AFF", speech="1AC",
             liveness={"1AC": CONTESTED, "2AC": CONCEDED}),
        BallotDirective(id="bd", label="B", side="AFF", speech="2AR"),
        Support(id="e1", source="n1", target="bd"),
    ])
    text1 = serialize.dumps(rnd)
    reloaded = serialize.loads(text1)     # already v2 -> not re-converted
    text2 = serialize.dumps(reloaded)
    assert text1 == text2
    assert reloaded.version == 2
    assert '"liveness"' in text1          # liveness is written


# --- the §5 converter ---------------------------------------------------------

def _v1_dict():
    """AFF link introduced 1AC, extended to 2AC (duplicate + ExtensionEdge);
    NEG reads a defensive attack on the 1AC instance in 1NC. The 2AC instance is
    unattacked. Edge to the BD is drawn from the 2AC duplicate."""
    return {
        "version": 1,
        "elements": [
            {"data": {"id": "n1", "label": "L", "ntype": "Link", "side": "AFF", "speech": "1AC"},
             "position": {"x": 1, "y": 2}},
            {"data": {"id": "n2", "label": "L", "ntype": "Link", "side": "AFF", "speech": "2AC"}},
            {"data": {"id": "e1", "source": "n1", "target": "n2", "etype": "ExtensionEdge"}},
            {"data": {"id": "d1", "label": "D", "ntype": "Link", "side": "NEG", "speech": "1NC"}},
            {"data": {"id": "e2", "source": "d1", "target": "n1", "etype": "DefensiveAttackEdge"}},
            {"data": {"id": "bd", "label": "B", "ntype": "BallotDirective", "side": "AFF", "speech": "2AR"}},
            {"data": {"id": "e3", "source": "n2", "target": "bd", "etype": "SupportEdge"}},
        ],
    }


def test_v1_load_converts_to_v2():
    rnd = serialize.from_dict(_v1_dict())
    assert rnd.version == 2
    # duplicate collapsed: 4 nodes -> 3; extension edge gone: 3 edges -> 2
    assert len(rnd.nodes) == 3
    assert len(rnd.edges) == 2
    assert _no_extension_edges(rnd)


def test_converter_collapses_and_infers_liveness():
    rnd = serialize.from_dict(_v1_dict())
    by_id = {n.id: n for n in rnd.nodes}
    # survivor keeps the introduction id/speech (earliest = 1AC), duplicate gone
    assert "n1" in by_id and "n2" not in by_id
    link = by_id["n1"]
    assert link.speech == "1AC"
    # 1AC instance was defensively attacked -> contested; 2AC instance was not.
    assert link.liveness == {"1AC": CONTESTED, "2AC": CONCEDED}
    assert link.position is not None and link.position.x == 1   # rep position kept
    # the NEG attacker was never attacked back -> conceded
    assert by_id["d1"].liveness == {"1NC": CONCEDED}


def test_converter_rewires_edges_to_survivor():
    rnd = serialize.from_dict(_v1_dict())
    support = next(e for e in rnd.edges if e.etype == "SupportEdge")
    # was n2 -> bd; n2 collapsed into n1, so it must now read n1 -> bd
    assert support.source == "n1" and support.target == "bd"


def test_versionless_file_is_v1_and_converts():
    d = _v1_dict()
    del d["version"]                       # pre-version file
    rnd = serialize.from_dict(d)
    assert rnd.version == 2                 # treated as v1, then converted
    assert _no_extension_edges(rnd)


def test_converter_is_deterministic():
    a = serialize.dumps(serialize.from_dict(_v1_dict()))
    b = serialize.dumps(serialize.from_dict(_v1_dict()))
    assert a == b


def test_converter_operates_on_raw_dicts_and_does_not_mutate_input():
    # convert() runs at the raw-element-dict level (pre-parse), so ExtensionEdge
    # never needs to exist as a class.
    raw = [
        {"data": {"id": "n1", "label": "L", "ntype": "Link", "side": "AFF", "speech": "1AC"}},
        {"data": {"id": "n2", "label": "L", "ntype": "Link", "side": "AFF", "speech": "2AC"}},
        {"data": {"id": "e1", "source": "n1", "target": "n2", "etype": "ExtensionEdge"}},
    ]
    out = convert(raw)
    assert len(raw) == 3                                  # original untouched
    out_nodes = [el for el in out if "source" not in el["data"]]
    out_edges = [el for el in out if "source" in el["data"]]
    assert len(out_nodes) == 1                            # duplicates collapsed
    assert not any(el["data"].get("etype") == "ExtensionEdge" for el in out_edges)
    assert out_nodes[0]["data"]["liveness"] == {"1AC": CONCEDED, "2AC": CONCEDED}


# --- Step 0: liveness survives the judge/save path (version tag) --------------

def test_authored_liveness_survives_the_v2_from_dict_path():
    # The app's judge/save path calls from_dict with version=SCHEMA_VERSION so it
    # does NOT re-run the v1 converter and wipe authored liveness. A round with
    # hand-authored liveness must come back with that liveness intact.
    authored = {"1AC": CONTESTED, "2AC": CONCEDED, "1AR": CONCEDED, "2AR": CONCEDED}
    elements = [
        {"data": {"id": "n1", "label": "L", "ntype": "Link", "side": "AFF",
                  "speech": "1AC", "liveness": dict(authored)}},
    ]
    rnd = serialize.from_dict({"version": SCHEMA_VERSION, "elements": elements})
    assert [n for n in rnd.nodes][0].liveness == authored     # intact

    # Contrast: without a version tag it defaults to v1 and the converter rebuilds
    # liveness from the (absent) ExtensionEdge structure -> only the intro speech.
    wiped = serialize.from_dict({"elements": elements})
    assert [n for n in wiped.nodes][0].liveness == {"1AC": CONCEDED}


# --- real round ---------------------------------------------------------------

def test_nsda_round_converts_shrinks_and_restabilizes():
    import json
    raw = json.load(open(ORACLE))
    before_nodes = sum(1 for e in raw["elements"] if "source" not in e["data"])

    rnd = serialize.load(ORACLE)           # v1 -> auto-convert -> v2
    assert rnd.version == 2
    assert len(rnd.nodes) < before_nodes                       # duplicates collapsed
    assert _no_extension_edges(rnd)   # no ExtensionEdges
    # every node carries liveness including its intro speech
    for n in rnd.nodes:
        assert n.liveness and n.speech in n.liveness
    # re-saving the v2 round and reloading is byte-stable (no second conversion)
    text = serialize.dumps(rnd)
    assert serialize.dumps(serialize.loads(text)) == text
