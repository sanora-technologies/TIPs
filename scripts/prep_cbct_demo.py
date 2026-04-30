"""
Build public/data/cbct-demo.json for the landing page 3D globe.
Uses trimesh quadric decimation (proper QEM — no z-fighting artifacts).
"""
import json, pathlib, numpy as np
import trimesh

SRC_JSON = pathlib.Path(__file__).parent.parent / "output/24/result_cache.json"
OUT_JSON = pathlib.Path("/home/oaiz/Documents/sanora/faawebfrontend/public/data/cbct-demo.json")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# ── QEM decimation ────────────────────────────────────────────────────────────

def qem_decimate(vertices: list, faces: list, target_faces: int):
    """Quadric error metric decimation via trimesh. Pass-through if already at target."""
    v = np.array(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.array(faces,    dtype=np.int32).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    n = len(mesh.faces)
    if n <= target_faces:
        return mesh.vertices.tolist(), mesh.faces.flatten().tolist()
    mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    return mesh.vertices.tolist(), mesh.faces.flatten().tolist()

# ── coordinate remapping ──────────────────────────────────────────────────────

def remap_coords(vertices: list):
    """CBCT(x,y,z) → Three.js(x, z, -y)  so Z-superior becomes Y-up."""
    v = np.array(vertices, dtype=np.float64).reshape(-1, 3)
    out = np.stack([v[:,0], v[:,2], -v[:,1]], axis=1)
    return out.flatten().tolist()

def r1(verts: list):
    return [round(x, 1) for x in verts]

# ── load ──────────────────────────────────────────────────────────────────────

print(f"Loading {SRC_JSON} …")
data = json.loads(SRC_JSON.read_text())

# Compute scene center from structures
struct_verts_all = []
for m in data["structures"]["meshes"]:
    struct_verts_all.extend(remap_coords(m["vertices"]))

sv = np.array(struct_verts_all).reshape(-1, 3)
cx, cy, cz = (sv.min(0) + sv.max(0)) / 2
print(f"Scene center (Three.js): ({cx:.2f}, {cy:.2f}, {cz:.2f})")

def center(verts: list):
    v = np.array(verts).reshape(-1, 3)
    v[:,0] -= cx; v[:,1] -= cy; v[:,2] -= cz
    return r1(v.flatten().tolist())

def process(mesh_list, target, skip_labels=None):
    out = []
    for m in mesh_list:
        lbl = m["label"]
        if skip_labels and lbl in skip_labels:
            continue
        print(f"  [{m.get('name','?')}] {len(m['faces'])//3:,} → ", end="", flush=True)
        v3 = remap_coords(m["vertices"])
        v_dec, f_dec = qem_decimate(v3, m["faces"], target)
        v_dec = center(v_dec)
        print(f"{len(f_dec)//3:,} tris")
        entry = {k: m[k] for k in ("label","name","color","opacity") if k in m}
        entry["vertices"] = v_dec
        entry["faces"]    = f_dec
        out.append(entry)
    return out

STRUCT_TARGETS = {1: 8000, 2: 10000, 5: 2500}

print("\n── structures ──")
struct_meshes = []
for m in data["structures"]["meshes"]:
    lbl    = m["label"]
    target = STRUCT_TARGETS.get(lbl, 3000)
    print(f"  [{m.get('name','?')}] {len(m['faces'])//3:,} → ", end="", flush=True)
    v3 = remap_coords(m["vertices"])
    vd, fd = qem_decimate(v3, m["faces"], target)
    vd = center(vd)
    print(f"{len(fd)//3:,} tris")
    entry = {k: m[k] for k in ("label","name","color","opacity") if k in m}
    entry["vertices"] = vd
    entry["faces"]    = fd
    struct_meshes.append(entry)

print("\n── toothseg (individual teeth only, QEM 4000) ──")
teeth_meshes = process(
    data["toothseg"]["meshes"], target=4000,
    skip_labels={11,12,13,14,15},
)

print("\n── pulp (QEM 1200) ──")
pulp_meshes = process(data["pulp"]["meshes"], target=1200)

# ── assemble ──────────────────────────────────────────────────────────────────

payload = {
    "case": data.get("case", "24"),
    "toothseg":   {"base": "24", "layer": "toothseg",   "meshes": teeth_meshes},
    "pulp":       {"base": "24", "layer": "pulp",       "meshes": pulp_meshes},
    "structures": {"base": "24", "layer": "structures", "meshes": struct_meshes},
    "perio":      data.get("perio", {"teeth": [], "summary": {}}),
}

out_str = json.dumps(payload, separators=(',', ':'))
OUT_JSON.write_text(out_str)

total = sum(len(m["faces"])//3 for m in struct_meshes + teeth_meshes + pulp_meshes)
print(f"\nDone → {OUT_JSON}")
print(f"  Triangles : {total:,}")
print(f"  File size : {len(out_str)/1e6:.2f} MB")
