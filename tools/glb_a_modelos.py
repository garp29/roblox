"""Convierte los modelos .glb de bloques con animaciones (exportados con three.js) en:

  - src/server/Util/ModelosZona1Data.luau: huesos (articulaciones) y bloques de cada modelo,
    para que ModelosZona1 los arme en el servidor.
  - src/shared/ClipsZona1.luau: las animaciones (Idle, Walk, Attack, Hit, Death) como
    cambios de cada hueso respecto a su pose de reposo; EnemyAnimator las reproduce en
    el cliente.

Uso:
    python tools/glb_a_modelos.py --data src/server/Util/ModelosZona1Data.luau \
        --clips src/shared/ClipsZona1.luau \
        Arbolito=assets/modelos/arbolito.glb ArbolGuardian=assets/modelos/arbol_guardian.glb \
        GiganteDelBosque=assets/modelos/gigante_del_bosque.glb

Requiere numpy. Cada malla del .glb debe ser una caja (24 vértices, centrada). Los huesos son
los nodos que mueven las animaciones (rig, hip_L, hip_R, spine, arm_*_shoulder,
arm_*_joint1, crown); cada bloque va con el hueso animado más cercano por encima de él. Se
gira 180° en Y (frente hacia -Z, como los enemigos del juego) y se quitan los bloques
tapados por otros del mismo hueso.
"""
import json, struct, sys
import numpy as np

ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
STEP, OFF = 0.002, 1024  # bloques: 2 caracteres por valor
PSTEP, POFF = 0.0002, 131072  # animación (posición): 3 caracteres por valor
FPS = {"Idle": 10, "Walk": 20, "Death": 15}  # fotogramas por segundo de cada clip (30 si no está)
LOOPS = {"Idle", "Walk"}
FLIP = np.diag([-1.0, 1.0, -1.0])  # 180° en Y


def lin2srgb(c):
    return [round(255 * (12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055)) for x in c[:3]]


CT = {5126: ("f", 4), 5123: ("H", 2), 5125: ("I", 4), 5121: ("B", 1)}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_glb(fn):
    data = open(fn, "rb").read()
    off, chunks = 12, []
    while off < len(data):
        clen, _ = struct.unpack_from("<II", data, off)
        off += 8
        chunks.append(data[off:off + clen])
        off += clen
    return json.loads(chunks[0]), (chunks[1] if len(chunks) > 1 else b"")


def accessor(js, binc, i):
    a = js["accessors"][i]
    bv = js["bufferViews"][a["bufferView"]]
    fmt, size = CT[a["componentType"]]
    n = NC[a["type"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", size * n)
    return np.array([struct.unpack_from("<" + fmt * n, binc, start + k * stride) for k in range(a["count"])], dtype=float)


def q2m(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def m2q(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q) / np.linalg.norm(q)
    return -q if q[3] < 0 else q


def local_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], dtype=float).reshape(4, 4).T
    M = np.eye(4)
    M[:3, :3] = q2m(n.get("rotation", [0, 0, 0, 1])) * np.array(n.get("scale", [1, 1, 1]))
    M[:3, 3] = n.get("translation", [0, 0, 0])
    return M


def to_roblox(M):
    """Matriz 4x4 del .glb -> (posición, rotación) girada 180° en Y."""
    return FLIP @ M[:3, 3], FLIP @ M[:3, :3] @ FLIP


def load(fn):
    js, binc = read_glb(fn)
    nodes = js["nodes"]
    animated = sorted({ch["target"]["node"] for an in js.get("animations", []) for ch in an["channels"]})
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i
    joint_of_node = {}

    def owner(i):
        while i is not None and i not in animated:
            i = parent.get(i)
        return i

    joints = []  # (nodo, nombre, índice del hueso padre o -1, matriz de reposo en el mundo)
    world = {}
    boxes = []

    def walk(i, W):
        n = nodes[i]
        Wi = W @ local_matrix(n)
        world[i] = Wi
        if i in animated:
            p = owner(parent.get(i))
            joint_of_node[i] = len(joints)
            joints.append(dict(node=i, name=n.get("name", f"joint{i}"), parent=joint_of_node.get(p, -1), W=Wi))
        if "mesh" in n:
            prim = js["meshes"][n["mesh"]]["primitives"][0]
            a = js["accessors"][prim["attributes"]["POSITION"]]
            mat = js["materials"][prim["material"]]
            emissive = mat.get("emissiveFactor") or [0, 0, 0]
            center = (np.array(a["max"]) + np.array(a["min"])) / 2
            Wc = Wi.copy()
            Wc[:3, 3] = (Wi @ np.append(center, 1))[:3]
            o = owner(i)
            boxes.append(dict(name=n.get("name", ""), joint=joint_of_node[o] if o is not None else 0, W=Wc,
                              size=np.array(a["max"]) - np.array(a["min"]), mat=mat["name"],
                              color=tuple(lin2srgb(mat["pbrMetallicRoughness"]["baseColorFactor"])),
                              glow=max(emissive) > 0))
        for c in n.get("children", []):
            walk(c, Wi)

    for r in js["scenes"][js.get("scene", 0)]["nodes"]:
        walk(r, np.eye(4))
    for b in boxes:
        b["t"], b["R"] = to_roblox(b["W"])
    return js, binc, joints, boxes


def cull(boxes, n=6):
    """Quita los bloques cuya superficie queda entera dentro de otros del mismo hueso."""
    keep = [True] * len(boxes)
    groups = {}
    for i, b in enumerate(boxes):
        groups.setdefault(b["joint"], []).append(i)
    lin = np.linspace(-1, 1, n)
    for idxs in groups.values():
        Rs = np.array([boxes[i]["R"] for i in idxs])
        ts = np.array([boxes[i]["t"] for i in idxs])
        hs = np.array([boxes[i]["size"] / 2 for i in idxs])
        for a, i in enumerate(idxs):
            b = boxes[i]
            if b["name"].startswith("eye_") or b["glow"]:
                continue
            h = b["size"] / 2
            pts = []
            for ax in range(3):
                o = [x for x in range(3) if x != ax]
                for s in (-1, 1):
                    for u in lin:
                        for v in lin:
                            p = np.zeros(3)
                            p[ax] = s * h[ax]
                            p[o[0]] = u * h[o[0]]
                            p[o[1]] = v * h[o[1]]
                            pts.append(p)
            P = (b["R"] @ np.array(pts).T).T + b["t"]
            covered = np.zeros(len(P), bool)
            for c in range(len(idxs)):
                if c == a or not keep[idxs[c]]:
                    continue
                L = (Rs[c].T @ (P - ts[c]).T).T
                covered |= np.all(np.abs(L) < hs[c] - 1e-4, axis=1)
                if covered.all():
                    break
            if covered.all():
                keep[i] = False
    return [b for b, k in zip(boxes, keep) if k]


def enc(v, chars=2):
    assert 0 <= v < 64 ** chars, v
    out = ""
    for _ in range(chars):
        out = ALPH[v % 64] + out
        v //= 64
    return out


def fmt(v):
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def wrap(s, width=110):
    return [s[i:i + width] for i in range(0, len(s), width)] or [""]


# Datos del servidor --------------------------------------------------------------------

def emit_data(key, fn):
    js, binc, joints, boxes = load(fn)
    boxes = cull(boxes)
    mats = []
    for b in boxes:
        m = (b["mat"], b["color"], b["glow"])
        if m not in mats:
            mats.append(m)
    assert len(mats) <= 64 and len(joints) <= 64
    recs, eyes = [], []
    for idx, b in enumerate(boxes):
        r = ALPH[b["joint"]] + ALPH[mats.index((b["mat"], b["color"], b["glow"]))]
        r += "".join(enc(int(round(x / STEP)) + OFF) for x in list(b["t"]) + list(b["size"]))
        if np.allclose(b["R"], np.eye(3), atol=1e-5):
            r += "."
        else:
            r += "q" + "".join(enc(int(round((c + 1) * 2047))) for c in m2q(b["R"]))
        recs.append(r)
        if b["name"].startswith("eye_"):
            eyes.append(idx + 1)
    out = [f"\t{key} = {{", f"\t\tSource = \"{fn.split('/')[-1]}\",", "\t\t-- Huesos: nombre, hueso padre (0 = la raíz del enemigo) y pose de reposo (x y z, cuaternión)", "\t\tJoints = {"]
    for j in joints:
        p, R = to_roblox(j["W"])
        q = m2q(R)
        out.append(f"\t\t\t{{ \"{j['name']}\", {j['parent'] + 1}, {{ {', '.join(fmt(v) for v in p)}, {', '.join(fmt(v) for v in q)} }} }},")
    out += ["\t\t},", "\t\tPalette = {"]
    for (n, c, g) in mats:
        out.append(f"\t\t\t{{ \"{n}\", Color3.fromRGB({c[0]}, {c[1]}, {c[2]}), {'true' if g else 'false'} }},")
    out += ["\t\t},", f"\t\tEyes = {{ {', '.join(str(e) for e in eyes)} }},", f"\t\tCount = {len(recs)},", "\t\tBoxes = table.concat({"]
    lines, cur = [], ""
    for r in recs:
        if len(cur) + len(r) > 110:
            lines.append(cur)
            cur = ""
        cur += r
    lines.append(cur)
    out += [f"\t\t\t\"{l}\"," for l in lines]
    out += ["\t\t}),", "\t},"]
    return "\n".join(out)


DATA_HEADER = """-- Datos de los 3 modelos 3D de la Zona 1 (Bosque), sacados de los .glb de Santiago
-- (arbolito.glb, arbol_guardian.glb, gigante_del_bosque.glb), ya girados al estilo del juego
-- (frente hacia -Z, suelo en Y = 0) y en la unidad del .glb; ModelosZona1 los escala a studs.
-- Joints: los huesos que mueven las animaciones (ClipsZona1), con su padre y su pose de
-- reposo en el modelo. Cada bloque va soldado a un hueso.
-- Boxes va comprimido: un registro por bloque =
--   hueso (1 carácter, base 64, desde 0) + color (1 carácter, índice de Palette desde 0)
--   + x y z ancho alto fondo (2 caracteres cada uno, base 64, paso 0.002, desplazados 1024)
--   + "." (sin giro) o "q" y el cuaternión x y z w (2 caracteres cada uno, de -1 a 1).
-- Eyes: posiciones (desde 1) de los bloques que son ojos.
-- Generado automáticamente desde los .glb con tools/glb_a_modelos.py (no editar a mano).

return {
"""


# Animaciones (cliente) -------------------------------------------------------------------

def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def qinv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def sample(times, values, t, rot):
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    k = np.searchsorted(times, t) - 1
    a = (t - times[k]) / (times[k + 1] - times[k])
    v0, v1 = values[k], values[k + 1]
    if rot:
        if np.dot(v0, v1) < 0:
            v1 = -v1
        v = v0 * (1 - a) + v1 * a
        return v / np.linalg.norm(v)
    return v0 * (1 - a) + v1 * a


def emit_clips(key, fn):
    js, binc, joints, _ = load(fn)
    nodes = js["nodes"]
    names = [j["name"] for j in joints]
    out = [f"\t{key} = {{", f"\t\tJoints = {{ {', '.join(chr(34) + n + chr(34) for n in names)} }},", "\t\tClips = {"]
    for an in js["animations"]:
        chans = {}
        length = 0
        for ch in an["channels"]:
            s = an["samplers"][ch["sampler"]]
            t = accessor(js, binc, s["input"])[:, 0]
            v = accessor(js, binc, s["output"])
            chans[(ch["target"]["node"], ch["target"]["path"])] = (t, v)
            length = max(length, t[-1])
        fps = FPS.get(an["name"], 30)
        frames = int(round(length * fps)) + 1
        times = np.arange(frames) / fps
        tracks = []
        poses = {}
        for j in joints:
            n = nodes[j["node"]]
            T0 = np.array(n.get("translation", [0, 0, 0]), dtype=float)
            Q0 = np.array(n.get("rotation", [0, 0, 0, 1]), dtype=float)
            R0inv = q2m(Q0).T
            ps, qs = [], []
            prev = None
            for t in times:
                T = sample(*chans[(j["node"], "translation")], t, False) if (j["node"], "translation") in chans else T0
                Q = sample(*chans[(j["node"], "rotation")], t, True) if (j["node"], "rotation") in chans else Q0
                dp = R0inv @ (T - T0)
                dq = qmul(qinv(Q0), Q)
                dq = dq / np.linalg.norm(dq)
                # Al estilo del juego: 180° en Y
                dp = FLIP @ dp
                dq = np.array([-dq[0], dq[1], -dq[2], dq[3]])
                if prev is not None and np.dot(prev, dq) < 0:
                    dq = -dq
                prev = dq
                ps.append(dp)
                qs.append(dq)
            ps, qs = np.array(ps), np.array(qs)
            poses[j["name"]] = (ps, qs)
            if np.all(np.ptp(ps, axis=0) < 1e-4):
                P = "c" + "".join(enc(int(round(x / PSTEP)) + POFF, 3) for x in ps[0])
            else:
                P = "k" + "".join(enc(int(round(x / PSTEP)) + POFF, 3) for p in ps for x in p)
            if np.all(np.ptp(qs, axis=0) < 2e-4):
                R = "c" + "".join(enc(int(round((c + 1) * 2047))) for c in qs[0])
            else:
                R = "k" + "".join(enc(int(round((c + 1) * 2047))) for q in qs for c in q)
            tracks.append((P, R))
        name = an["name"]
        extra = ""
        if name == "Attack":
            # Momento del golpe: entre el giro más rápido del brazo derecho y el máximo
            # adelanto del tronco.
            _, qa = poses["arm_R_shoulder"]
            speed = [2 * np.arccos(min(1.0, abs(float(np.dot(qa[k], qa[k + 1]))))) for k in range(len(qa) - 1)]
            t_arm = times[1:][int(np.argmax(speed))]
            _, qsp = poses["spine"]
            pitch = [2 * np.arcsin(np.clip(q[0], -1, 1)) for q in qsp]
            # el tronco se inclina hacia delante (x negativo tras el giro de 180°)
            t_spine = times[int(np.argmin(pitch))] if abs(min(pitch)) > abs(max(pitch)) else times[int(np.argmax(pitch))]
            extra = f" Impact = {fmt((t_arm + t_spine) / 2)},"
        out.append(f"\t\t\t{name} = {{ Length = {fmt(times[-1])}, Fps = {fps}, Frames = {frames}, Loop = {'true' if name in LOOPS else 'false'},{extra} Tracks = {{")
        for P, R in tracks:
            pl, rl = wrap(P, 120), wrap(R, 120)
            ps = pl[0] if len(pl) == 1 else "table.concat({ " + ", ".join(f'"{x}"' for x in pl) + " })"
            rs = rl[0] if len(rl) == 1 else "table.concat({ " + ", ".join(f'"{x}"' for x in rl) + " })"
            ps = f'"{ps}"' if len(pl) == 1 else ps
            rs = f'"{rs}"' if len(rl) == 1 else rs
            out.append(f"\t\t\t\t{{ P = {ps}, R = {rs} }},")
        out.append("\t\t\t} },")
    out += ["\t\t},", "\t},"]
    return "\n".join(out)


CLIPS_HEADER = """-- Animaciones de los 3 modelos 3D de la Zona 1 (Bosque), sacadas de los .glb de Santiago:
-- Idle, Walk, Attack, Hit y Death. Las reproduce EnemyAnimator (cliente) escribiendo el
-- Transform de los Motor6D "Joint" que arma ModelosZona1 (servidor); el servidor solo lee la
-- duración de Death para saber cuánto deja el cuerpo.
-- Cada pista (una por hueso, en el orden de Joints) guarda el cambio respecto a la pose de
-- reposo, a Fps fotogramas por segundo y en la unidad del .glb (se escala con el modelo):
--   P: "c" + x y z (fijo) o "k" + x y z por fotograma (3 caracteres cada uno, base 64, paso
--      0.0002, desplazados 131072)
--   R: "c" + cuaternión x y z w (fijo) o "k" + cuaternión por fotograma (2 caracteres cada
--      uno, de -1 a 1)
-- Impact (Attack): segundo del clip en que cae el golpe.
-- Generado automáticamente desde los .glb con tools/glb_a_modelos.py (no editar a mano).

return {
"""

if __name__ == "__main__":
    args = sys.argv[1:]
    data_path = clips_path = None
    pairs = []
    i = 0
    while i < len(args):
        if args[i] == "--data":
            data_path = args[i + 1]
            i += 2
        elif args[i] == "--clips":
            clips_path = args[i + 1]
            i += 2
        else:
            pairs.append(args[i].split("=", 1))
            i += 1
    if not pairs or not data_path or not clips_path:
        sys.exit(__doc__)
    with open(data_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(DATA_HEADER + "\n".join(emit_data(k, fn) for k, fn in pairs) + "\n}\n")
    with open(clips_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(CLIPS_HEADER + "\n".join(emit_clips(k, fn) for k, fn in pairs) + "\n}\n")
