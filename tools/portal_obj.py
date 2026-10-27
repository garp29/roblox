# Convierte el diseño "Portal del Renacimiento" (OBJ exportado de Claude Design) en el
# módulo de datos src/server/Util/PortalDesign.luau, que PortalBuilder pasa a piezas.
#
#   python3 tools/portal_obj.py portal-renacimiento.obj src/server/Util/PortalDesign.luau
#
# Reconoce las formas de three.js del diseño por su número de vértices y caras: cajas,
# octaedros (cristales y runas planas), cilindros de 64 lados (la tarima), esferas
# escaladas (musgo, nudos de raíz), cilindros finos (raíces) y los obeliscos (tronco de
# pirámide de 4 lados con su punta). Deja fuera el escenario de muestra (hierba, camino,
# árboles y el avatar de referencia). Necesita numpy.
import itertools
import json
import math
import sys

import numpy as np

SCENERY = {"Grass", "Path", "Trunk", "Crown", "Leg", "Arm", "Torso", "Head"}

# Capas finas del suelo de la tarima: se separan un poco para que no parpadeen en Roblox
# (z-fighting). Rango vertical nuevo (abajo, arriba) en metros del diseño.
LAYERS = {
    "DaisInlayInner": (0.2245, 0.2455),
    "RuneRing1": (0.2455, 0.2625),
}


def read_obj(path):
    objects, current, vertices = [], None, []
    for line in open(path):
        if line.startswith("o "):
            current = {"name": line[2:].strip(), "mat": None, "faces": []}
            objects.append(current)
        elif line.startswith("usemtl "):
            current["mat"] = line[7:].strip()
        elif line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            current["faces"].append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    vertices = np.array(vertices)
    for obj in objects:
        index = sorted({i for face in obj["faces"] for i in face})
        obj["all"] = vertices[index]
        obj["unique"] = np.unique(np.round(vertices[index], 5), axis=0)
    return objects


def rotation(columns):
    """Columnas (vectores de los ejes, con su largo) -> (matriz de giro, tamaños)."""
    sizes = [float(np.linalg.norm(c)) for c in columns]
    matrix = np.array([c / np.linalg.norm(c) for c in columns]).T
    if np.linalg.det(matrix) < 0:
        matrix[:, 2] *= -1
    return matrix, sizes


def near_identity(columns):
    """Ordena y orienta los ejes para que el giro quede lo más cerca posible de ninguno."""
    best = None
    for perm in itertools.permutations(range(3)):
        score = sum(abs(columns[p][i]) / np.linalg.norm(columns[p]) for i, p in enumerate(perm))
        if best is None or score > best[0] + 1e-9:
            best = (score, perm)
    ordered = [np.array(columns[p], float) for p in best[1]]
    for i in range(3):
        if ordered[i][i] < 0:
            ordered[i] = -ordered[i]
    return rotation(ordered)


def euler(matrix):
    """Matriz -> grados (rx, ry, rz) de CFrame.Angles (= Rx·Ry·Rz)."""
    ry = math.asin(max(-1.0, min(1.0, matrix[0, 2])))
    rx = math.atan2(-matrix[1, 2], matrix[2, 2])
    rz = math.atan2(-matrix[0, 1], matrix[0, 0])
    return [round(math.degrees(a), 1) + 0.0 for a in (rx, ry, rz)]


def angles(rx, ry, rz):
    """Grados de CFrame.Angles -> matriz (Rx·Ry·Rz)."""
    a, b, c = (math.radians(v) for v in (rx, ry, rz))
    x = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    y = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    z = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return x @ y @ z


def box_edges(points):
    origin = points[0]
    others = [p - origin for p in points[1:]]
    for trio in itertools.combinations(range(7), 3):
        edges = [others[i] for i in trio]
        lengths = [np.linalg.norm(e) for e in edges]
        square = all(abs(np.dot(edges[i], edges[j])) < 1e-3 * lengths[i] * lengths[j] + 1e-9 for i, j in ((0, 1), (0, 2), (1, 2)))
        if square and min(np.linalg.norm(sum(edges) - p) for p in others) < 1e-3:
            return edges
    raise ValueError("no parece una caja")


def octahedron_axes(points):
    center = points.mean(0)
    used, axes = set(), []
    for i in range(6):
        if i in used:
            continue
        j = min((k for k in range(6) if k != i), key=lambda k: np.linalg.norm(points[i] + points[k] - 2 * center))
        used |= {i, j}
        axes.append(points[i] - points[j])
    return center, axes


def r3(values):
    return [round(float(v), 3) + 0.0 for v in values]


def convert(obj):
    pts, name, mat = obj["unique"], obj["name"], obj["mat"]
    count, faces = len(pts), len(obj["faces"])
    if count == 8 and faces == 12:
        matrix, size = near_identity(box_edges(pts))
        return ["Box", name, mat, *r3(pts.mean(0)), *r3(size), *euler(matrix)]
    if name == "ObeliskTip" or count == 10:
        ys = pts[:, 1]
        bottom = pts[np.abs(ys - ys.min()) < 1e-4]
        top = pts[np.abs(ys - ys.max()) < 1e-4]
        ring = [p for p in bottom if np.linalg.norm(p - bottom.mean(0)) > 1e-4]
        base = np.mean(ring, 0)
        corner = math.degrees(math.atan2(ring[0][2] - base[2], ring[0][0] - base[0]))
        yaw = (45 - corner) % 90
        yaw = yaw - 90 if yaw > 45 else yaw
        side = float(np.mean([np.linalg.norm(p - base) for p in ring]) * math.sqrt(2))
        height = round(float(ys.max() - ys.min()), 3)
        if count == 10:
            top_ring = [p for p in top if np.linalg.norm(p - top.mean(0)) > 1e-4]
            top_side = float(np.mean([np.linalg.norm(p - np.mean(top_ring, 0)) for p in top_ring]) * math.sqrt(2))
            return ["Frustum", name, mat, *r3(base), height, round(side, 3), round(top_side, 3), round(yaw, 1)]
        return ["Pyramid", name, mat, *r3(base), height, round(side, 3), round(yaw, 1)]
    if count == 6 and faces == 8:
        center, axes = octahedron_axes(pts)
        lengths = [np.linalg.norm(a) for a in axes]
        order = np.argsort(lengths)
        if lengths[order[0]] < 0.6 * lengths[order[1]]:
            # Runa plana: rombo en su plano XY, grosor en Z.
            a1, a2, thin = axes[order[1]], axes[order[2]], axes[order[0]]
            if np.dot(np.cross(a1, a2), thin) < 0:
                thin = -thin
            matrix, _ = rotation([a1, a2, thin])
            diag = (lengths[order[1]] + lengths[order[2]]) / 2
            return ["Glyph", name, mat, *r3(center), round(float(diag), 3), round(float(lengths[order[0]]), 3), *euler(matrix)]
        # Cristal: la punta es el eje que más se distingue de los otros dos.
        apart = [abs(lengths[i] - np.mean([lengths[j] for j in range(3) if j != i])) for i in range(3)]
        k = int(np.argmax(apart))
        rest = [i for i in range(3) if i != k]
        apex, side = axes[k], axes[rest[0]]
        third = np.cross(side, apex)
        matrix, _ = rotation([side, apex, third / np.linalg.norm(third) * lengths[rest[1]]])
        width = (lengths[rest[0]] + lengths[rest[1]]) / 2
        return ["Crystal", name, mat, *r3(center), round(float(lengths[k]), 3), round(float(width), 3), *euler(matrix)]
    if count == 130:
        ys = pts[:, 1]
        bottom, top = LAYERS.get(name, (ys.min(), ys.max()))
        diameter = pts[:, 0].max() - pts[:, 0].min()
        return ["Disc", name, mat, *r3([pts[:, 0].mean(), (bottom + top) / 2, pts[:, 2].mean()]), round(float(diameter), 3), round(float(top - bottom), 3)]
    if count in (262, 362):
        center = pts.mean(0)
        spread = pts - center
        _, vectors = np.linalg.eigh(spread.T @ spread)
        projected = spread @ vectors
        extent = projected.max(0) - projected.min(0)
        center = center + ((projected.max(0) + projected.min(0)) / 2) @ vectors.T
        matrix, size = near_identity([vectors[:, i] * extent[i] for i in range(3)])
        return ["Ball", name, mat, *r3(center), *r3(size), *euler(matrix)]
    if name == "Root":
        values, counts = np.unique(np.round(obj["all"], 4), axis=0, return_counts=True)
        caps = sorted(values[counts >= 6], key=lambda p: p[1])
        radius = []
        for cap in caps:
            near = [np.linalg.norm(p - cap) for p in pts]
            radius.append(np.mean([d for d in near if 1e-4 < d < 0.3]))
        return ["Rod", name, mat, *r3(caps[0]), *r3(caps[1]), round(float(sum(radius)), 3)]
    raise ValueError(f"forma desconocida: {name} ({count} vértices, {faces} caras)")


def relayer(part):
    """Aplica LAYERS a las piezas planas del suelo (cajas)."""
    if part[0] == "Box" and part[1] in LAYERS:
        bottom, top = LAYERS[part[1]]
        part[4] = round((bottom + top) / 2, 4)
        part[7] = round(top - bottom, 4)
    return part


def fmt(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def main(source, target):
    parts = [relayer(convert(o)) for o in read_obj(source) if o["name"] not in SCENERY]
    edges = [p for p in parts if p[1] == "InnerEdge"]
    posts = [p for p in edges if p[7] > 1]
    half = abs(posts[0][3]) - posts[0][6] / 2
    bottom = posts[0][4] - posts[0][7] / 2
    spring = posts[0][4] + posts[0][7] / 2
    depth = posts[0][8]
    footprint = 0.0
    height = 0.0
    for p in parts:
        if p[0] == "Disc":
            footprint = max(footprint, math.hypot(p[3], p[5]) + p[6] / 2)
        elif p[0] == "Box" and p[4] < 0.5:  # lo que está a ras de suelo: esquinas reales
            matrix = angles(p[9], p[10], p[11])
            for signs in itertools.product((-0.5, 0.5), repeat=3):
                corner = np.array(p[3:6]) + matrix @ (np.array(signs) * np.array(p[6:9]))
                footprint = max(footprint, math.hypot(corner[0], corner[2]))
        elif p[0] == "Crystal":
            height = max(height, p[4] + p[6] / 2)
    lines = [
        "-- Portal del Renacimiento: el diseño 3D de Santiago (Claude Design, exportado como",
        "-- portal-renacimiento.obj) convertido en piezas. GENERADO con tools/portal_obj.py: no",
        "-- se edita a mano; si cambia el diseño, se vuelve a generar desde el .obj.",
        "--",
        "-- Medidas en metros del diseño (su avatar de referencia mide 1,46): PortalBuilder las",
        "-- pasa a studs con ZoneConfig.PortalScale. Origen: centro de la tarima, en el suelo. El",
        "-- frente (por donde se llega) mira a +Z y el arco cruza de lado a lado en X.",
        "--",
        "-- Cada pieza: { forma, nombre, material, ... }",
        "--   Box      x, y, z, ancho, alto, fondo, rx, ry, rz   (grados, como CFrame.Angles)",
        "--   Ball     x, y, z, ancho, alto, fondo, rx, ry, rz   (elipsoide)",
        "--   Glyph    x, y, z, diagonal, grosor, rx, ry, rz     (runa: rombo en su plano XY)",
        "--   Crystal  x, y, z, alto, ancho, rx, ry, rz          (octaedro con la punta en su Y)",
        "--   Disc     x, y, z, diámetro, alto",
        "--   Rod      ax, ay, az, bx, by, bz, grosor            (raíz recta de A a B)",
        "--   Frustum  x, y, z, alto, lado abajo, lado arriba, giro   (obelisco, base en y)",
        "--   Pyramid  x, y, z, alto, lado, giro                 (punta del obelisco, base en y)",
        "",
        "return {",
        "\t-- Hueco del arco (donde va la energía): rectángulo y medio círculo encima.",
        f"\tOpening = {{ HalfWidth = {fmt(half)}, Bottom = {fmt(bottom)}, Spring = {fmt(spring)}, Radius = {fmt(half)}, Depth = {fmt(depth)} }},",
        f"\tFootprint = {fmt(footprint)}, -- radio que ocupa en el suelo (tarima y escalones)",
        f"\tHeight = {fmt(height)}, -- hasta la punta del cristal del arco",
        "\tParts = {",
    ]
    for p in parts:
        lines.append("\t\t{ " + ", ".join(fmt(v) for v in p) + " },")
    lines += ["\t},", "}", ""]
    with open(target, "w", newline="\n") as out:
        out.write("\n".join(lines))
    kinds = {}
    for p in parts:
        kinds[p[0]] = kinds.get(p[0], 0) + 1
    print(f"{len(parts)} piezas {kinds} -> {target}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
