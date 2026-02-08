/**
 * Pure geometry helper functions extracted from MainScene for testability.
 */

export type Point = { x: number; y: number };

/**
 * Signed area of a polygon (positive = clockwise in screen coords).
 *
 * Uses the shoelace formula.
 */
export function polygonArea(verts: Point[]): number {
  let area = 0;
  const n = verts.length;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    area += verts[i].x * verts[j].y;
    area -= verts[j].x * verts[i].y;
  }
  return area / 2;
}

/**
 * Simplify a polygon to at most `maxVerts` vertices using iterative
 * removal of the vertex that contributes the least area (Visvalingam-like).
 */
export function simplifyPolygon(
  verts: Point[],
  maxVerts: number,
): Point[] {
  const pts = verts.slice(); // copy
  while (pts.length > maxVerts) {
    // Find the vertex whose removal changes the area the least
    let minCost = Infinity;
    let minIdx = 1;
    for (let i = 0; i < pts.length; i++) {
      const prev = pts[(i - 1 + pts.length) % pts.length];
      const curr = pts[i];
      const next = pts[(i + 1) % pts.length];
      // Triangle area formed by prev-curr-next
      const cost = Math.abs(
        (prev.x * (curr.y - next.y) +
          curr.x * (next.y - prev.y) +
          next.x * (prev.y - curr.y)) / 2,
      );
      if (cost < minCost) {
        minCost = cost;
        minIdx = i;
      }
    }
    pts.splice(minIdx, 1);
  }
  return pts;
}
