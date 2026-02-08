/**
 * Tests for the pure geometry helper functions (polygonArea, simplifyPolygon).
 */

import { polygonArea, simplifyPolygon } from "../geometry";
import type { Point } from "../geometry";

// ---------------------------------------------------------------------------
// polygonArea
// ---------------------------------------------------------------------------

describe("polygonArea", () => {
  test("unit square (CW in screen coords) returns positive area", () => {
    // (0,0)->(100,0)->(100,100)->(0,100) is CW in screen coords (y-down)
    // Shoelace gives positive for this winding
    const square: Point[] = [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 100, y: 100 },
      { x: 0, y: 100 },
    ];
    const area = polygonArea(square);
    expect(area).toBeCloseTo(10000, 0);
  });

  test("unit square (CCW in screen coords) returns negative area", () => {
    // Reversed winding
    const square: Point[] = [
      { x: 0, y: 0 },
      { x: 0, y: 100 },
      { x: 100, y: 100 },
      { x: 100, y: 0 },
    ];
    const area = polygonArea(square);
    expect(area).toBeCloseTo(-10000, 0);
  });

  test("right triangle has correct area", () => {
    // Triangle (0,0)-(100,0)-(0,100) → area = 5000
    const tri: Point[] = [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 0, y: 100 },
    ];
    const area = polygonArea(tri);
    expect(Math.abs(area)).toBeCloseTo(5000, 0);
  });

  test("degenerate (collinear points) returns zero area", () => {
    const line: Point[] = [
      { x: 0, y: 0 },
      { x: 50, y: 50 },
      { x: 100, y: 100 },
    ];
    expect(polygonArea(line)).toBeCloseTo(0, 5);
  });

  test("single point returns zero", () => {
    expect(polygonArea([{ x: 5, y: 5 }])).toBe(0);
  });

  test("empty array returns zero", () => {
    expect(polygonArea([])).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// simplifyPolygon
// ---------------------------------------------------------------------------

describe("simplifyPolygon", () => {
  test("reduces vertex count to maxVerts", () => {
    // Circle-like polygon with 20 vertices
    const pts: Point[] = [];
    for (let i = 0; i < 20; i++) {
      const angle = (2 * Math.PI * i) / 20;
      pts.push({ x: 100 * Math.cos(angle), y: 100 * Math.sin(angle) });
    }
    const simplified = simplifyPolygon(pts, 8);
    expect(simplified.length).toBe(8);
  });

  test("does not modify when already at maxVerts", () => {
    const square: Point[] = [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 100, y: 100 },
      { x: 0, y: 100 },
    ];
    const result = simplifyPolygon(square, 4);
    expect(result).toEqual(square);
  });

  test("does not modify when below maxVerts", () => {
    const tri: Point[] = [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 50, y: 100 },
    ];
    const result = simplifyPolygon(tri, 10);
    expect(result).toEqual(tri);
  });

  test("does not mutate the input array", () => {
    const pts: Point[] = [];
    for (let i = 0; i < 10; i++) {
      pts.push({ x: i * 10, y: (i % 2) * 20 });
    }
    const copy = pts.map((p) => ({ ...p }));
    simplifyPolygon(pts, 5);
    expect(pts).toEqual(copy);
  });

  test("preserves overall shape (area stays similar)", () => {
    // Regular octagon
    const pts: Point[] = [];
    for (let i = 0; i < 8; i++) {
      const angle = (2 * Math.PI * i) / 8;
      pts.push({ x: 100 * Math.cos(angle), y: 100 * Math.sin(angle) });
    }
    const originalArea = Math.abs(polygonArea(pts));
    const simplified = simplifyPolygon(pts, 4);
    const newArea = Math.abs(polygonArea(simplified));
    // Area should still be in a reasonable range (within 50% of original)
    expect(newArea).toBeGreaterThan(originalArea * 0.5);
    expect(newArea).toBeLessThan(originalArea * 1.5);
  });
});
