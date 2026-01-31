# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for 0.0.6-alpha

- **Serialization:** Optional Serde support (`serde` feature) for `Point`, `Vector`, `FloatSign`, `IntersectionResult`, `AABB`, `Line`, `Segment`, `Hypersphere`, `Hyperplane`. Round-trip tests with `serde_json` in each module.
- **Documentation:** Expanded docs, doc tests, and examples across points, vectors, utils, and primitives. Methods returning `IntersectionResult` now document exactly which variants they can return.
- **Testing:** New serialization round-trip tests (run with `cargo test --features serde`).

*Note: 0.0.6-alpha is not yet published to crates.io.*

---

## [0.0.5-alpha] - (current)

### Added

- **Line–Line intersection:** `Line::intersect_line` for two lines (single point, parallel, or collinear).
- **Segment–Segment intersection:** `Segment::intersect_segment` with support for overlapping and touching collinear segments; broad-phase AABB only when lines are not parallel.
- **Hypersphere–Hyperplane intersection:** `Hypersphere::intersect_hyperplane` (tangent, secant, or half-space penetration). `Hypersphere::submerged_ratio` for the fraction of volume below a hyperplane.
- **Line–Segment / Segment–Line:** `Line::intersect_segment` and `Segment::intersect_line` (single point, parallel, or collinear).
- **Line–Hyperplane / Segment–Hyperplane:** `Hyperplane::intersect_line`, `Hyperplane::intersect_segment`; corresponding methods on `Line` and `Segment`.

---

## [0.0.4-alpha]

### Changed

- Documentation updates and re-export corrections only.

---

## [0.0.3-alpha]

### Added

- **Hypersphere** (and type aliases `Circle`, `Sphere`) with `closest_point`, `contains`, `is_inside`, `aabb`, and intersection with line and segment.
- **Hyperplane** with origin, normal, `closest_point`, and plane–segment intersection.
- **AABB** (Axis-Aligned Bounding Box) with `intersects` for broad-phase overlap.
- **Utils:** `classify_to_zero` and `FloatSign` enum for robust geometric comparisons with epsilon tolerance.
- **Trait `Bounded`** for types that provide an AABB.

### Changed

- **SpatialRelation:** Refactored to provide a default implementation of `distance_to_point` based on `closest_point`.
- **Segment:** Now caches squared length (and related fields) at creation for performance.

---

## [0.0.2-alpha]

### Added

- **Line** primitive: infinite line from origin and direction; `at(t)`, `closest_point`, `distance_to_point`, `contains`.
- **Segment** primitive: finite segment between two points; `at(t)`, `closest_point`, `length`, `length_squared`, `midpoint`, `direction`, `contains`.
- **Trait `SpatialRelation`** to standardize `closest_point` and `distance_to_point` across geometric primitives.

---

## [0.0.1-alpha] - Initial release

### Added

- **Point** and **Vector** with const-generic dimension `N` (`Point<T, N>`, `Vector<T, N>`).
- Type aliases: `Point2D`, `Point3D`, `Vector2D`, `Vector3D`.
- **Metrics:** `MetricSquared` and `EuclideanMetric` for points (squared distance and Euclidean distance).
- **Vector operations:** `VectorMetricSquared` (magnitude_squared, dot product), `EuclideanVector` (magnitude, normalize).
- **Arithmetic:** Point ± Vector, Point − Point → Vector, Vector ± Vector, scalar × Vector.
- **3D cross product:** `Vector3D::cross`.
- Conversions: `From<(T, T)>` / `(T, T, T)` for 2D/3D points and vectors; `From<Vector>` for `Point`; `From<(&Point, &Point)>` and `From<&Point>` for `Vector`.

---

[Unreleased]: https://github.com/MDKMauricioKlainbard/apollonius/compare/v0.0.5-alpha...HEAD
[0.0.5-alpha]: https://github.com/MDKMauricioKlainbard/apollonius/compare/v0.0.4-alpha...v0.0.5-alpha
[0.0.4-alpha]: https://github.com/MDKMauricioKlainbard/apollonius/compare/v0.0.3-alpha...v0.0.4-alpha
[0.0.3-alpha]: https://github.com/MDKMauricioKlainbard/apollonius/compare/v0.0.2-alpha...v0.0.3-alpha
[0.0.2-alpha]: https://github.com/MDKMauricioKlainbard/apollonius/compare/v0.0.1-alpha...v0.0.2-alpha
[0.0.1-alpha]: https://github.com/MDKMauricioKlainbard/apollonius/releases/tag/v0.0.1-alpha
