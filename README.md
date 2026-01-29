# Apollonius 🌌

![Version](https://img.shields.io/badge/version-0.0.4--alpha-orange)
![Rust](https://img.shields.io/badge/language-Rust-red)
![License](https://img.shields.io/badge/license-MIT-blue)

**Apollonius** is a lightweight, high-performance N-dimensional geometry library for Rust. It provides the mathematical and structural foundations for physics engines, collision detection systems, and spatial simulations using `const generics`.



## ✨ Key Features
* **N-Dimensional Support:** Type-safe coordinates and vectors for 2D, 3D, and higher-dimensional spaces using Rust's `const generics`.
* **Efficient Primitives:**
    * **Hyperspheres:** (Circles, Spheres, N-Spheres) with incremental AABB caching.
    * **Segments & Lines:** Precise parametric evaluation and projection.
    * **Hyperplanes:** For half-space queries and boundary definitions.
* **Broad-Phase Foundations:** Native support for **AABB** (Axis-Aligned Bounding Boxes) with optimized overlap theorems.
* **Intersection Engine:** A unified `IntersectionResult` system that distinguishes between Tangent, Secant, and Single-point crossings.
* **Numerical Stability:** Robust floating-point classification via epsilon-weighted logic to handle accumulation errors.

## 🛠 Technical Stack
* **Language:** Rust (Stable)
* **Math Traits:** `num-traits` for generic support over `f32` and `f64`.
* **Core Philosophy:** Zero-dependency architecture (Core logic is independent of rendering or external physics frameworks).

## 📦 Installation
Add this to your `Cargo.toml`:

```toml
[dependencies]
apollonius = "0.0.4-alpha"
```

## 📖 Quick Example: Ray-Sphere Intersection
```rust
use apollonius::{Point, Line, Hypersphere, IntersectionResult};

let line = Line::new(Point::new([-5.0, 0.0]), Vector::new([1.0, 0.0]));
let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 2.0);

match line.intersect_sphere(&sphere) {
    IntersectionResult::Secant(p1, p2) => println!("Intersects at {:?} and {:?}", p1, p2),
    IntersectionResult::Tangent(p) => println!("Grazing contact at {:?}", p),
    _ => println!("No intersection"),
}
```

## 🛰 Roadmap
- [x] N-Dimensional Point & Vector algebra.
- [x] Core Primitives (Hypersphere, Line, Segment, Hyperplane).
- [x] AABB Broad-phase pruning logic.
- [x] Documentation & Doc tests overhaul.
- [ ] GJK (Gilbert-Johnson-Keerthi) implementation for narrow-phase.
- [ ] Support for Oriented Bounding Boxes (OBB).
- [ ] Spatial partitioning structures (BVH/Quadtree).

## 📝 License
This project is licensed under the MIT License.

---

**Developed with 🦀 by Mauricio Klainbard**
