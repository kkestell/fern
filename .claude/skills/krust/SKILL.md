---
name: krust
description: "Rust API design guidelines from the Rust library team — naming, interoperability, predictability, type safety, future-proofing, and more. Use when designing, writing, or reviewing a Rust crate's public API to make it idiomatic and interoperable with the existing crate ecosystem."
---

## Overview

This skill bundles the official Rust API Guidelines: recommendations from the Rust library team on how to design and present public APIs, drawn from building the standard library and other ecosystem crates.

Treat them as considerations, not mandates — some are firm, others are still in development. Crates that conform tend to integrate more cleanly with the rest of the ecosystem.

## When to use

Consult these when you are designing a new Rust crate's public surface, adding public items to an existing crate, or reviewing an API for idiomatic and interoperable design.

## How to use

Start with the [checklist](assets/checklist.md) for a condensed, scannable summary of every guideline. When a checklist item is relevant, open the matching topic file below for the full rationale and examples, and apply it to the code under review.

The full guidelines live in this skill's `assets/` directory:

- [Checklist](assets/checklist.md) — condensed summary of all guidelines
- [Naming](assets/naming.md)
- [Interoperability](assets/interoperability.md)
- [Macros](assets/macros.md)
- [Documentation](assets/documentation.md)
- [Predictability](assets/predictability.md)
- [Flexibility](assets/flexibility.md)
- [Type safety](assets/type-safety.md)
- [Dependability](assets/dependability.md)
- [Debuggability](assets/debuggability.md)
- [Future proofing](assets/future-proofing.md)
- [Necessities](assets/necessities.md)
