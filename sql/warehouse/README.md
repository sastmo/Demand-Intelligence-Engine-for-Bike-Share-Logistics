# Warehouse Layer

<div style="display:flex;flex-wrap:wrap;gap:6px 8px;margin:6px 0 14px 0;">
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Layer</span>
    <span style="background:#69BE45;color:#FFFFFF;padding:5px 9px;">Warehouse</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Purpose</span>
    <span style="background:#2F80C9;color:#FFFFFF;padding:5px 9px;">Modernization Bridge</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Scope</span>
    <span style="background:#2F80C9;color:#FFFFFF;padding:5px 9px;">Contracts + Utilities + Ops</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Status</span>
    <span style="background:#E88A45;color:#FFFFFF;padding:5px 9px;">Phase 1 Foundation</span>
  </span>
</div>

This directory contains the new project assets introduced during the modernization pass.

## Purpose

The warehouse layer serves as the bridge between the preserved legacy SQL and a more reusable, maintainable data engineering structure.

## Contents

- `contracts/`: inventories, manifests, and repository-level expectations
- `orchestration/`: runbooks and execution notes
- `utilities/`: reusable SQL helpers shared across multiple scripts

## Current state

Phase 1 is intentionally light-touch. The goal is to improve the project’s structure and presentation without implying that the repository has already been fully rewritten into a new platform.

Phase 2 can build on this layer by introducing canonical schemas, repeatable orchestration, and stricter data testing.