# Legacy SQL

<div style="display:flex;flex-wrap:wrap;gap:6px 8px;margin:6px 0 14px 0;">
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Layer</span>
    <span style="background:#69BE45;color:#FFFFFF;padding:5px 9px;">Legacy SQL</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Purpose</span>
    <span style="background:#2F80C9;color:#FFFFFF;padding:5px 9px;">Preserved Project History</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Structure</span>
    <span style="background:#2F80C9;color:#FFFFFF;padding:5px 9px;">Foundation to Marts</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Status</span>
    <span style="background:#E88A45;color:#FFFFFF;padding:5px 9px;">Maintained for Reference</span>
  </span>
</div>

This directory preserves the original SQL work while giving it a clearer and more navigable structure.

The goal is not to hide or discard the earlier implementation. It is to keep the original depth visible while making the project easier to understand and explain.

## Directory structure

- `foundation/` contains bootstrap logic and shared cleaning steps
- `staging/` contains quarter-level trip cleaning workflows
- `features/` contains classification and clustering work
- `marts/` contains analysis-ready views
- `enrichment/` contains demographic and transportation joins

Some scripts still reflect historical assumptions and older naming conventions. They remain here intentionally as preserved project history while the new `sql/warehouse/` layer continues to grow around them.