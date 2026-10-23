# Rules

## Heavy work and the PC
- grip_scan is special: it does heavy optimisation work. Agents may work on Olex's PC (laptop OLEX, checkout `D:\dev\grip_scan`, via Remote Control) and use its GPU/CPU acceleration for optimisation runs.
- Every product of heavy runs (results, meshes, caches, logs, renders, virtual environments, downloads) goes inside the repo's `gen/` folder, which is gitignored. Nothing like that goes anywhere else in the repo.
- Blender work still runs headless in the cloud.
- Ask Olex only before something really important on the PC. Routine runs need no question.

## Privacy
- This repo is public. Never commit or push hand photos, videos, crops, scans (.ply etc.) or .blend files containing the hand. Move those through the project files or the PC instead. Code only on GitHub.

## Blender output
- Renders use flat shading, never smooth shading, so the real facets show.
- Grip scenes leave out the heavy case top/bottom geometry: only the basemesh plus the buttons needed.
- Any .blend handed to Olex matches his original Grip.blend setup: same workspaces, same addons enabled, tidy, work only in the scene GripV.
