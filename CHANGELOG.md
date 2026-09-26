# Changelog

## 2.3 - one box per object, and seeing further

Two complaints from using the system on real footage: a single person came back
as several boxes with several ids, and only movement close to the camera was
reported.

### One box per object

- New **object-detection mode** (`--detect-objects`, or "Objects" in the
  dashboard): YOLOv8n finds the objects and each becomes one box labelled with
  its class (`person 3`, `car 5`), instead of background subtraction reporting
  the parts of a body that happen to differ from the background.
- Motion decides which objects matter: an object alarms when it moves, or when
  it has just appeared (someone who steps into view and stands still is still an
  intruder). Objects that have never moved - a parked van, a pallet stack the
  model reads as a truck - are tracked and drawn but stay quiet.
- Movement the model cannot name is still reported as an unlabelled motion box,
  so nothing that moves is ever silently dropped.
- Detections and motion are de-duplicated symmetrically: a motion blob that is
  part of, or larger than, an object already found does not become a second box
  on the same body.
- The detector runs every frame for file analysis, and a few times a second for
  a live camera, with each box carried along by its track's velocity in between.
- Overlay labels now read `person 3` rather than `ID 3 - person`.

### Fragmentation in motion-only mode

- Fragments are merged by the **gap between boxes** rather than the distance
  between their centres: pieces of one upright body overlap horizontally and
  stack vertically, which a centre distance cannot express. A guard refuses any
  merge that would produce a box wider than it is tall, so merging cannot chain
  sideways and swallow the person standing next to the first one.
- A box left nested inside the one that absorbed its neighbours is now dropped.
- On the sample footage this took one frame from six boxes for three people to
  three, without merging separate people.

### Seeing further

- The area a blob must reach at the top of the frame dropped from 45% to 22% of
  the requirement at the bottom, and the minimum box side now scales with
  distance too; both were tuned for movement close to the camera.
- The object detector runs at 960 px rather than 640: on the sample frame that
  is the difference between finding four and seven of the eight people.
- Detector confidence defaults to 0.25, which picks up a worker bent over a
  crate that 0.35 missed.

### Also

- `--detector-size` on the CLI, and a single "Detection" control in the
  dashboard replacing the classifier dropdown.
- The detection interval may now be zero, meaning "run on every frame".
- 17 new tests covering merging, distance sensitivity and object mode.

## 2.2 - control-room interface

The page was a pale form that happened to run a detector. It is now a
monitoring console, and the redesign fixed real defects as well as the looks.

### Fixed

- **`[hidden]` did nothing.** Several layout rules set `display`, which beats
  the attribute, so the Cancel button and the progress bar were on screen at
  all times. Elements marked hidden are now hidden.
- **Templates were cached.** With the debugger off, Jinja does not reload an
  edited page, so markup changes needed a server restart. `TEMPLATES_AUTO_RELOAD`
  is on.
- **The clip viewer used button classes that no longer existed**, so its
  actions rendered as bare links.
- **Live stats waited a full second** before their first update, leaving
  placeholders on screen after the camera was already running.

### Interface

- Palette derived from the overlay colours, so page and footage agree; one
  signal state drives the title-bar pill, the status strip and the stage badge.
- Incident timeline showing where incidents fall in the recording, what share
  of the footage was flagged, and clicking through to a clip.
- Built-in clip player with arrow-key navigation and `Esc` to close.
- Toast notifications for actions that used to change text silently, skeleton
  loaders for the incident grid, and designed empty states.
- Live monitor with resolution, measured frame rate, frames, incidents,
  tracked objects and uptime.
- The layout adapts to the task: browsing incidents widens the console, and the
  video stage only appears when a camera is actually running.
- Deep links: `#live`, `#incidents`, `?job=<id>` (survives a reload) and
  `?theme=light|dark`.
- Opening the dashboard while a camera session is running now attaches to it.
- Dark and light themes, keyboard focus rings throughout, `prefers-reduced-motion`
  respected, and a layout that holds down to phone width.
- `tools/check_ui.py` checks that the page, the scripts and the stylesheet
  agree about ids, icons and classes.

## 2.1 - zone drawing and object classification

### Draw zones instead of typing coordinates

- The dashboard renders a frame from the selected video or camera, and zones
  are drawn by dragging on it; the monitored zone dims everything outside it,
  and ignore zones are shaded over regions that must never alarm.
- Click a zone to delete it; Undo and Clear all are available. The numeric
  fields remain as a two-way fallback, so drawn zones can be copied into a CLI
  command and typed values appear in the editor.
- For local files the browser decodes the preview frame itself, so nothing is
  uploaded until the analysis starts. `POST /api/preview` is the fallback for
  containers browsers cannot decode and for URL sources, and it deletes its
  temporary copy immediately. `GET /api/live/preview` returns a camera frame,
  reusing the running session's frame when there is one.
- `POST /api/live/start` now accepts `roi` and `ignore_zones` as JSON
  rectangles, validated the same way as the typed form fields.

### Classify what moved

- New `src/classifier.py` with two backends behind one interface: YOLOv8n via
  `ultralytics`, and the OpenCV HOG people detector that was previously wired
  directly into the detector. With a classifier on, only recognised classes
  raise an alarm; the default is still `none`.
- YOLO runs once per frame over the whole image and matches detections to
  motion boxes by overlap, so the cost does not grow with the number of tracks.
  Each verdict is cached on the track and refreshed every 0.5 s by default.
- The class is drawn on the overlay (`ID 3 - person`) and stored in each
  incident's JSON metadata as `labels`.
- Selectable per run: `--classifier yolo --classes person,car` on the CLI, and
  a dropdown in both dashboard tabs. A missing `ultralytics` falls back to HOG
  with a warning instead of failing; `/api/health` reports `yolo_available`.
- The YOLOv8n weights (6 MB) download once into `models/`; override the path
  with `SC_YOLO_MODEL`.

### Also

- The catalogue now forgets entries whose clip is no longer on disk, instead of
  listing a broken row: `IncidentStore.prune_missing()` runs at startup, and
  removes only database rows, never files.
- The upload folder is recreated on demand, so deleting `outputs/web_uploads/`
  while the server runs no longer breaks the next upload.
- Added `.gitattributes` (LF line endings, binary file handling) and
  `legacy/README.md` explaining that the archived v1 code is not imported.
- The evidence overlay now shows the uploaded file's real name instead of the
  temporary `source.mp4`.
- `require_person` is kept as a compatibility alias for `--classifier hog`.
- The suite grew to 97 tests, covering classifier selection, fallback, verdict
  caching, classifier failure, the preview endpoints, zone validation and
  catalogue self-healing.

## 2.0 — rebuild

Everything below was a defect found in the 1.x code, which is archived under
`legacy/` and is still used by `tools/benchmark.py` for A/B comparison.

### Detection

- **No background warm-up.** The first frame of any video is 100% foreground,
  so a full-frame "motion" box appeared on frame 1 and every video opened with
  a false incident. The detector now learns the scene for ~1.2 s (capped in
  both seconds and frames) before any alarm can be raised.
- **Whole-scene changes alarmed.** Lights switching on, a bumped camera or an
  auto-exposure jump were reported as motion. Frames where more than 45% of
  the image changes now trigger a background re-learn instead.
- **Shadows counted as objects.** MOG2 ran with `detectShadows=False` and a
  magic grey threshold. Shadow detection is now on and shadow pixels are
  dropped using MOG2's own label.
- **Boxes were inflated.** The mask was dilated with an 11×11 kernel and never
  eroded back, so every box was ~10 px too large on each side. Measured
  against synthetic ground truth, boxes went from 1.38× to 1.18× the true
  area, and mean IoU from 0.72 to 0.85.
- **Thresholds were absolute pixel counts,** so the same settings behaved
  differently at 480p and 4K. Analysis now runs at a fixed width and all size
  thresholds are fractions of frame area.
- **Only area was checked.** Blobs are now also filtered on fill ratio,
  aspect ratio and minimum side, which rejects scattered compression noise.
- **ROI membership was "touches the box anywhere".** It is now an overlap
  fraction, and ignore zones were added for burnt-in clocks, tickers and
  similar always-moving regions.

### Tracking and alerting

- **Unmatched detections counted as "moving"**, which defeated the movement
  filter entirely: any new blob raised an alarm immediately. A track must now
  be confirmed over several frames before it can alert.
- **Movement was measured per step,** so a blob jittering on the spot looked
  like an intruder. Movement is now the displacement between the mean position
  of each half of the history window, which cancels oscillation.
- **The greedy matching loop could drop valid matches** (it ran a fixed number
  of rounds and consumed one per skipped pair), handing a new ID to an object
  that was already tracked. Matching now scores every candidate pair once and
  consumes them in cost order, with bounding-box overlap breaking ties.
- **Match distance was a fixed 75 px** regardless of resolution; it is now a
  fraction of the frame diagonal.
- **Labels were matched to boxes by a second nearest-neighbour pass** that
  could put someone else's ID on a box. Boxes are now drawn per track.

### Recording

- **Live clips were written with `mp4v` (MPEG-4 Part 2), which browsers cannot
  play** — the reason the viewer needed a "browser-safe stream" fallback. The
  recorder now probes encoders once, writes a throwaway file, reads it back and
  picks the first verified codec, preferring H.264/MP4 then VP8/WebM.
- **The frame that triggered an incident was written twice** (once from the
  pre-roll buffer, once as the current frame).
- **Frame rate was taken from the driver.** Webcams routinely report 30 fps
  while delivering 10, so live clips played back at triple speed. The real rate
  is now measured, and container frame rates are clamped to a sane range.
- **Every burst of noise left a clip behind.** Incidents shorter than
  `min_incident_seconds` are now discarded, and incidents longer than
  `max_incident_seconds` are split into separate clips with correct,
  non-overlapping start times.
- Clips now come with a JPEG thumbnail (the busiest frame) and a JSON sidecar
  recording times, frame counts, track IDs and the settings used.

### Live camera

- **Media Foundation was used by default on Windows,** which is slow to open
  and fails on some drivers. DirectShow is tried first, then MSMF, then any.
- **A device was reported present if `isOpened()` returned true,** which is
  true for phantom devices. A camera is only listed once it has delivered a
  frame, and the dashboard has a device picker and a rescan button.
- **The capture was released by the HTTP handler while the worker thread was
  still reading from it.** Start and stop are now serialised, and stop waits
  for the worker, so restarting the camera works.
- **The MJPEG generator spun on a `while True` loop with no wait,** burning a
  core whenever a frame was not ready and re-sending identical frames. Viewers
  now block on a condition variable.
- **Live monitoring reimplemented detection, tracking and clipping** inside the
  Flask thread, so it silently behaved differently from file analysis. Both now
  use the same pipeline.
- Dropped frames are retried and the device is reopened before a session is
  declared dead; live status reports fps, frames, incidents and alert state.
- Camera failures now produce actionable messages instead of "Could not open
  camera".

### Web application and storage

- **Deleting an incident decoded a base64 id straight into a filesystem path,**
  so a crafted id could delete files outside the output folder. Incidents are
  now identified by a database id, and both path segments are pattern-checked
  and confirmed to resolve inside the output folder.
- **Analysis ran inside the POST handler,** so the browser sat on a blocked
  request with no progress and no way to cancel. Jobs now run on a worker pool
  with progress polling and cancellation.
- **Two listings disagreed:** live clips came from SQLite, uploaded incidents
  from a filesystem walk. There is now one SQLite catalogue; clips written by
  earlier versions are imported on first start.
- **Live clip times were stored as Unix timestamps** and rendered as durations,
  so the UI showed values like "29845m".
- Added upload and download size limits, URL scheme checks, request timeouts,
  typed error responses, structured logging, byte-range media serving,
  `/api/health`, and environment-based configuration. The debugger no longer
  runs by default.
- Uploaded and downloaded source files are deleted once analysis finishes.
- Browser output is now escaped before being inserted into the page.

### Project

- Split one 768-line module into focused modules with docstrings and type
  hints; `app.py` contains routes only.
- Added a pytest suite (65 tests) built on synthetic clips with known ground
  truth, an A/B benchmark against the archived v1 code, a clip generator, a
  `.gitignore` (the repository was tracking generated clips), pinned
  requirements and a rewritten README.
- Timing settings are expressed in seconds rather than frames, so the same
  configuration behaves the same on a 6 fps export and a 30 fps webcam.
- Added an optional HOG person check (`--require-person`) for sites where only
  people should raise an alarm.
