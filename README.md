# Smart Security Camera

Detects unauthorised motion in recorded footage or on a live camera, records
each incident as a clip with a timestamped overlay, tracked object IDs, a
thumbnail and a JSON metadata sidecar, and catalogues everything in a browser
dashboard.

```
┌──────────────┐   ┌─────────────────────────────────────────┐   ┌───────────┐
│ video file   │   │ pipeline                                │   │ incident  │
│ URL          │──▶│ detector → tracker → alert logic →      │──▶│ clip      │
│ live camera  │   │ annotator → recorder                    │   │ thumbnail │
└──────────────┘   └─────────────────────────────────────────┘   │ metadata  │
                                                                 └───────────┘
```

## Quick start

```bash
pip install -r requirements.txt
python app.py                       # dashboard on http://127.0.0.1:5000
```

Command line:

```bash
python detect.py --input cctv_video_1.mp4          # analyse a file
python detect.py --live --preview                  # live camera
python detect.py --list-cameras                    # which cameras work
```

Windows PowerShell, using the project virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

## How detection works

1. **Warm-up.** The background model learns the scene for ~1.2 s before any
   alarm can be raised. (The first frame of any video is 100% foreground; this
   is why footage no longer opens with a false incident.)
2. **Foreground segmentation.** MOG2 background subtraction at a fixed analysis
   width, with MOG2's own shadow label used to drop shadows.
3. **Scene guard.** If more than 45% of the frame changes at once — lights
   switched on, camera knocked, exposure jump — the frame is used to re-learn
   the background instead of raising an alarm.
4. **Blob filtering.** Area (relative to frame size), bounding-box fill ratio,
   aspect ratio and minimum side, plus the optional monitored zone and ignore
   zones.
5. **Tracking.** Centroid tracking with overlap-assisted matching. A track must
   be seen `min_hits` times before anything trusts it.
6. **Alert test.** A track alarms only when it has actually *travelled*:
   displacement is measured between the mean position of each half of the
   history window, so something that wobbles on the spot (foliage, a flapping
   sheet, compression noise) never qualifies.
7. **Classification (optional).** A confirmed, moving track can be checked
   against YOLOv8n or the HOG people detector, or the detector can supply the
   boxes itself - one labelled box per object. See *Detection modes* below.
8. **Incident state machine.** Motion must persist for `min_motion_seconds` to
   open an incident; it closes after `post_event_seconds` of quiet; clips
   shorter than `min_incident_seconds` are discarded and clips longer than
   `max_incident_seconds` are split.

Everything above runs identically for files and for the live camera — the same
`SecurityPipeline` object drives both.

## Sensitivity

| Level | Use it for | Effect |
|-------|-----------|--------|
| `low` | Busy or noisy scenes, outdoor views | Only large, sustained, clearly travelling objects |
| `balanced` (default) | Indoor and warehouse CCTV | The tuned default |
| `high` | Quiet, controlled rooms | Small or brief movement still alarms |

The old `--suppression-level` wording still works: `medium` maps to `balanced`.

## The dashboard

The page is built as a monitoring console rather than a form: a console on the
left (analyse / live / incidents) and a monitor on the right that shows what is
happening now.

- **Signal colours match the footage.** The palette is taken from the overlay
  the detector draws, so a red badge on the page means the same thing as a red
  box in a clip: alert red, amber for a tracked-but-calm object, cyan for the
  monitored zone, green for calm.
- **One state, everywhere.** The pill in the title bar, the status strip and
  the stage badge all read from the same state - idle, busy, live, alert - so a
  glance at the top of the window is enough.
- **Incident timeline.** After an analysis, a bar shows where in the recording
  each incident falls and what share of the footage was flagged; clicking a
  segment opens that clip.
- **Built-in player.** Clips open in a player over the page, with arrow keys
  to move between incidents and `Esc` to close.
- **Live stats.** Frame rate, frames, incidents, tracked objects and uptime
  update once a second while the camera runs.
- **Deep links.** `#live` and `#incidents` open a tab directly, `?job=<id>`
  reopens a run in progress or its results after a reload, and `?theme=light`
  or `?theme=dark` pins the theme for a shared link or a kiosk screen.
- **Adopts a running session.** Opening the dashboard while the camera is
  already running attaches to the feed instead of offering to start it again,
  so a second screen shows the same picture.
- **Dark by default, light on request**, with the choice remembered; the layout
  works down to phone width, and animation is dropped for anyone who has asked
  their system to reduce motion.

## Drawing zones

The dashboard shows a frame from the chosen video (or the camera) and lets you
drag the zones onto it instead of typing coordinates:

- **Monitored zone** - one rectangle; everything outside it is dimmed in the
  editor and ignored by the detector.
- **Ignore zones** - any number of rectangles that never raise an alarm. Use
  them for burnt-in clocks, tickers, a busy pavement or a swaying branch.

Click a zone to remove it, or use Undo / Clear all. The coordinates stay
visible under *Type coordinates instead*, so anything drawn can be copied into
a CLI command, and anything typed appears in the editor.

For a local file the browser decodes the preview frame itself, so nothing is
uploaded until you start the analysis; `.avi`/`.mkv` files and URLs fall back
to `POST /api/preview`, which returns one JPEG and deletes its temporary copy
immediately.

Live-camera zones are drawn on a still from the camera and applied the next
time the camera starts.

## Detection modes

| Mode | What draws the boxes | Result | Cost |
|------|---------------------|--------|------|
| **Motion only** (default) | Background subtraction | Boxes around whatever changed | ~10 ms/frame |
| **Objects** | YOLOv8n, gated by motion | **One labelled box per person or vehicle**, distant ones included | ~250 ms/frame on CPU |
| Motion + YOLO | Motion, vetted by YOLO | Motion boxes, but only where a known object sits | ~100 ms per check |
| Motion + HOG | Motion, vetted by HOG | People only, no extra install | slower per region |

```bash
python detect.py --input cctv_video_1.mp4 --detect-objects
python detect.py --input forecourt.mp4 --detect-objects --classes person,car,truck
python detect.py --input warehouse.mp4 --classifier yolo      # vet motion instead
```

### Why "objects" gives one box per person

Background subtraction does not see bodies, it sees the parts of a body that
differ from the background: a head above a shelf, a torso, a pair of legs. Each
part becomes a box, each box becomes a track, and one person ends up wearing
three ids. Merging fragments helps and is done by default, but it cannot tell
one person's legs from the person standing next to them.

An object detector reports bodies. In this mode YOLOv8n finds the objects and
each one becomes a single box labelled with its class - `person 3`, `car 5` -
while motion decides which of them deserve an alarm:

* an object that is **moving** alarms;
* an object that has **just appeared** alarms, so someone who steps into view
  and stands still is not missed;
* an object that has sat still since the camera started - a parked van, a stack
  of pallets the model calls a truck - is tracked and drawn, but stays quiet.

Anything that moves and the model cannot name is still reported, as an
unlabelled motion box. A detector that has never seen a warehouse worker bent
over a crate should not make that movement disappear.

The detector runs on every frame for file analysis and a few times a second for
a live camera, with boxes carried along by each track's velocity in between.
Weights (~6 MB) download once into `models/`; set `SC_YOLO_MODEL` for a
different file. If `ultralytics` is missing, `yolo` falls back to HOG with a
warning rather than failing the run.

### Seeing further

Distant objects are small, and small was being filtered out: the minimum blob
size, the minimum box side and the perspective allowance were all tuned for
someone walking past the camera. Movement at the far end of a warehouse now
passes the same filters - the size a blob must reach at the top of the frame is
22% of the size demanded at the bottom, rather than 45% - and object mode looks
at the frame at 960 px rather than 640, which is the difference between finding
four and seven of the eight people in the sample footage.

## Command-line options

| Option | Default | Description |
|--------|---------|-------------|
| `--input PATH` | – | Video file to analyse |
| `--live` | off | Use a camera instead of a file |
| `--device N` | 0 | Camera index |
| `--list-cameras` | – | Print the cameras that deliver frames, then exit |
| `--output-dir PATH` | `outputs/incidents` | Where clips are written |
| `--no-record` | off | Detect only, write nothing |
| `--preview` | off | Show an annotated preview window (`q` quits) |
| `--sensitivity LEVEL` | `balanced` | `low`, `balanced` or `high` |
| `--roi X1 Y1 X2 Y2` | – | Monitor only this normalised region |
| `--ignore-zone "X1 Y1 X2 Y2"` | – | Never alarm here; repeatable |
| `--detect-objects` | off | YOLO finds the objects: one labelled box each |
| `--detector-size N` | 960 | Detector input size; larger finds smaller objects |
| `--classifier MODE` | `none` | `none`, `yolo` or `hog` (vets motion) |
| `--classes LIST` | see above | Classes that may alarm, e.g. `person,car` |
| `--classifier-confidence` | 0.35 | Minimum classifier confidence |
| `--require-person` | off | Shorthand for `--classifier hog` |
| `--min-motion-seconds` | 0.6 | Motion needed to open an incident |
| `--pre-event-seconds` | 2.0 | Footage kept before the trigger |
| `--post-event-seconds` | 2.5 | Quiet time before an incident closes |
| `--min-incident-seconds` | 1.0 | Discard shorter clips |
| `--max-incident-seconds` | 300 | Split longer clips |
| `--min-area-ratio` | preset | Minimum blob size as a fraction of frame area |
| `--process-width` | 480 | Analysis width in pixels |
| `--log-level` | INFO | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Timing is given in **seconds**, not frames, so the same settings behave the
same on a 6 fps CCTV export and a 30 fps webcam.

## Web API

| Method | Route | Purpose |
|--------|-------|---------|
| `POST` | `/api/jobs` | Start an analysis (multipart upload or `video_url`) |
| `GET` | `/api/jobs/<id>` | Progress, status and results |
| `POST` | `/api/jobs/<id>/cancel` | Cancel a running analysis |
| `GET` | `/api/incidents` | Catalogue, filterable by `?source=upload\|live\|archive` |
| `DELETE` | `/api/incidents/<id>` | Delete an incident and its files |
| `POST` | `/api/preview` | One JPEG frame from an upload or URL, for drawing zones |
| `GET` | `/api/live/preview` | One JPEG frame from a camera, for drawing zones |
| `GET` | `/api/live/devices` | Cameras that actually work (`?refresh=1` to rescan) |
| `POST` | `/api/live/start` / `/api/live/stop` | Live session control |
| `GET` | `/api/live/status` | fps, frames, incidents, alert state |
| `GET` | `/api/live/feed` | MJPEG preview stream |
| `GET` | `/api/live/snapshot` | Current frame as JPEG |
| `GET` | `/api/health` | Service health and storage stats |
| `GET` | `/media/<job>/<file>` | Clip, thumbnail or metadata (range requests supported) |
| `GET` | `/live/clips` | Legacy alias kept from v1 (live clips only) |

Server settings come from the environment: `SC_HOST`, `SC_PORT`, `SC_DEBUG`,
`SC_MAX_UPLOAD_MB`, `SC_MAX_DOWNLOAD_MB`, `SC_WORKERS`, `SC_OUTPUT_DIR`.

## Output layout

```
outputs/
  incidents.db                      catalogue (SQLite)
  web_jobs/<job_id>/incidents/
      incident_001.mp4              clip, browser-playable codec
      incident_001.jpg              thumbnail (busiest frame)
      incident_001.json             metadata: times, frames, track IDs, settings
```

The recorder probes the available encoders once and picks the first that
writes a file this machine can read back, preferring H.264/MP4 then VP8/WebM.
MPEG-4 Part 2 (`mp4v`) is only a last resort because browsers cannot play it.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest                 # 97 tests
python tools/benchmark.py        # v1 vs current, on synthetic ground truth
python tools/check_ui.py         # page, scripts and stylesheet agree
python tools/make_test_video.py  # write the synthetic clips to outputs/testclips
```

The test suite builds synthetic clips whose ground truth is known — an empty
room, an intruder crossing between two exact frames, a light switch, a ticking
burnt-in clock and swaying foliage — and asserts both that the intruder is
found and that the other four raise nothing.

## Project layout

```
app.py                 Flask routes only; all logic lives in src/
src/config.py          every tunable, plus the sensitivity presets
src/detector.py        foreground segmentation and blob filtering
src/tracker.py         centroid tracking, confirmation, movement test
src/classifier.py      optional YOLOv8n / HOG "what is it?" stage
src/pipeline.py        alert logic and the incident state machine
src/recorder.py        codec probing, clip writing, thumbnails, metadata
src/annotator.py       the evidence overlay
src/camera_source.py   opening files and cameras; Windows backend handling
src/live.py            live session: thread, preview, statistics
src/jobs.py            background analysis jobs with progress and cancel
src/storage.py         SQLite incident catalogue and safe path resolution
src/security_camera.py public API and CLI (detect.py is the entry point)
legacy/                the archived v1 code, kept for the A/B benchmark
                       (see legacy/README.md - nothing imports it)
static/app.js          dashboard logic: jobs, live, incidents, player
static/zones.js        the zone-drawing editor
static/styles.css      design tokens and the control-room theme
models/                YOLO weights, downloaded on first use (gitignored)
outputs/               clips, thumbnails and the catalogue (gitignored)
tools/                 synthetic clip generator and the benchmark
tests/                 pytest suite
```

## Troubleshooting

**"Could not start camera 0"** — close anything else using the camera (Teams,
Zoom, the Camera app), check Settings → Privacy → Camera, then run
`python detect.py --list-cameras`.

**Too many alerts** — turn on `--classifier yolo` so only people and vehicles
alarm, lower the sensitivity, draw an ignore zone over the noisy region, or
raise `--min-motion-seconds`.

**Several boxes on one person** — that is background subtraction reporting body
parts. Use `--detect-objects`, which reports bodies.

**Only nearby movement is caught** — use `--detect-objects` (it looks at the
frame at 960 px and finds small distant figures), raise the sensitivity, or
lower `perspective_top_scale` in `src/config.py` for an unusually deep view.

**Missed events** — raise the sensitivity, lower `--min-motion-seconds`, or
check that the monitored zone covers the area. `--log-level DEBUG` reports
scene changes and warm-up.

**Clips will not play in the browser** — the log line `Using <fourcc> for
incident clips` says which encoder was selected. If it says `mp4v`, this
OpenCV build has no H.264 or VP8 encoder; install
`opencv-python` from PyPI (rather than a system package) or download the
clip and play it in VLC.
