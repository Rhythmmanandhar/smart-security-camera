# Archived version 1

The original implementation of this project, kept deliberately. **Nothing in
the running application imports these files.**

| File | What it is | Still used by |
|------|-----------|---------------|
| `security_camera_v1.py` | The original detector, tracker and clip writer | `tools/benchmark.py` |
| `app_v1.py` | The original Flask application | nothing — reference only |

## Why keep it

`tools/benchmark.py` runs the old detector and the current one over the same
synthetic clips, so the improvements are measured rather than asserted:

```bash
python tools/benchmark.py
```

```
case             scenario                   version  fp frames  recall  box IoU  box size
intruder         person crossing the room   v1               1   94.3%     0.72     1.38x
                                            current          0   97.1%     0.85     1.18x
```

The docstring at the top of `security_camera_v1.py` lists the defects the
benchmark demonstrates. `CHANGELOG.md` in the project root explains each one
and how it was fixed.

## Safe to delete?

Yes, if you do not need the comparison. Removing `app_v1.py` affects nothing;
removing `security_camera_v1.py` also means deleting `tools/benchmark.py`,
which imports it.
