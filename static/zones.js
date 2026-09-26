/* Interactive zone editor.
 *
 * Draws a monitored zone (one rectangle) and any number of ignore zones onto a
 * still frame from the video or camera, and reports them in normalised 0..1
 * coordinates so they are independent of the frame's pixel size.
 *
 * The canvas is kept pixel-aligned with the displayed image (including device
 * pixel ratio and any resize), so what is drawn is exactly what is saved.
 */

(function (global) {
  "use strict";

  const ROI_COLOR = "#e0a052";
  const IGNORE_COLOR = "#8a8a8a";
  const MIN_SIZE = 0.02; // ignore accidental clicks: 2% of the frame

  class ZoneEditor {
    /**
     * @param {object} options
     * @param {HTMLElement} options.root      container holding the toolbar and stage
     * @param {HTMLImageElement} options.image the frame to draw on
     * @param {HTMLCanvasElement} options.canvas overlay canvas
     * @param {function} [options.onChange]   called with {roi, ignore} on every edit
     */
    constructor(options) {
      this.root = options.root;
      this.image = options.image;
      this.canvas = options.canvas;
      this.onChange = options.onChange || function () {};
      this.context = this.canvas.getContext("2d");

      this.mode = "roi";
      this.roi = null; // [x1, y1, x2, y2] normalised
      this.ignore = []; // array of the same
      this.history = [];
      this.drag = null;

      this._bindToolbar();
      this._bindPointer();

      this._onResize = () => this.redraw();
      global.addEventListener("resize", this._onResize);
      if (global.ResizeObserver) {
        this._observer = new ResizeObserver(this._onResize);
        this._observer.observe(this.image);
      }
      this.image.addEventListener("load", this._onResize);
    }

    /* ---------------------------------------------------------- lifecycle */

    setImage(src) {
      return new Promise((resolve, reject) => {
        this.image.onload = () => {
          this.redraw();
          resolve();
        };
        this.image.onerror = () => reject(new Error("The preview frame could not be loaded."));
        this.image.src = src;
      });
    }

    setZones(roi, ignore) {
      this.roi = roi || null;
      this.ignore = Array.isArray(ignore) ? ignore.slice() : [];
      this.redraw();
    }

    getZones() {
      return { roi: this.roi, ignore: this.ignore.slice() };
    }

    setMode(mode) {
      this.mode = mode === "ignore" ? "ignore" : "roi";
      this.root.querySelectorAll("[data-zone-mode]").forEach((button) => {
        button.classList.toggle("active", button.dataset.zoneMode === this.mode);
      });
    }

    clear() {
      this._remember();
      this.roi = null;
      this.ignore = [];
      this._changed();
    }

    undo() {
      const previous = this.history.pop();
      if (!previous) return;
      this.roi = previous.roi;
      this.ignore = previous.ignore;
      this.redraw();
      this.onChange(this.getZones());
    }

    destroy() {
      global.removeEventListener("resize", this._onResize);
      if (this._observer) this._observer.disconnect();
    }

    /* ------------------------------------------------------------- events */

    _bindToolbar() {
      this.root.querySelectorAll("[data-zone-mode]").forEach((button) => {
        button.addEventListener("click", () => this.setMode(button.dataset.zoneMode));
      });
      this.root.querySelectorAll("[data-zone-action]").forEach((button) => {
        button.addEventListener("click", () => {
          if (button.dataset.zoneAction === "clear") this.clear();
          if (button.dataset.zoneAction === "undo") this.undo();
        });
      });
    }

    _bindPointer() {
      const canvas = this.canvas;

      canvas.addEventListener("pointerdown", (event) => {
        if (!this.image.naturalWidth) return;
        canvas.setPointerCapture(event.pointerId);
        const point = this._pointOf(event);
        this.drag = { start: point, current: point, moved: false };
      });

      canvas.addEventListener("pointermove", (event) => {
        if (!this.drag) return;
        this.drag.current = this._pointOf(event);
        const [dx, dy] = [
          Math.abs(this.drag.current[0] - this.drag.start[0]),
          Math.abs(this.drag.current[1] - this.drag.start[1]),
        ];
        this.drag.moved = dx > 0.005 || dy > 0.005;
        this.redraw();
      });

      const finish = (event) => {
        if (!this.drag) return;
        const rect = this._normalise(this.drag.start, this._pointOf(event));
        const wasDrag = this.drag.moved;
        this.drag = null;

        if (wasDrag && rect[2] - rect[0] >= MIN_SIZE && rect[3] - rect[1] >= MIN_SIZE) {
          this._remember();
          if (this.mode === "roi") this.roi = rect;
          else this.ignore.push(rect);
          this._changed();
          return;
        }
        // A click, not a drag: remove whichever zone sits under the pointer.
        this._removeAt(this._pointOf(event));
      };

      canvas.addEventListener("pointerup", finish);
      canvas.addEventListener("pointercancel", () => {
        this.drag = null;
        this.redraw();
      });
    }

    _removeAt(point) {
      const hitIndex = this.ignore.findIndex((zone) => this._contains(zone, point));
      if (hitIndex >= 0) {
        this._remember();
        this.ignore.splice(hitIndex, 1);
        this._changed();
        return;
      }
      if (this.roi && this._contains(this.roi, point)) {
        this._remember();
        this.roi = null;
        this._changed();
      }
    }

    /* ---------------------------------------------------------- geometry */

    _pointOf(event) {
      const bounds = this.canvas.getBoundingClientRect();
      const x = (event.clientX - bounds.left) / (bounds.width || 1);
      const y = (event.clientY - bounds.top) / (bounds.height || 1);
      return [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];
    }

    _normalise(a, b) {
      return [
        Math.min(a[0], b[0]),
        Math.min(a[1], b[1]),
        Math.max(a[0], b[0]),
        Math.max(a[1], b[1]),
      ];
    }

    _contains(zone, point) {
      return (
        point[0] >= zone[0] && point[0] <= zone[2] &&
        point[1] >= zone[1] && point[1] <= zone[3]
      );
    }

    _remember() {
      this.history.push({ roi: this.roi, ignore: this.ignore.slice() });
      if (this.history.length > 30) this.history.shift();
    }

    _changed() {
      this.redraw();
      this.onChange(this.getZones());
    }

    /* ---------------------------------------------------------- rendering */

    redraw() {
      const canvas = this.canvas;
      const bounds = this.image.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return;

      const ratio = global.devicePixelRatio || 1;
      canvas.width = Math.round(bounds.width * ratio);
      canvas.height = Math.round(bounds.height * ratio);
      canvas.style.width = `${bounds.width}px`;
      canvas.style.height = `${bounds.height}px`;

      const context = this.context;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, bounds.width, bounds.height);

      const toPixels = (zone) => [
        zone[0] * bounds.width,
        zone[1] * bounds.height,
        (zone[2] - zone[0]) * bounds.width,
        (zone[3] - zone[1]) * bounds.height,
      ];

      if (this.roi) {
        // Everything outside the monitored zone is dimmed, which shows at a
        // glance how much of the frame is actually being watched.
        context.fillStyle = "rgba(20, 16, 12, 0.45)";
        context.fillRect(0, 0, bounds.width, bounds.height);
        const [x, y, w, h] = toPixels(this.roi);
        context.clearRect(x, y, w, h);
        this._strokeZone([x, y, w, h], ROI_COLOR, "Monitored");
      }

      this.ignore.forEach((zone) => {
        const box = toPixels(zone);
        context.fillStyle = "rgba(120, 120, 120, 0.35)";
        context.fillRect(box[0], box[1], box[2], box[3]);
        this._strokeZone(box, IGNORE_COLOR, "Ignored");
      });

      if (this.drag) {
        const rect = this._normalise(this.drag.start, this.drag.current);
        const box = toPixels(rect);
        context.setLineDash([6, 4]);
        this._strokeZone(box, this.mode === "roi" ? ROI_COLOR : IGNORE_COLOR, "");
        context.setLineDash([]);
      }
    }

    _strokeZone(box, color, label) {
      const context = this.context;
      context.lineWidth = 2;
      context.strokeStyle = color;
      context.strokeRect(box[0], box[1], box[2], box[3]);

      if (!label) return;
      context.font = "600 12px system-ui, sans-serif";
      const width = context.measureText(label).width + 12;
      const x = Math.min(box[0], this.canvas.width - width);
      const y = Math.max(box[1], 18);
      context.fillStyle = color;
      context.fillRect(x, y - 16, width, 16);
      context.fillStyle = "#ffffff";
      context.fillText(label, x + 6, y - 4);
    }
  }

  /* ------------------------------------------------------------ helpers */

  /** Format a zone for the manual-entry inputs. */
  function formatZone(zone) {
    return zone ? zone.map((value) => value.toFixed(3)).join(" ") : "";
  }

  /** Parse "x1 y1 x2 y2" (commas allowed) into a zone, or null. */
  function parseZone(text) {
    const parts = String(text || "").trim().replace(/,/g, " ").split(/\s+/).filter(Boolean);
    if (parts.length !== 4) return null;
    const values = parts.map(Number);
    if (values.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) return null;
    if (values[0] >= values[2] || values[1] >= values[3]) return null;
    return values;
  }

  /**
   * Grab a frame from a local file without uploading it, using the browser's
   * own decoder. Rejects for containers the browser cannot open (.avi, .mkv),
   * where the caller falls back to the server.
   */
  function grabFrameFromFile(file, position = 0.1) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const video = document.createElement("video");
      video.preload = "metadata";
      video.muted = true;
      video.playsInline = true;

      const fail = (message) => {
        URL.revokeObjectURL(url);
        reject(new Error(message));
      };
      const timer = setTimeout(() => fail("Timed out reading the video"), 8000);

      video.onloadedmetadata = () => {
        const target = Number.isFinite(video.duration) && video.duration > 0
          ? Math.min(video.duration * position, video.duration - 0.05)
          : 0;
        video.currentTime = Math.max(0, target);
      };
      video.onseeked = () => {
        clearTimeout(timer);
        try {
          const canvas = document.createElement("canvas");
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          if (!canvas.width || !canvas.height) return fail("The video has no usable frame");
          canvas.getContext("2d").drawImage(video, 0, 0);
          const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
          URL.revokeObjectURL(url);
          resolve(dataUrl);
        } catch (error) {
          fail(error.message);
        }
      };
      video.onerror = () => {
        clearTimeout(timer);
        fail("This browser cannot decode that file");
      };
      video.src = url;
    });
  }

  global.ZoneEditor = ZoneEditor;
  global.zoneUtils = { formatZone, parseZone, grabFrameFromFile };
})(window);
