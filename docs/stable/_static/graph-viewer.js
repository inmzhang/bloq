// Filter gallery cards using the same native category tags as Bloq Editor.
document.querySelectorAll(".bloq-gallery-filters").forEach(filters => {
  const cards = [...document.querySelectorAll(".bloq-gallery-card")];
  const count = document.querySelector(".bloq-gallery-count");
  const select = button => {
    const category = button.dataset.category;
    filters.querySelectorAll("button").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
    cards.forEach(card => {
      card.hidden = category !== "all" && !card.dataset.categories.split(" ").includes(category);
    });
    count.textContent = `${cards.filter(card => !card.hidden).length} examples`;
  };
  filters.querySelectorAll("button").forEach(button => button.addEventListener("click", () => select(button)));
  select(filters.querySelector('[data-category="all"]'));
  filters.hidden = false;
});

// Enhance exported SVG images without changing their downloadable contents.
document.querySelectorAll(".bloq-ir-graph").forEach(panel => {
  const image = panel.querySelector("img");
  if (!image) return;
  const initialize = () => {
    if (!image.naturalWidth || panel.classList.contains("bloq-panzoom")) return;
    panel.classList.add("bloq-panzoom");
    panel.removeAttribute("tabindex");
    const toolbar = document.createElement("div");
    toolbar.className = "bloq-graph-toolbar";
    const viewport = document.createElement("div");
    viewport.className = "bloq-graph-viewport";
    viewport.tabIndex = 0;
    viewport.setAttribute("aria-label", "Graph canvas: drag to pan, scroll or use plus and minus to zoom, Home to fit");
    image.draggable = false;
    // Dimensionless SVGs otherwise stretch to the container before scaling.
    image.style.width = `${image.naturalWidth}px`;
    image.style.height = `${image.naturalHeight}px`;
    viewport.append(image);
    panel.replaceChildren(toolbar, viewport);
    const percentage = document.createElement("output");
    percentage.setAttribute("aria-label", "Zoom level");
    let scale = 1, fitScale = 1, x = 0, y = 0, drag;
    const draw = () => {
      image.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
      percentage.textContent = `${Math.round(scale * 100)}%`;
    };
    const fit = () => {
      fitScale = Math.min(viewport.clientWidth / image.naturalWidth, viewport.clientHeight / image.naturalHeight);
      scale = fitScale;
      x = (viewport.clientWidth - image.naturalWidth * scale) / 2;
      y = (viewport.clientHeight - image.naturalHeight * scale) / 2;
      draw();
    };
    const zoom = (factor, cx = viewport.clientWidth / 2, cy = viewport.clientHeight / 2) => {
      const next = Math.max(fitScale / 10, Math.min(fitScale * 20, scale * factor));
      x = cx - (cx - x) * next / scale;
      y = cy - (cy - y) * next / scale;
      scale = next;
      draw();
    };
    const button = (label, text, action) => {
      const control = document.createElement("button");
      control.type = "button";
      control.textContent = text;
      control.setAttribute("aria-label", label);
      control.addEventListener("click", action);
      toolbar.append(control);
    };
    button("Zoom out", "−", () => zoom(1 / 1.25));
    button("Zoom in", "+", () => zoom(1.25));
    button("Fit graph", "Fit", fit);
    toolbar.append(percentage);
    const hint = document.createElement("span");
    hint.textContent = "Drag to pan · Scroll to zoom";
    toolbar.append(hint);
    const download = document.createElement("a");
    download.className = "bloq-graph-download";
    download.href = image.currentSrc || image.src;
    download.download = "";
    download.textContent = "Download SVG";
    toolbar.append(download);
    viewport.addEventListener("wheel", event => {
      event.preventDefault();
      const rect = viewport.getBoundingClientRect();
      zoom(Math.exp(-Math.max(-100, Math.min(100, event.deltaY)) * 0.01), event.clientX - rect.left, event.clientY - rect.top);
    }, { passive: false });
    viewport.addEventListener("pointerdown", event => {
      if (event.button !== 0 || drag) return;
      event.preventDefault();
      viewport.focus({ preventScroll: true });
      viewport.setPointerCapture(event.pointerId);
      drag = { id: event.pointerId, x: event.clientX, y: event.clientY };
      viewport.classList.add("dragging");
    });
    viewport.addEventListener("pointermove", event => {
      if (drag?.id !== event.pointerId) return;
      x += event.clientX - drag.x;
      y += event.clientY - drag.y;
      drag.x = event.clientX;
      drag.y = event.clientY;
      draw();
    });
    viewport.addEventListener("lostpointercapture", () => {
      drag = undefined;
      viewport.classList.remove("dragging");
    });
    viewport.addEventListener("keydown", event => {
      if (["+", "=", "-", "Home", "0", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) event.preventDefault();
      else return;
      if (event.key === "+" || event.key === "=") zoom(1.25);
      else if (event.key === "-") zoom(1 / 1.25);
      else if (event.key === "Home" || event.key === "0") fit();
      else {
        x += event.key === "ArrowLeft" ? 40 : event.key === "ArrowRight" ? -40 : 0;
        y += event.key === "ArrowUp" ? 40 : event.key === "ArrowDown" ? -40 : 0;
        draw();
      }
    });
    new ResizeObserver(fit).observe(viewport);
    fit();
  };
  if (image.complete) initialize();
  else image.addEventListener("load", initialize, { once: true });
});
