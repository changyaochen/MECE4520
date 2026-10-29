(() => {
  const root = document.getElementById("convolution-2d-explorer");
  if (!root) return;

  const sourceCanvas = root.querySelector("#convolution-2d-source");
  const outputCanvas = root.querySelector("#convolution-2d-output");
  const description = root.querySelector("#convolution-2d-description");
  const kernelGrid = root.querySelector("#convolution-2d-kernel-grid");
  const formula = root.querySelector("#convolution-2d-formula");
  const playButton = root.querySelector("#convolution-2d-play");
  const resetButton = root.querySelector("#convolution-2d-reset");
  const status = root.querySelector("#convolution-2d-status");
  const filterButtons = [...root.querySelectorAll("[data-filter]")];
  if (!sourceCanvas || !outputCanvas || !description || !kernelGrid || !formula
    || !playButton || !resetButton || !status) return;

  const SIZE = 84;
  const KERNEL_SIZE = 3;
  const OUTPUT_SIZE = SIZE - KERNEL_SIZE + 1;
  const TOTAL_PIXELS = OUTPUT_SIZE ** 2;
  const DISPLAY_SIZE = sourceCanvas.width;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const sourceContext = sourceCanvas.getContext("2d");
  const outputContext = outputCanvas.getContext("2d");
  const pixelCanvas = document.createElement("canvas");
  pixelCanvas.width = SIZE;
  pixelCanvas.height = SIZE;
  const outputPixelCanvas = document.createElement("canvas");
  outputPixelCanvas.width = OUTPUT_SIZE;
  outputPixelCanvas.height = OUTPUT_SIZE;
  const sourcePixelContext = pixelCanvas.getContext("2d", { willReadFrequently: true });
  const outputPixelContext = outputPixelCanvas.getContext("2d");

  const FILTERS = {
    blur: {
      label: "Blur",
      description: "A local mean filter gives equal weight to the nine pixels in a three-by-three patch.",
      weights: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
      response: (value) => value,
      formula: "Each output pixel is the average of its 3 × 3 neighborhood.",
    },
    vertical: {
      label: "Vertical edges",
      description: "This filter responds where intensity changes from left to right, revealing vertical boundaries.",
      weights: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
      response: (value) => Math.min(1, Math.abs(value) / 2.6),
      formula: "The display shows the magnitude of the signed left-to-right change.",
    },
    horizontal: {
      label: "Horizontal edges",
      description: "This filter responds where intensity changes from top to bottom, revealing horizontal boundaries.",
      weights: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
      response: (value) => Math.min(1, Math.abs(value) / 2.6),
      formula: "The display shows the magnitude of the signed top-to-bottom change.",
    },
    sharpen: {
      label: "Sharpen",
      description: "This filter emphasizes a pixel when it differs from its immediate neighbors.",
      weights: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
      response: (value) => Math.max(0, Math.min(1, value)),
      formula: "The center pixel is strengthened while its four direct neighbors are subtracted.",
    },
  };

  const state = {
    filter: "blur",
    source: null,
    output: null,
    progress: 0,
    timer: null,
  };

  function currentFilter() {
    return FILTERS[state.filter];
  }

  function stopPlayback() {
    if (state.timer !== null) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
  }

  function isPlaying() {
    return state.timer !== null;
  }

  function setPixels(context, canvas, values, size) {
    const imageData = context.createImageData(size, size);
    values.forEach((value, index) => {
      const intensity = Math.round(Math.max(0, Math.min(1, value)) * 255);
      const offset = index * 4;
      imageData.data[offset] = intensity;
      imageData.data[offset + 1] = intensity;
      imageData.data[offset + 2] = intensity;
      imageData.data[offset + 3] = 255;
    });
    context.putImageData(imageData, 0, 0);
    return canvas;
  }

  function drawScaled(context, pixelSource, size) {
    context.fillStyle = "#fff";
    context.fillRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
    context.imageSmoothingEnabled = false;
    context.drawImage(pixelSource, 0, 0, size, size, 0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
  }

  function computeOutput() {
    const filter = currentFilter();
    const output = new Float32Array(TOTAL_PIXELS);
    for (let row = 0; row < OUTPUT_SIZE; row += 1) {
      for (let column = 0; column < OUTPUT_SIZE; column += 1) {
        let sum = 0;
        for (let kernelRow = 0; kernelRow < KERNEL_SIZE; kernelRow += 1) {
          for (let kernelColumn = 0; kernelColumn < KERNEL_SIZE; kernelColumn += 1) {
            const sourceIndex = (row + kernelRow) * SIZE + column + kernelColumn;
            sum += state.source[sourceIndex] * filter.weights[kernelRow][kernelColumn];
          }
        }
        output[row * OUTPUT_SIZE + column] = filter.response(sum);
      }
    }
    state.output = output;
  }

  function renderKernel() {
    const filter = currentFilter();
    kernelGrid.replaceChildren();
    filter.weights.flat().forEach((weight) => {
      const cell = document.createElement("span");
      cell.textContent = Number(weight.toFixed(2)).toString();
      kernelGrid.append(cell);
    });
    formula.textContent = filter.formula;
  }

  function renderControls() {
    const displayed = Math.min(state.progress + 1, TOTAL_PIXELS);
    description.textContent = currentFilter().description;
    status.textContent = `${displayed.toLocaleString()} / ${TOTAL_PIXELS.toLocaleString()} output pixels`;
    playButton.textContent = isPlaying() ? "Pause" : "Play all";
    resetButton.disabled = isPlaying();
    filterButtons.forEach((button) => {
      const active = button.dataset.filter === state.filter;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
      button.disabled = isPlaying();
    });
  }

  function activePosition() {
    const index = Math.min(state.progress, TOTAL_PIXELS - 1);
    return { row: Math.floor(index / OUTPUT_SIZE), column: index % OUTPUT_SIZE };
  }

  function renderCanvases() {
    if (!state.source || !state.output) return;
    drawScaled(sourceContext, setPixels(sourcePixelContext, pixelCanvas, state.source, SIZE), SIZE);
    const visibleOutput = new Float32Array(TOTAL_PIXELS).fill(0.94);
    const visibleCount = Math.min(state.progress + 1, TOTAL_PIXELS);
    visibleOutput.set(state.output.slice(0, visibleCount));
    drawScaled(outputContext, setPixels(outputPixelContext, outputPixelCanvas, visibleOutput, OUTPUT_SIZE), OUTPUT_SIZE);

    const { row, column } = activePosition();
    const sourceScale = DISPLAY_SIZE / SIZE;
    const outputScale = DISPLAY_SIZE / OUTPUT_SIZE;
    sourceContext.strokeStyle = "#d24734";
    sourceContext.lineWidth = 2.5;
    sourceContext.strokeRect(column * sourceScale, row * sourceScale, KERNEL_SIZE * sourceScale, KERNEL_SIZE * sourceScale);
    outputContext.strokeStyle = "#d24734";
    outputContext.lineWidth = 2.5;
    outputContext.strokeRect(column * outputScale, row * outputScale, outputScale, outputScale);
  }

  function renderAll() {
    renderControls();
    renderKernel();
    renderCanvases();
  }

  function reset() {
    stopPlayback();
    state.progress = 0;
    renderAll();
  }

  function playAll() {
    if (isPlaying()) {
      stopPlayback();
      renderAll();
      return;
    }
    if (state.progress >= TOTAL_PIXELS - 1) state.progress = 0;
    state.timer = window.setInterval(() => {
      state.progress = Math.min(TOTAL_PIXELS - 1, state.progress + 46);
      if (state.progress >= TOTAL_PIXELS - 1) stopPlayback();
      renderAll();
    }, 34);
    renderAll();
  }

  function initializeSource() {
    const image = new Image();
    image.onload = () => {
      sourcePixelContext.fillStyle = "#fff";
      sourcePixelContext.fillRect(0, 0, SIZE, SIZE);
      sourcePixelContext.drawImage(image, 0, 0, SIZE, SIZE);
      const rgba = sourcePixelContext.getImageData(0, 0, SIZE, SIZE).data;
      state.source = new Float32Array(SIZE ** 2);
      state.source.forEach((_, index) => {
        const offset = index * 4;
        const alpha = rgba[offset + 3] / 255;
        const red = rgba[offset] * alpha + 255 * (1 - alpha);
        const green = rgba[offset + 1] * alpha + 255 * (1 - alpha);
        const blue = rgba[offset + 2] * alpha + 255 * (1 - alpha);
        state.source[index] = (0.299 * red + 0.587 * green + 0.114 * blue) / 255;
      });
      computeOutput();
      renderAll();
    };
    image.src = "../assets/columbia-engineering-logo.png";
  }

  filterButtons.forEach((button) => {
    button.addEventListener("click", () => {
      stopPlayback();
      state.filter = button.dataset.filter;
      state.progress = 0;
      computeOutput();
      renderAll();
    });
  });
  playButton.addEventListener("click", playAll);
  resetButton.addEventListener("click", reset);
  window.addEventListener("pagehide", stopPlayback, { once: true });
  initializeSource();
})();
