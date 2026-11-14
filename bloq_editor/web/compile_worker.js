import init, { compile_in_worker, simplify_zx_in_worker, analyze_blog_in_worker } from "./build/bloq_editor.js";

try {
  await init();
  self.onmessage = ({ data }) => {
    try {
      if (data.operation === "analyze-blog") {
        const analysisJson = analyze_blog_in_worker(data.source, BigInt(data.revision), data.baseline ?? undefined);
        self.postMessage({ kind: "complete", analysisJson });
        return;
      }
      if (data.operation === "simplify-zx") {
        const graphJson = simplify_zx_in_worker(data.source, BigInt(data.seed));
        self.postMessage({ kind: "complete", graphJson });
        return;
      }
      const [bytes, durationSeconds] = compile_in_worker(
        data.source,
        data.codeDistance,
        data.prepareTWithMpps,
      );
      self.postMessage({ kind: "complete", bytes, durationSeconds }, [bytes.buffer]);
    } catch (error) {
      self.postMessage({ kind: "error", message: String(error) });
    }
  };
  self.postMessage({ kind: "ready" });
} catch (error) {
  self.postMessage({ kind: "error", message: `Cannot start compiler: ${error}` });
}
