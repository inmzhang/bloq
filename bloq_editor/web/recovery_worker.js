import init, { prepare_recovery_in_worker } from "./build/bloq_editor.js";
import { openRecoveryStore, loadRecovery } from "./recovery_storage.js";

try {
  await init();
  const testDatabase = new URL(self.location.href).searchParams.get("testDatabase");
  if (testDatabase && !testDatabase.startsWith("bloq-recovery-test-")) throw new Error("Invalid isolated test database name");
  const store = await openRecoveryStore(globalThis.indexedDB, testDatabase ?? undefined);
  // Serial requests and transactional CAS protect both intra- and inter-tab saves.
  let queue = Promise.resolve();
  self.onmessage = ({ data }) => {
    queue = queue.then(async () => {
      try {
        if (data.operation === "load") {
          self.postMessage(await loadRecovery(store, data, prepare_recovery_in_worker));
        } else if (data.operation === "save") {
          const saved = await store.save(data.expectedRevision, JSON.stringify(data.snapshot));
          self.postMessage({ kind: "saved", revision: saved.revision });
        } else {
          throw new Error("Unknown recovery operation");
        }
      } catch (error) {
        self.postMessage({ kind: "error", operation: data.operation,
          conflict: error?.code === "conflict", message: String(error) });
      }
    });
  };
  self.postMessage({ kind: "ready" });
} catch (error) {
  self.postMessage({ kind: "error", operation: "load", message: `Cannot start recovery: ${error}` });
}
