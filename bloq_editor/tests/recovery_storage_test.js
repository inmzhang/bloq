// Browser: open recovery_storage_test.htm on an isolated localhost origin.
// Node: pass an IndexedDB factory (e.g. fake-indexeddb) to runRecoveryStorageTests.
// For browser QA copy this module and its HTML into <bundle>/tests/.
const { openRecoveryStore, loadRecovery } = await import(
  typeof document === "undefined" ? "../web/recovery_storage.js" : "../recovery_storage.js"
);

export async function runRecoveryStorageTests(factory = globalThis.indexedDB, objectStorePrototype = globalThis.IDBObjectStore?.prototype) {
  const name = `bloq-recovery-test-${crypto.randomUUID()}`;
  const stores = [];
  let testFailure;
  const check = (value, message) => { if (!value) throw new Error(message); };
  const rejects = async (operation) => {
    try { await operation(); } catch (error) { return error; }
    throw new Error("Expected failure");
  };
  const open = async () => { const store = await openRecoveryStore(factory, name); stores.push(store); return store; };
  try {
    const a = await open();
    check(await a.load() === null, "new database must be empty");
    const unavailableLegacy = await loadRecovery(a, {legacyError:"localStorage disabled"}, (text) => text);
    check(unavailableLegacy.kind === "loaded" && unavailableLegacy.warning, "legacy access failure must warn without disabling IndexedDB");
    await rejects(() => loadRecovery(a, { legacy: "invalid" }, () => { throw new Error("bad legacy"); }));
    check(await a.load() === null, "invalid legacy must not be saved");
    const legacy = '{"version":1,"draft":"unfinished source"}';
    const migrated = await loadRecovery(a, { legacy }, (text) => { JSON.parse(text); return text; });
    check(migrated.prepared === legacy, "legacy migration must preserve payload");
    const original = await a.load();
    check(original.payload === legacy, "migration must save identical source");
    const b = await open();
    check((await b.load()).revision === original.revision, "reopen must retain revision");
    const raced = await Promise.allSettled([
      a.save(original.revision, "first writer"), b.save(original.revision, "second writer"),
    ]);
    check(raced.filter((result) => result.status === "fulfilled").length === 1, "CAS must admit one writer");
    const rejected = raced.find((result) => result.status === "rejected");
    check(rejected.reason.code === "conflict", "loser must report conflict");
    const winner = await a.load();
    await rejects(() => a.save(winner.revision, null));
    check((await a.load()).payload === winner.payload, "failed write must retain current save");
    if (objectStorePrototype) {
      const put = objectStorePrototype.put;
      objectStorePrototype.put = function(value, key) {
        if (this.transaction.db.name === name && key === "current") throw new DOMException("Injected quota failure", "QuotaExceededError");
        return put.call(this, value, key);
      };
      try {
        const error = await rejects(() => a.save(winner.revision, "must not replace"));
        check(error.name === "QuotaExceededError", "storage failure must reach caller");
      } finally { objectStorePrototype.put = put; }
      check((await a.load()).payload === winner.payload, "aborted transaction must retain current save");
    }

    const preferred = await loadRecovery(b, { legacy: "stale legacy", legacyError: "disabled" }, (text) => text);
    check(preferred.prepared === winner.payload, "IndexedDB must win over old or unavailable localStorage");
    const database = await new Promise((resolve, reject) => {
      const request = factory.open(name, 1); request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
    });
    try {
      const previous = await new Promise((resolve, reject) => {
        const request = database.transaction("sessions").objectStore("sessions").get("previous");
        request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
      });
      check(previous.payload === legacy, "previous successful save must be backed up atomically");
    } finally { database.close(); }
    return "PASS: initialize, legacy validation/migration, reopen, concurrent CAS, failed-write preservation, backup, IndexedDB precedence";
  } catch (error) {
    testFailure = error;
    throw error;
  } finally {
    for (const store of stores) store.close();
    try {
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error("Test database cleanup remained blocked after 10 seconds")), 10000);
        const request = factory.deleteDatabase(name);
        request.onsuccess = () => { clearTimeout(timeout); resolve(); };
        request.onerror = () => { clearTimeout(timeout); reject(request.error); };
        // Safari can report blocked while closed connections finish transactions.
        request.onblocked = () => {};
      });
    } catch (cleanupError) {
      if (testFailure) testFailure.message += `; cleanup: ${cleanupError.message}`;
      else throw cleanupError;
    }
  }
}
