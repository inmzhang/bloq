// Run in the recovery worker: no main-thread JSON or synchronous storage writes.
const DATABASE = "bloq-editor-recovery-v1";
const STORE = "sessions";

function conflict() {
  return Object.assign(new Error("Another tab changed this session. Export your work before reloading."), { code: "conflict" });
}

function validate(record) {
  if (record !== undefined &&
      (typeof record?.revision !== "string" || typeof record?.payload !== "string")) {
    throw new Error("Saved recovery record is invalid; it has been left unchanged.");
  }
  return record ?? null;
}

export async function openRecoveryStore(factory = globalThis.indexedDB, name = DATABASE) {
  if (!factory) throw new Error("IndexedDB is unavailable");
  const database = await new Promise((resolve, reject) => {
    let settled = false;
    const request = factory.open(name, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onerror = () => { settled = true; reject(request.error); };
    request.onblocked = () => { settled = true; reject(new Error("Recovery database upgrade is blocked by another tab")); };
    request.onsuccess = () => {
      if (settled) { request.result.close(); return; }
      settled = true;
      resolve(request.result);
    };
  });
  database.onversionchange = () => database.close();

  return {
    load() {
      return new Promise((resolve, reject) => {
        const transaction = database.transaction(STORE, "readonly");
        const request = transaction.objectStore(STORE).get("current");
        let record;
        request.onsuccess = () => {
          try { record = validate(request.result); }
          catch (error) { reject(error); }
        };
        transaction.oncomplete = () => resolve(record);
        transaction.onabort = () => reject(transaction.error ?? new Error("Recovery read aborted"));
        transaction.onerror = () => {}; // onabort reports the transaction failure.
      });
    },

    // Read/check/write in ONE transaction: concurrent tabs cannot both pass CAS.
    // The previous complete save moves to a backup in that same transaction.
    save(expectedRevision, payload) {
      return new Promise((resolve, reject) => {
        if (typeof payload !== "string") {
          reject(new Error("Recovery payload must be serialized before storage"));
          return;
        }
        const transaction = database.transaction(STORE, "readwrite");
        const store = transaction.objectStore(STORE);
        const request = store.get("current");
        let saved, failure;
        request.onsuccess = () => {
          try {
            const current = validate(request.result);
            if ((current?.revision ?? null) !== expectedRevision) throw conflict();
            if (current?.payload === payload) { saved = current; return; }
            saved = { revision: crypto.randomUUID(), payload };
            if (current) store.put(current, "previous");
            store.put(saved, "current");
          } catch (error) {
            failure = error;
            transaction.abort();
          }
        };
        transaction.oncomplete = () => resolve(saved);
        transaction.onabort = () => reject(failure ?? transaction.error ?? new Error("Recovery write aborted"));
        transaction.onerror = () => {};
      });
    },

    close() { database.close(); },
  };
}

// Validate legacy data before the first IndexedDB write. An existing IndexedDB
// save wins; localStorage remains untouched for manual recovery/older versions.
export async function loadRecovery(store, request, prepare) {
  let record = await store.load();
  let prepared = null;
  const warning = !record && request.legacyError
    ? `Legacy recovery could not be checked: ${request.legacyError}. IndexedDB remains available; legacy storage was not changed.`
    : null;
  if (!record && request.legacy != null) {
    prepared = prepare(request.legacy);
    record = await store.save(null, request.legacy);
  } else if (record) {
    prepared = prepare(record.payload);
  }
  return { kind: "loaded", revision: record?.revision ?? null, prepared, warning };
}
