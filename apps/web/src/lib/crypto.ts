/**
 * Zero-leakage client-side encryption using WebCrypto AES-GCM.
 * Non-extractable device key stored in IndexedDB.
 */

const KEY_DB_NAME = "anjaney_crypto_keys";
const KEY_STORE = "keys";
const KEY_NAME = "device_master_key";

async function getOrGenerateKey(): Promise<CryptoKey> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(KEY_DB_NAME, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore(KEY_STORE);
    };
    request.onsuccess = async () => {
      const db = request.result;
      const tx = db.transaction(KEY_STORE, "readwrite");
      const store = tx.objectStore(KEY_STORE);
      const getReq = store.get(KEY_NAME);

      getReq.onsuccess = async () => {
        if (getReq.result) {
          resolve(getReq.result as CryptoKey);
        } else {
          const newKey = await crypto.subtle.generateKey(
            { name: "AES-GCM", length: 256 },
            false, // non-extractable
            ["encrypt", "decrypt"]
          );
          const putTx = db.transaction(KEY_STORE, "readwrite");
          putTx.objectStore(KEY_STORE).put(newKey, KEY_NAME);
          resolve(newKey);
        }
      };
      getReq.onerror = () => reject(getReq.error);
    };
    request.onerror = () => reject(request.error);
  });
}

export async function encryptData(plainText: string): Promise<{ cipher: string; iv: string }> {
  const key = await getOrGenerateKey();
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const encoded = new TextEncoder().encode(plainText);

  const cipherBuffer = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    key,
    encoded
  );

  return {
    cipher: Array.from(new Uint8Array(cipherBuffer)).map((b) => b.toString(16).padStart(2, "0")).join(""),
    iv: Array.from(iv).map((b) => b.toString(16).padStart(2, "0")).join(""),
  };
}