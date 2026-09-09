import { beforeEach, describe, expect, test } from "bun:test";
import { createOpenAIChatAdapter } from "../../../src/adapters/openai-chat";
import {
  hasShrinkableOpenAIChatImages,
  normalizeOpenAIChatImages,
  OPENAI_CHAT_IMAGE_BASE64_BUDGET,
} from "../../../src/adapters/openai-chat-images";
import { resetNormalizeStateForTests, TIER_SPECS, type EncodeFn } from "../../../src/adapters/anthropic-image-normalize";
import type { OcxMessage, OcxParsedRequest, OcxProviderConfig } from "../../../src/types";

// Issue #4112 follow-up: chat-completions providers such as GitHub Copilot reject a body
// over roughly 5.2MB with a bare 413 and no diagnostic content. Nothing downstream of the
// adapter can shrink a built request, so inline image bytes are normalized here. Images are
// never dropped on this wire: there is no downstream guard that would re-attach them.

const provider: OcxProviderConfig = {
  adapter: "openai-chat",
  baseUrl: "https://api.githubcopilot.com",
  apiKey: "sk-test",
  authMode: "key",
};

const ONE_PX_PNG =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

async function realPngB64(width: number, height: number): Promise<string> {
  const buf = await new Bun.Image(Buffer.from(ONE_PX_PNG, "base64")).resize(width, height).png().toBuffer();
  return Buffer.from(buf).toString("base64");
}

/**
 * A flat-colour PNG compresses to almost nothing, so budget behaviour needs incompressible
 * pixels. Deterministic noise is written as an uncompressed BMP and converted, which keeps
 * the fixture in-repo and the encoded size realistic.
 */
async function noisyPngB64(width: number, height: number): Promise<string> {
  const rowSize = width * 3 + ((4 - ((width * 3) % 4)) % 4);
  const pixelBytes = rowSize * height;
  const bmp = Buffer.alloc(54 + pixelBytes);
  bmp.write("BM", 0);
  bmp.writeUInt32LE(bmp.length, 2);
  bmp.writeUInt32LE(54, 10);
  bmp.writeUInt32LE(40, 14);
  bmp.writeInt32LE(width, 18);
  bmp.writeInt32LE(height, 22);
  bmp.writeUInt16LE(1, 26);
  bmp.writeUInt16LE(24, 28);
  bmp.writeUInt32LE(pixelBytes, 34);
  let seed = 0x2545f491;
  for (let y = 0; y < height; y++) {
    let offset = 54 + y * rowSize;
    for (let x = 0; x < width; x++) {
      seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; seed >>>= 0;
      bmp[offset++] = seed & 0xff;
      bmp[offset++] = (seed >>> 8) & 0xff;
      bmp[offset++] = (seed >>> 16) & 0xff;
    }
  }
  const png = await new Bun.Image(bmp).png().toBuffer();
  return Buffer.from(png).toString("base64");
}


function dataUrl(b64: string, mediaType = "image/png"): string {
  return `data:${mediaType};base64,${b64}`;
}

interface ChatPart {
  type: string;
  text?: string;
  image_url?: { url: string };
}

interface ChatMsg {
  role: string;
  content?: string | ChatPart[];
}

function parsedWith(messages: OcxMessage[]): OcxParsedRequest {
  return {
    modelId: "claude-opus-5",
    context: { messages },
    stream: false,
    options: {},
  } as unknown as OcxParsedRequest;
}

function imageMessage(urls: string[], text = "what is this"): OcxMessage {
  return {
    role: "user",
    content: [
      { type: "text", text },
      ...urls.map(url => ({ type: "image" as const, imageUrl: url })),
    ],
    timestamp: 0,
  } as unknown as OcxMessage;
}

function wireMessages(body: string): ChatMsg[] {
  return (JSON.parse(body) as { messages: ChatMsg[] }).messages;
}

function imageParts(messages: ChatMsg[]): ChatPart[] {
  return messages.flatMap(m => (Array.isArray(m.content) ? m.content : [])).filter(p => p.type === "image_url");
}

/** Deterministic encoder: output size is a function of the tier's max edge. */
const sizedEncoder = (sizeFor: (maxEdge: number) => number): EncodeFn =>
  (_input, spec) => Promise.resolve({
    data: "A".repeat(sizeFor(spec.maxEdge)),
    mediaType: "image/jpeg",
  });

describe("openai-chat inline image normalization", () => {
  beforeEach(() => resetNormalizeStateForTests());

  test("a text-only turn builds synchronously and is unchanged", () => {
    const request = createOpenAIChatAdapter(provider).buildRequest(
      parsedWith([{ role: "user", content: "hello", timestamp: 0 } as unknown as OcxMessage]),
    );
    expect(request).not.toBeInstanceOf(Promise);
    const messages = wireMessages((request as { body: string }).body);
    expect(messages.at(-1)?.content).toBe("hello");
  });

  test("an image turn under the budget stays synchronous and keeps its exact bytes", async () => {
    const small = await realPngB64(8, 8);
    expect(hasShrinkableOpenAIChatImages([
      { role: "user", content: [{ type: "image_url", image_url: { url: dataUrl(small) } }] },
    ])).toBe(false);

    const request = createOpenAIChatAdapter(provider).buildRequest(parsedWith([imageMessage([dataUrl(small)])]));
    expect(request).not.toBeInstanceOf(Promise);
    const parts = imageParts(wireMessages((request as { body: string }).body));
    expect(parts).toHaveLength(1);
    expect(parts[0]?.image_url?.url).toBe(dataUrl(small));
  });

  test("an oversized turn is re-encoded through the adapter and keeps every image", async () => {
    const big = await noisyPngB64(1000, 1000);
    const urls = Array.from({ length: 4 }, () => dataUrl(big));
    expect(hasShrinkableOpenAIChatImages([
      { role: "user", content: urls.map(url => ({ type: "image_url", image_url: { url } })) },
    ])).toBe(true);

    const built = createOpenAIChatAdapter(provider).buildRequest(parsedWith([imageMessage(urls)]));
    expect(built).toBeInstanceOf(Promise);
    const { body } = await (built as Promise<{ body: string }>);
    const messages = wireMessages(body);
    const parts = imageParts(messages);

    expect(parts).toHaveLength(4);
    const total = parts.reduce((sum, p) => sum + (p.image_url?.url.split(",")[1]?.length ?? 0), 0);
    expect(total).toBeLessThanOrEqual(OPENAI_CHAT_IMAGE_BASE64_BUDGET);
    for (const part of parts) expect(part.image_url?.url.startsWith("data:image/")).toBe(true);
    // The caption survives alongside the images.
    expect(JSON.stringify(messages)).toContain("what is this");
  });

  test("terminal overflow keeps images attached instead of dropping the oldest", async () => {
    const terminalEdge = TIER_SPECS[TIER_SPECS.length - 1]!.maxEdge;
    const small = await realPngB64(40, 40);
    const messages = [{
      role: "user",
      content: Array.from({ length: 6 }, () => ({
        type: "image_url",
        image_url: { url: dataUrl(small) },
      })),
    }];
    // Every tier, including the floor, still exceeds the budget on its own.
    await normalizeOpenAIChatImages(messages, {
      encode: sizedEncoder(() => OPENAI_CHAT_IMAGE_BASE64_BUDGET),
      validate: () => Promise.resolve(),
    });
    const parts = imageParts(messages as ChatMsg[]);
    expect(parts).toHaveLength(6);
    expect(terminalEdge).toBeGreaterThan(0);
    for (const part of parts) expect(part.image_url?.url).toContain("base64,");
  });

  test("a remote https image is left untouched", async () => {
    const messages = [{
      role: "user",
      content: [{ type: "image_url", image_url: { url: "https://example.com/cat.png" } }],
    }];
    expect(hasShrinkableOpenAIChatImages(messages)).toBe(false);
    await normalizeOpenAIChatImages(messages);
    expect(imageParts(messages as ChatMsg[])[0]?.image_url?.url).toBe("https://example.com/cat.png");
  });

  test("an undecodable image keeps its original url rather than being dropped", async () => {
    const corrupt = dataUrl("!!!!not-base64-image!!!!");
    const messages = [{
      role: "user",
      content: [{ type: "image_url", image_url: { url: corrupt } }],
    }];
    await normalizeOpenAIChatImages(messages, {
      encode: () => Promise.reject(new Error("undecodable")),
      validate: () => Promise.reject(new Error("undecodable")),
    });
    expect(imageParts(messages as ChatMsg[])[0]?.image_url?.url).toBe(corrupt);
  });

  test("malformed message shapes neither throw nor lose parts", async () => {
    const messages: unknown[] = [
      null,
      "not-a-message",
      { role: "user" },
      { role: "user", content: "plain text" },
      { role: "user", content: [{ type: "image_url" }, { type: "image_url", image_url: {} }] },
    ];
    const before = JSON.stringify(messages);
    expect(hasShrinkableOpenAIChatImages(messages)).toBe(false);
    await normalizeOpenAIChatImages(messages);
    expect(JSON.stringify(messages)).toBe(before);
    await normalizeOpenAIChatImages(undefined);
    await normalizeOpenAIChatImages("nonsense");
  });

  test("imageTierBias from incoming meta reaches the normalizer", async () => {
    const big = await noisyPngB64(1000, 1000);
    const urls = Array.from({ length: 4 }, () => dataUrl(big));
    const adapter = createOpenAIChatAdapter(provider);

    const build = async (imageTierBias?: number) => {
      resetNormalizeStateForTests();
      const built = adapter.buildRequest(parsedWith([imageMessage(urls)]), {
        headers: new Headers(),
        ...(imageTierBias !== undefined ? { imageTierBias } : {}),
      });
      const request = await (built as Promise<{ body: string }>);
      return imageParts(wireMessages(request.body))
        .reduce((sum, part) => sum + (part.image_url?.url.length ?? 0), 0);
    };

    expect(await build(3)).toBeLessThan(await build());
  });
});
