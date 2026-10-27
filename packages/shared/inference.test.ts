import { beforeEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import { AnthropicInferenceClient, OpenAIInferenceClient } from "./inference";
import type {
  AnthropicInferenceConfig,
  OpenAIInferenceConfig,
} from "./inference";

const capturedBodies: Record<string, unknown>[] = [];
let anthropicResponse: Record<string, unknown> = {};
const tagSchema = z.object({ tags: z.array(z.string()) });

vi.mock("openai", () => {
  const OpenAIMock = vi.fn().mockImplementation(function () {
    return {
      chat: {
        completions: {
          create: vi.fn(async (body: Record<string, unknown>) => {
            capturedBodies.push(body);
            return {
              choices: [{ message: { content: "{}" } }],
              usage: { total_tokens: 1 },
            };
          }),
        },
      },
    };
  });

  return { default: OpenAIMock };
});

vi.mock("@anthropic-ai/sdk", () => {
  const AnthropicMock = vi.fn().mockImplementation(function () {
    return {
      messages: {
        create: vi.fn(async (body: Record<string, unknown>) => {
          capturedBodies.push(body);
          return anthropicResponse;
        }),
      },
    };
  });

  return { default: AnthropicMock };
});

vi.mock("openai/helpers/zod", () => ({
  zodResponseFormat: (schema: unknown, name: string) => ({
    type: "json_schema",
    json_schema: { name, schema },
  }),
}));

function makeConfig(
  outputSchema: OpenAIInferenceConfig["outputSchema"],
): OpenAIInferenceConfig {
  return {
    apiKey: "test-key",
    textModel: "test-text-model",
    imageModel: "test-image-model",
    contextLength: 2048,
    maxOutputTokens: 1024,
    useMaxCompletionTokens: false,
    outputSchema,
  };
}

describe("OpenAIInferenceClient response_format", () => {
  beforeEach(() => {
    capturedBodies.length = 0;
  });

  it("omits response_format for schema-less text inference in json mode", async () => {
    const client = new OpenAIInferenceClient(makeConfig("json"));

    await client.inferFromText("summarize this text", { schema: null });

    expect(capturedBodies).toHaveLength(1);
    expect(capturedBodies[0].response_format).toBeUndefined();
  });

  it("keeps json_object for schema-backed text inference in json mode", async () => {
    const client = new OpenAIInferenceClient(makeConfig("json"));

    await client.inferFromText("infer tags as json", { schema: tagSchema });

    expect(capturedBodies).toHaveLength(1);
    expect(capturedBodies[0].response_format).toEqual({ type: "json_object" });
  });

  it("omits response_format for schema-less image inference in json mode", async () => {
    const client = new OpenAIInferenceClient(makeConfig("json"));

    await client.inferFromImage("describe this image", "image/png", "BASE64", {
      schema: null,
    });

    expect(capturedBodies).toHaveLength(1);
    expect(capturedBodies[0].response_format).toBeUndefined();
  });

  it("keeps structured response_format for schema-backed text inference in structured mode", async () => {
    const client = new OpenAIInferenceClient(makeConfig("structured"));

    await client.inferFromText("infer tags", { schema: tagSchema });

    expect(capturedBodies).toHaveLength(1);
    expect(capturedBodies[0].response_format).toMatchObject({
      type: "json_schema",
      json_schema: { name: "schema" },
    });
  });
});

function makeAnthropicConfig(
  overrides: Partial<AnthropicInferenceConfig> = {},
): AnthropicInferenceConfig {
  return {
    apiKey: "test-key",
    textModel: "test-text-model",
    imageModel: "test-image-model",
    maxOutputTokens: 1024,
    outputSchema: "structured",
    ...overrides,
  };
}

describe("AnthropicInferenceClient", () => {
  beforeEach(() => {
    capturedBodies.length = 0;
    anthropicResponse = {
      content: [{ type: "text", text: '{"tags":["a"]}' }],
      stop_reason: "end_turn",
      usage: { input_tokens: 3, output_tokens: 4 },
    };
  });

  it("sends a json_schema output format in structured mode", async () => {
    const client = new AnthropicInferenceClient(makeAnthropicConfig());

    const res = await client.inferFromText("infer tags", { schema: tagSchema });

    expect(res).toEqual({ response: '{"tags":["a"]}', totalTokens: 7 });
    expect(capturedBodies[0].output_config).toMatchObject({
      format: { type: "json_schema" },
    });
  });

  it("omits output_config without a schema or effort", async () => {
    const client = new AnthropicInferenceClient(makeAnthropicConfig());

    await client.inferFromText("summarize", { schema: null });

    expect(capturedBodies[0].output_config).toBeUndefined();
  });

  it("passes effort and skips the format outside structured mode", async () => {
    const client = new AnthropicInferenceClient(
      makeAnthropicConfig({ outputSchema: "json", effort: "low" }),
    );

    await client.inferFromText("infer tags", { schema: tagSchema });

    expect(capturedBodies[0].output_config).toEqual({ effort: "low" });
  });

  it("sends images as base64 blocks with the image model", async () => {
    const client = new AnthropicInferenceClient(makeAnthropicConfig());

    await client.inferFromImage("describe", "image/png", "BASE64", {});

    expect(capturedBodies[0].model).toBe("test-image-model");
    expect(capturedBodies[0].messages).toEqual([
      {
        role: "user",
        content: [
          {
            type: "image",
            source: { type: "base64", media_type: "image/png", data: "BASE64" },
          },
          { type: "text", text: "describe" },
        ],
      },
    ]);
  });

  it("rejects unsupported image types", async () => {
    const client = new AnthropicInferenceClient(makeAnthropicConfig());

    await expect(
      client.inferFromImage("describe", "image/bmp", "BASE64", {}),
    ).rejects.toThrow("image/bmp");
  });

  it("throws on refusals", async () => {
    anthropicResponse = {
      content: [],
      stop_reason: "refusal",
      stop_details: { type: "refusal", category: "cyber" },
      usage: { input_tokens: 1, output_tokens: 0 },
    };
    const client = new AnthropicInferenceClient(makeAnthropicConfig());

    await expect(client.inferFromText("x", {})).rejects.toThrow("cyber");
  });
});
