import { Buffer } from "node:buffer"
import { createOpenAI } from "@ai-sdk/openai"
import { generateText, streamText } from "ai"
import { describe, expect, test, vi } from "vitest"
import { toModelMessages } from "../src/chat-messages.js"

const png =
	"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="

describe("inline image conversion through the AI SDK", () => {
	test.each([
		false,
		true,
	])("forwards images with streaming=%s", async (stream) => {
		const fetch = vi.fn(async (_url: unknown, init?: RequestInit) => {
			const body = JSON.parse(String(init?.body))
			expect(body.input[0].content).toEqual([
				{ type: "input_text", text: "Describe these images" },
				{ type: "input_image", image_url: `data:image/png;base64,${png}` },
				{ type: "input_image", image_url: "https://example.com/image.png" },
			])
			return new Response(
				[
					'data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5.4","created_at":1}}',
					"",
					'data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"hello"}',
					"",
					'data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.4","created_at":1,"status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"hello","annotations":[]}]}],"usage":{"input_tokens":3,"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0}}}}',
					"",
					"",
				].join("\n"),
				{ headers: { "Content-Type": "text/event-stream" } },
			)
		})
		const model = createOpenAI({ apiKey: "test", fetch }).responses("gpt-5.4")
		const messages = toModelMessages([
			{
				role: "user",
				content: [
					{ type: "text", text: "Describe these images" },
					{
						type: "image_url",
						image_url: { url: `data:image/png;base64,${png}` },
					},
					{
						type: "image_url",
						image_url: { url: "https://example.com/image.png" },
					},
				],
			},
		])
		if (stream) {
			const result = streamText({ model, messages })
			let text = ""
			for await (const chunk of result.textStream) text += chunk
			expect(text).toBe("hello")
		} else {
			// The OAuth transport always requests upstream SSE, even for JSON clients.
			const streamingFetch = vi.fn(async (url: unknown, init?: RequestInit) => {
				const response = await fetch(url, init)
				const lines = (await response.text()).split("\n")
				const completed = lines.find((line) =>
					line.includes('"response.completed"'),
				)
				return Response.json(JSON.parse(completed?.slice(6) ?? "{}").response)
			})
			const jsonModel = createOpenAI({
				apiKey: "test",
				fetch: streamingFetch,
			}).responses("gpt-5.4")
			expect((await generateText({ model: jsonModel, messages })).text).toBe(
				"hello",
			)
		}
		expect(fetch).toHaveBeenCalledTimes(1)
	})

	test.each([
		"png",
		"jpeg",
		"webp",
	])("decodes %s and preserves its media type", (type) => {
		const [message] = toModelMessages([
			{
				role: "user",
				content: [
					{
						type: "image_url",
						image_url: { url: `data:image/${type};base64,${png}` },
					},
				],
			},
		])
		expect(message?.content).toEqual([
			{
				type: "image",
				image: Buffer.from(png, "base64"),
				mediaType: `image/${type}`,
			},
		])
	})

	test("preserves text-only conversations", () => {
		expect(toModelMessages([{ role: "user", content: "hello" }])).toEqual([
			{ role: "user", content: "hello" },
		])
	})
})
