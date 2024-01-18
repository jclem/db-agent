import { EventEmitter } from "node:events";
import { OpenAI } from "openai";
import type {
  ChatCompletionChunk,
  ChatCompletionMessageParam,
} from "openai/resources/index.mjs";
import { z } from "zod";

const Message = z.object({
  role: z.string(),
  name: z.string().optional(),
  content: z.string(),
});

type Message = z.infer<typeof Message>;

const Input = z.object({
  messages: z.array(Message),
});

const openai = new OpenAI({ apiKey: Bun.env.OPENAI_API_KEY });
const pscaleTokenID = Bun.env.PLANETSCALE_API_TOKEN_ID;
const pscaleToken = Bun.env.PLANETSCALE_API_TOKEN;
const pscaleAuth = `${pscaleTokenID}:${pscaleToken}`;
const pscaleOrg = Bun.env.PLANETSCALE_ORG;

Bun.serve({
  port: Bun.env.PORT ?? "3000",

  async fetch(request) {
    // Do nothing with the OAuth callback, for now. Just return a 200.
    if (new URL(request.url).pathname === "/oauth/callback") {
      return Response.json({ ok: true }, { status: 200 });
    }

    // Parsing with Zod strips unknown Copilot-specific fields in the request
    // body, which cause OpenAI errors if they're included.
    const json = await request.json();
    console.debug("received", json);
    const input = Input.safeParse(json);

    if (!input.success) {
      return Response.json({ error: "Bad request" }, { status: 400 });
    }

    const responseStream = handleRequest(input.data.messages);
    return new Response(responseStream, {
      headers: {
        "content-type": "text/event-stream",
      },
    });
  },
});

function handleRequest(messages: Message[]) {
  const ps = new PscaleClient();

  const runner = openai.beta.chat.completions.runTools({
    model: "gpt-4-1106-preview",
    stream: true,
    messages: messages as ChatCompletionMessageParam[],
    tools: [
      {
        type: "function",
        function: {
          function: function listDatabases() {
            return ps.fetch("/databases");
          },
          description: "List all PlanetScale databases.",
          parameters: {
            type: "object",
            properties: {},
            required: [],
          },
        },
      },
      {
        type: "function",
        function: {
          function: function getDatabase(args: { name: string }) {
            return ps.fetch(`/databases/${args.name}`);
          },
          description: "Get information about a single PlanetScale database.",
          parse: JSON.parse,
          parameters: {
            type: "object",
            properties: {
              name: {
                type: "string",
                description: "The name of the database to get.",
              },
            },
            required: ["name"],
          },
        },
      },
      {
        type: "function",
        function: {
          function: function listBranches(args: { name: string }) {
            return ps.fetch(`/databases/${args.name}/branches`);
          },
          description: "List all database branches for the given database.",
          parse: JSON.parse,
          parameters: {
            type: "object",
            properties: {
              name: {
                type: "string",
                description: "The name of the database to get branches for.",
              },
            },
            required: ["name"],
          },
        },
      },
      {
        type: "function",
        function: {
          function: function getBranch(args: { name: string; branch: string }) {
            return ps.fetch(`/databases/${args.name}/branches/${args.branch}`);
          },
          description: "Get a specific branch of a database.",
          parse: JSON.parse,
          parameters: {
            type: "object",
            properties: {
              name: {
                type: "string",
                description: "The name of the database to get branches for.",
              },
              branch: {
                type: "string",
                description: "The name of the branch to get.",
              },
            },
            required: ["name", "branch"],
          },
        },
      },
      {
        type: "function",
        function: {
          function: function getBranchSchema(args: {
            name: string;
            branch: string;
          }) {
            return ps.fetch(
              `/databases/${args.name}/branches/${args.branch}/schema`,
            );
          },
          description: "Get the schema of a specific branch of a database.",
          parse: JSON.parse,
          parameters: {
            type: "object",
            properties: {
              name: {
                type: "string",
                description:
                  "The name of the database to get a branch's schema for.",
              },
              branch: {
                type: "string",
                description: "The name of the branch whose schema to get.",
              },
            },
            required: ["name", "branch"],
          },
        },
      },
      {
        type: "function",
        function: {
          function: function listDeployRequests(args: { name: string }) {
            return ps.fetch(`/databases/${args.name}/deploy-requests`);
          },
          description: "List all deploy requests for the given database.",
          parse: JSON.parse,
          parameters: {
            type: "object",
            properties: {
              name: {
                type: "string",
                description:
                  "The name of the database to get deploy requests for.",
              },
            },
            required: ["name"],
          },
        },
      },
      {
        type: "function",
        function: {
          function: function getDeployRequest(args: {
            name: string;
            requestID: number;
          }) {
            return ps.fetch(
              `/databases/${args.name}/deploy-requests/${args.requestID}`,
            );
          },
          description: "Get a specific deploy request for a database.",
          parse: JSON.parse,
          parameters: {
            type: "object",
            properties: {
              name: {
                type: "string",
                description:
                  "The name of the database to get a deploy request for.",
              },
              requestID: {
                type: "number",
                description: "The ID of the deploy request to get.",
              },
            },
            required: ["name", "requestID"],
          },
        },
      },
    ],
  });

  // Proxy the OpenAI API response right back to the extensibility platform.
  return new ReadableStream({
    async start(controller) {
      runner.on("end", () => {
        controller.close();
      });

      runner.on("chunk", (chunk) => {
        if (chunk.choices.at(0)?.finish_reason === "tool_calls") {
          return;
        }

        const data = `data: ${JSON.stringify(chunk)}\n\n`;
        controller.enqueue(data);
      });

      ps.on("update", (update) => {
        const chunk: ChatCompletionChunk = {
          id: "chunk",
          object: "chat.completion.chunk",
          created: new Date().getDate(),
          model: "gpt-4-1106-preview",
          choices: [
            { index: 0, delta: { content: update }, finish_reason: null },
          ],
        };

        const data = `data: ${JSON.stringify(chunk)}\n\n`;
        controller.enqueue(data);
      });
    },

    cancel() {
      runner.abort();
    },
  });
}

class PscaleClient {
  readonly #emitter = new EventEmitter();

  async fetch(path: string, init?: RequestInit) {
    const method = init?.method ?? "GET";
    this.#emitter.emit("update", `\`${method} ${path}\`...`);
    console.debug(`Fetching ${method} ${path}...`);

    const resp = await fetch(
      `https://api.planetscale.com/v1/organizations/${pscaleOrg}${path}`,
      {
        headers: {
          Authorization: pscaleAuth,
          ...init?.headers,
        },
        ...init,
        method,
      },
    );

    if (!resp.ok) {
      this.#emitter.emit("update", `\`Error ${resp.status}\`  \n`);

      return {
        ok: false,
        status: resp.status,
        statusText: resp.statusText,
        body: await resp.json(),
      };
    }

    this.#emitter.emit("update", "`OK`  \n\n");

    return {
      ok: true,
      data: await resp.json(),
    };
  }

  on(event: "update", listener: (content: string) => void) {
    this.#emitter.on(event, listener);
  }

  update(content: string) {
    this.#emitter.emit("update", `${content}  \n\n`);
  }
}
