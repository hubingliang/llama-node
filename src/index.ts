import {
    LLama as LLamaNode,
    LLamaConfig,
    LLamaArguments,
} from "@llama-node/core";

export interface ChatMessage {
    role: "system" | "assistant" | "user";
    content: string;
}

export type ChatParams = Omit<LLamaArguments, "prompt"> & {
    messages: ChatMessage[];
};

export interface CompletionCallback {
    (data: { token: string; completed: boolean }): void;
}

export class LLamaClient {
    private llamaNode: LLamaNode;

    constructor(config: LLamaConfig, enableLogger?: boolean) {
        if (enableLogger) {
            LLamaNode.enableLogger();
        }
        this.llamaNode = LLamaNode.create(config);
    }

    createChatCompletion = (
        params: ChatParams,
        callback: CompletionCallback
    ) => {
        console.warn(
            "The create chat completion function is just a simulation of dialog, it does not provide chatting interaction"
        );
        const data = new Date().toISOString();
        const { messages, ...rest } = params;
        const prompt = `You are AI assistant, please complete a dialog, where user interacts with AI assistant. AI assistant is helpful, kind, obedient, honest, and knows its own limits. AI assistant can do programming tasks and return codes.
Knowledge cutoff: 2021-09-01
Current date: ${data}
${messages.map(({ role, content }) => `${role}: ${content}`).join("\n")}
assistant: `;
        const completionParams = Object.assign({}, rest, { prompt });
        return this.createTextCompletion(completionParams, callback);
    };

    createTextCompletion = (
        params: LLamaArguments,
        callback: CompletionCallback
    ) => {
        let completed = false;
        return new Promise<boolean>((res, rej) => {
            this.llamaNode.inference(params, (response) => {
                switch (response.type) {
                    case "DATA": {
                        const data = {
                            token: response.data.token,
                            completed: !!response.data.completed,
                        };
                        if (data.completed) {
                            completed = true;
                        }
                        callback(data);
                        break;
                    }
                    case "END": {
                        res(completed);
                        break;
                    }
                    case "ERROR": {
                        rej(response.message);
                        break;
                    }
                }
            });
        });
    };
}