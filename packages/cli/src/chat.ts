import yargs from "yargs";

import { LLama, InferenceResultType } from "@llama-node/core";
import path from "path";
import { CLIInferenceArguments } from ".";
const readline = require("readline");

export class ChatCommand implements yargs.CommandModule<any, any> {
  command = "chat";
  describe = "Chat with llama";
  builder(args: yargs.Argv) {
    return (
      (args as yargs.Argv<CLIInferenceArguments>)
        .help("help")
        .example('llama inference -p "How are you?"', "Inference LLaMA")
        .options("feedPrompt", {
          type: "boolean",
          demandOption: false,
          description: "Set it to true to hide promt feeding progress",
        })
        .options("float16", { type: "boolean", demandOption: false })
        .options("ignoreEos", { type: "boolean", demandOption: false })
        .options("nBatch", { type: "number", demandOption: false })
        .options("nThreads", { type: "number", demandOption: false })
        .options("numPredict", { type: "number", demandOption: false })
        // .options("prompt", {
        //   type: "string",
        //   demandOption: true,
        //   alias: "p",
        // })
        .options("repeatLastN", { type: "number", demandOption: false })
        .options("repeatPenalty", { type: "number", demandOption: false })
        .options("seed", { type: "number", demandOption: false })
        .options("temp", { type: "number", demandOption: false })
        .options("tokenBias", { type: "string", demandOption: false })
        .options("topK", { type: "number", demandOption: false })
        .options("topP", { type: "number", demandOption: false })
        .options("path", {
          type: "string",
          demandOption: true,
          alias: ["m", "model"],
        })
        .options("numCtxTokens", { type: "number", demandOption: false })
        .options("logger", {
          type: "boolean",
          demandOption: false,
          default: true,
          alias: "verbose",
        })
    );
  }
  async handler(args: yargs.ArgumentsCamelCase) {
    const {
      $0,
      _,
      path: model,
      numCtxTokens,
      logger,
      ...rest
    } = args as yargs.ArgumentsCamelCase<CLIInferenceArguments>;
    const absolutePath = path.isAbsolute(model)
      ? model
      : path.join(process.cwd(), model);
    if (logger) {
      LLama.enableLogger();
    }
    const llama = LLama.create({ path: absolutePath, numCtxTokens });
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    rl.setPrompt("Talk with llama:");
    rl.prompt();
    rl.on("line", (line: string) => {
      const prompt = `### Instruction:

${line.trim()}
      
### Response:`;
      rest.prompt = prompt;
      llama.inference(rest, (result) => {
        switch (result.type) {
          case InferenceResultType.Data:
            process.stdout.write(result.data?.token ?? "");
            break;
          case InferenceResultType.Error:
            console.error(result.message);
            break;
          case InferenceResultType.End:
            rl.prompt();
            break;
        }
      });
    });
  }
}
