package study.langchain4jdemo;

import dev.langchain4j.model.gpullama3.GPULlama3ChatModel;

import java.nio.file.Path;

public class Main {
    public static void main(String[] args) {
        String modelPath = args.length > 0
                ? args[0]
                : System.getenv("GPULLAMA3_MODEL_USED");
        if (modelPath == null) {
            throw new IllegalArgumentException(
                    "Pass the GGUF model path as arg[0] or set GPULLAMA3_MODEL_USED");
        }

        System.out.println("Building GPULlama3ChatModel from " + modelPath);
        GPULlama3ChatModel model = GPULlama3ChatModel.builder()
                .modelPath(Path.of(modelPath))
                .temperature(0.0)
                .topP(0.9)
                .maxTokens(64)
                .onGPU(Boolean.TRUE)
                .build();

        System.out.println("Prompt: In exactly one sentence, explain what a GPU kernel is.");
        String answer = model.chat("In exactly one sentence, explain what a GPU kernel is.");
        System.out.println("Response: " + answer);
    }
}
