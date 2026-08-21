package study.quarkusdemo;

import io.quarkus.runtime.Quarkus;
import io.quarkus.runtime.QuarkusApplication;
import io.quarkus.runtime.annotations.QuarkusMain;
import jakarta.inject.Inject;

@QuarkusMain(name = "GpuLlama3QuarkusChatDemo")
public class Main implements QuarkusApplication {

    @Inject
    GpuExplainerService ai;

    @Override
    public int run(String... args) throws Exception {
        System.out.println("Prompt: In exactly one sentence, explain what a GPU kernel is.");
        String answer = ai.explainGpuKernel();
        System.out.println("Response: " + answer);
        return 0;
    }

    public static void main(String[] args) {
        Quarkus.run(Main.class, args);
    }
}
