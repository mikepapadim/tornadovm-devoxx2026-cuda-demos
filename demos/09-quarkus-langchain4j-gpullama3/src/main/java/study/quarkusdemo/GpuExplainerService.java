package study.quarkusdemo;

import dev.langchain4j.service.UserMessage;
import io.quarkiverse.langchain4j.RegisterAiService;
import jakarta.enterprise.context.ApplicationScoped;

@RegisterAiService
@ApplicationScoped
public interface GpuExplainerService {

    @UserMessage("In exactly one sentence, explain what a GPU kernel is.")
    String explainGpuKernel();
}
