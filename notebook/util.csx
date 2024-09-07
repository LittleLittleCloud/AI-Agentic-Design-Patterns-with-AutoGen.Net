using Azure;
using Azure.AI.OpenAI;

public static class OpenAIClientProvider
{
    public static OpenAIClient Create()
    {
        // if the OPENAI_API_KEY is set
        // we will use an OpenAI client pointing directly to OpenAI
        var openAIKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        if (!string.IsNullOrWhiteSpace(openAIKey))
        {
            var openaiClient = new OpenAIClient(openAIKey);
            return openaiClient;
        }
        
        // if no OPENAI_API_KEY was set,
        // and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY were set
        // we will use an OpenAI client pointing at an Azure OpenAI Service
        var azureOaiEndpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT");
        var azureOaiKey = Environment.GetEnvironmentVariable("AZURE_OPENAI_KEY");

        if (!string.IsNullOrWhiteSpace(azureOaiEndpoint) && !string.IsNullOrWhiteSpace(azureOaiKey))
        {
            var openaiClient = new OpenAIClient(new Uri(azureOaiEndpoint), new AzureKeyCredential(azureOaiKey));
            return openaiClient;
        }

        throw new Exception(
            "Set environment variable 'OPENAI_API_KEY' to use models directly from OpenAI, or 'AZURE_OPENAI_ENDPOINT' and 'AZURE_OPENAI_KEY' to use models from Azure OpenAI.");
    }
}