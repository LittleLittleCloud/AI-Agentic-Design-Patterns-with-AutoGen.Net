using Azure;
using Azure.AI.OpenAI;
using OpenAI;
using OpenAI.Chat;

public static class ChatClientProvider
{
    public static ChatClient Create(string openAIModel)
    {
        // if the OPENAI_API_KEY is set
        // we will use an OpenAI client pointing directly to OpenAI
        var openAIKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        if (!string.IsNullOrWhiteSpace(openAIKey))
        {
            var openaiClient = new OpenAIClient(openAIKey);
            var chatClient = openaiClient.GetChatClient(openAIModel);
            return chatClient;
        }
        
        // if no OPENAI_API_KEY was set,
        // and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY were set
        // we will use an OpenAI client pointing at an Azure OpenAI Service
        var azureOaiEndpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT");
        var azureOaiKey = Environment.GetEnvironmentVariable("AZURE_OPENAI_KEY");

        if (!string.IsNullOrWhiteSpace(azureOaiEndpoint) && !string.IsNullOrWhiteSpace(azureOaiKey))
        {
            var azureClient = new AzureOpenAIClient(new Uri(azureOaiEndpoint), new AzureKeyCredential(azureOaiKey));
            var chatClient = azureClient.GetChatClient(openAIModel);
            return chatClient;
        }

        throw new Exception(
            "Set environment variable 'OPENAI_API_KEY' to use models directly from OpenAI, or 'AZURE_OPENAI_ENDPOINT' and 'AZURE_OPENAI_KEY' to use models from Azure OpenAI.");
    }
}