using AutoGen.Core;
using AutoGen.OpenAI;
using AutoGen.OpenAI.Extension;
using Azure.AI.OpenAI;
using OpenAI;
using OpenAI.Chat;
using System.Runtime.CompilerServices;
using static Google.Cloud.AIPlatform.V1.PublisherModel.Types.CallToAction.Types;

var openAIKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? throw new Exception("Please set the OPENAI_API_KEY environment variable.");
var openAIModel = "gpt-4o-mini";
var openaiClient = new OpenAIClient(openAIKey);
var chatClient = openaiClient.GetChatClient(openAIModel);

// Define an OpenAI Chat Agent
// You can also connect to other LLM platforms like Mistral, Gemini, Ollama by using a specific agent
// For example, using MistralChatAgent to connect to Mistral
var agent = new OpenAIChatAgent(
    chatClient: chatClient,
    name: "chatbot")
    .RegisterMessageConnector() // convert OpenAI Message to AutoGen Message
    .RegisterPrintMessage(); // print the message to the console

// Start the conversation and print the response
var _ = await agent.SendAsync("Tell me a joke");

// We use a token count middleware to collect all the oai messages which contain the token count information
// In the rest of examples, we use non-streaming agent because the token information is not available in streaming chunks.
var tokenCountMiddleware = new CountTokenMiddleware();

// Conversation
// Setting up a conversation between Cathy and Joe
IAgent cathy = new OpenAIChatAgent(
    chatClient: chatClient,
    name: "cathy",
    systemMessage: "Your name is Cathy and you are a stand-up comedian.")
    .RegisterStreamingMiddleware(tokenCountMiddleware)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

IAgent joe = new OpenAIChatAgent(
    chatClient: chatClient,
    name: "joe",
    systemMessage: """
    Your name is Joe and you are a stand-up comedian.
    Start the next joke from the punchline of the previous joke.
    """)
    .RegisterStreamingMiddleware(tokenCountMiddleware)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

var chatResult = await joe.SendAsync(
    receiver: cathy,
    message: "I'm Joe. Let's keep the jokes rolling.",
    maxRound: 4)
    .ToListAsync();

// Print token consumption
Console.WriteLine($"Total Token count: {tokenCountMiddleware.GetTokenCount()}");

// Get a better summary of conversation from a summary agent.
var summaryAgent = new OpenAIChatAgent(
    chatClient: chatClient,
    name: "summary",
    systemMessage: "You are a helpful AI assistant.")
    .RegisterMessageConnector()
    .RegisterPrintMessage();

var summary = await summaryAgent.SendAsync("summarize the converation", chatResult);

// Chat Termination
// Terminate the conversation by running the chat step by step and check if the conversation meeting the terminate condition
cathy = new OpenAIChatAgent(
    chatClient: chatClient,
    name: "cathy",
    systemMessage: """
    Your name is Cathy and you are a stand-up comedian.
    When you're ready to end the conversation, say 'I gotta go'.
    """)
    .RegisterStreamingMiddleware(tokenCountMiddleware)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

joe = new OpenAIChatAgent(
    chatClient: chatClient,
    name: "joe",
    systemMessage: """
    Your name is Joe and you are a stand-up comedian.
    When you're ready to end the conversation, say 'I gotta go'.
    End the conversation when you see two jokes.
    """)
    .RegisterStreamingMiddleware(tokenCountMiddleware)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

var chatHistory = new List<IMessage>
{
    new TextMessage(Role.User, "I'm Joe. Let's keep the jokes rolling.", from: joe.Name)
};

await foreach(var msg in joe.SendAsync(receiver: cathy, chatHistory, maxRound: 10))
{
    if (msg.GetContent()?.ToLower().Contains("i gotta go") is true)
    {
        break;
    }
}

class CountTokenMiddleware : IStreamingMiddleware
{
    private readonly List<ChatCompletion> messages = new();
    private readonly List<StreamingChatCompletionUpdate> streamingMessages = new();
    public string? Name => nameof(CountTokenMiddleware);

    public int GetTokenCount()
    {
        return messages.Sum(m => m.Usage.TotalTokens) + streamingMessages.Sum(m => m.Usage?.TotalTokens ?? 0);
    }

    public async Task<IMessage> InvokeAsync(MiddlewareContext context, IAgent agent, CancellationToken cancellationToken = default)
    {
        var reply = await agent.GenerateReplyAsync(context.Messages, context.Options, cancellationToken: cancellationToken);

        if (reply is IMessage<ChatCompletion> message)
        {
            messages.Add(message.Content);
        }

        return reply;
    }

    public async IAsyncEnumerable<IMessage> InvokeAsync(
        MiddlewareContext context,
        IStreamingAgent agent,
        [EnumeratorCancellation]
        CancellationToken cancellationToken = default)
    {
        await foreach (var reply in agent.GenerateStreamingReplyAsync(context.Messages, context.Options, cancellationToken: cancellationToken))
        {
            if (reply is IMessage<StreamingChatCompletionUpdate> message)
            {
                streamingMessages.Add(message.Content);
            }

            yield return reply;
        }
    }
}
