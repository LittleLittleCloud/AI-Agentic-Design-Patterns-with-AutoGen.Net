using AutoGen.Core;
using AutoGen.OpenAI;
using AutoGen.OpenAI.Extension;
using Azure.AI.OpenAI;

var openAIKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? throw new Exception("Please set the OPENAI_API_KEY environment variable.");
var openAIModel = "gpt-3.5-turbo";

// Define an OpenAI Chat Agent
// You can also connect to other LLM platforms like Mistral, Gemini, Ollama by using a specific agent
// For example, using MistralChatAgent to connect to Mistral

var openaiClient = new OpenAIClient(openAIKey);
var agent = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "chatbot",
    modelName: openAIModel)
    .RegisterMessageConnector() // convert OpenAI Message to AutoGen Message
    .RegisterPrintMessage(); // print the message to the console

// Start the conversation and print the response
var _ = await agent.SendAsync("Tell me a joke");

// We use a token count middleware to collect all the oai messages which contain the token count information
// In the rest of examples, we use non-streaming agent because the token information is not available in streaming chunks.
var tokenCountMiddleware = new CountTokenMiddleware();
var openaiMessageConnector = new OpenAIChatRequestMessageConnector();

// Conversation
// Setting up a conversation between Cathy and Joe
var cathy = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "cathy",
    modelName: openAIModel,
    systemMessage: "Your name is Cathy and you are a stand-up comedian.")
    .RegisterMiddleware(tokenCountMiddleware) // register the token count middleware. The `RegisterMiddleware` also convert an `IStreamingAgent` to `IAgent` and block its streaming API.
    .RegisterMiddleware(openaiMessageConnector)
    .RegisterPrintMessage();

var joe = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "joe",
    modelName: openAIModel,
    systemMessage: """
    Your name is Joe and you are a stand-up comedian.
    Start the next joke from the punchline of the previous joke.
    """)
    .RegisterMiddleware(tokenCountMiddleware)
    .RegisterMiddleware(openaiMessageConnector)
    .RegisterPrintMessage();

var chatResult = await joe.SendAsync(
    receiver: cathy,
    message: "I'm Joe. Let's keep the jokes rolling.",
    maxRound: 4);

// Print token consumption
Console.WriteLine($"Total Token count: {tokenCountMiddleware.GetTokenCount()}");

// Get a better summary of conversation from a summary agent.
var summaryAgent = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "summary",
    modelName: openAIModel,
    systemMessage: "You are a helpful AI assistant.")
    .RegisterMessageConnector()
    .RegisterPrintMessage();

var summary = await summaryAgent.SendAsync("summarize the converation", chatResult);

// Chat Termination
// Terminate the conversation by running the chat step by step and check if the conversation meeting the terminate condition
cathy = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "cathy",
    modelName: openAIModel,
    systemMessage: """
    Your name is Cathy and you are a stand-up comedian.
    When you're ready to end the conversation, say 'I gotta go'.
    """)
    .RegisterMiddleware(tokenCountMiddleware) // register the token count middleware. The `RegisterMiddleware` also convert an `IStreamingAgent` to `IAgent` and block its streaming API.
    .RegisterMiddleware(openaiMessageConnector)
    .RegisterPrintMessage();

joe = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "joe",
    modelName: openAIModel,
    systemMessage: """
    Your name is Joe and you are a stand-up comedian.
    When you're ready to end the conversation, say 'I gotta go'.
    End the conversation when you see two jokes.
    """)
    .RegisterMiddleware(tokenCountMiddleware)
    .RegisterMiddleware(openaiMessageConnector)
    .RegisterPrintMessage();

var chatHistory = new List<IMessage>
{
    new TextMessage(Role.User, "I'm Joe. Let's keep the jokes rolling.", from: joe.Name)
};
var roundLeft = 10;
while(roundLeft > 0)
{
    var replies = await joe.SendAsync(
        receiver: cathy,
        chatHistory,
        maxRound: 1);

    var reply = replies.Last();
    if (reply.GetContent()?.ToLower().Contains("i gotta go") is true)
    {
        break;
    }
    chatHistory.Add(reply);
    roundLeft--;
}


class CountTokenMiddleware : IMiddleware
{
    private readonly List<ChatCompletions> messages = new();
    public string? Name => nameof(CountTokenMiddleware);

    public int GetTokenCount()
    {
        return messages.Sum(m => m.Usage.CompletionTokens);
    }

    public async Task<IMessage> InvokeAsync(MiddlewareContext context, IAgent agent, CancellationToken cancellationToken = default)
    {
        var reply = await agent.GenerateReplyAsync(context.Messages, context.Options, cancellationToken: cancellationToken);

        if (reply is IMessage<ChatCompletions> message)
        {
            messages.Add(message.Content);
        }

        return reply;
    }
}
