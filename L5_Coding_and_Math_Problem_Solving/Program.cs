using AutoGen.Core;
using AutoGen.DotnetInteractive;
using AutoGen.DotnetInteractive.Extension;
using AutoGen.OpenAI;
using AutoGen.OpenAI.Extension;
using Azure.AI.OpenAI;

var openAIKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? throw new Exception("Please set the OPENAI_API_KEY environment variable.");
var openAIModel = "gpt-4o-mini";

var openaiClient = new OpenAIClient(openAIKey);

// Define a code executor to run dotnet code
// Here we use the dotnet interactive as the code executor
// NOTE:
// GPT is much better at writing python code than dotnet code given the rich python training data.
// And the code execution feature in dotnet AutoGen is very limited if we asks agent to resolve tasks using C#.
// Therefore, this example is more of a demonstration of how to use dotnet interactive as code executor.
// It is not the apple-to-apple comparison with the corresponding python example.

// setup dotnet interactive
using var kernel = DotnetInteractiveKernelBuilder
    .CreateEmptyInProcessKernelBuilder()
    .AddCSharpKernel()
    .Build();

// Agent with code executor configuration
var codeExecutorAgent = new DefaultReplyAgent(
    name: "code_executor_agent",
    defaultReply: "no code to execute")
    .RegisterMiddleware(async (msgs, option, next, ct) =>
    {
        // check if the last message contain csharp code blcok
        var lastMsg = msgs.LastOrDefault();
        if (lastMsg?.ExtractCodeBlock("```csharp", "```") is string csharpCode)
        {
            var result = await kernel.RunSubmitCodeCommandAsync(csharpCode, "csharp");

            return new TextMessage(Role.Assistant, result ?? string.Empty, from: next.Name);
        }

        return await next.GenerateReplyAsync(msgs, option, ct);
    })
    .RegisterPrintMessage();

// Agent with dotnet coding writing capability
var coderAgent = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "code_writer_agent",
    modelName: openAIModel,
    systemMessage: """
    You act as dotnet coder, you write dotnet code to resolve task. Once you finish writing code, ask runner to run the code for you.

    Here're some rules to follow on writing dotnet code:
    - put code between ```csharp and ```
    - When creating http client, use `var httpClient = new HttpClient()`. Don't use `using var httpClient = new HttpClient()` because it will cause error when running the code.
    - Try to use `var` instead of explicit type.
    - Try avoid using external library, use .NET Core library instead.
    - Use top level statement to write code.
    - Always print out the result to console. Don't write code that doesn't print out anything.
    
    If you need to install nuget packages, put nuget packages in the following format:
    ```nuget
    nuget_package_name
    ```
    
    If your code is incorrect, Fix the error and send the code again.
    Once the task is resolved, say 'task completed' to finish the task.
    """,
    temperature: 0.4f)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

// The task!
var task = """
    calculate the 39th fibonacci number and save the result to result.txt
    """;

// Start the conversation
var chatHistory = new List<IMessage>()
{
    new TextMessage(Role.Assistant, task, from: codeExecutorAgent.Name),
};

await foreach(var reply in coderAgent.SendAsync(receiver: codeExecutorAgent, chatHistory: chatHistory, maxRound: 10))
{

    if (reply.GetContent()?.ToLower().Contains("task completed") == true)
    {
        break;
    }
    else
    {
        chatHistory.Add(reply);
    }
}

// Read the result from result.txt
var resultPath = Path.Combine("result.txt");
var result = File.ReadAllText(resultPath);
Console.WriteLine(result);

// User-Defined functions are not supported in C# AutoGen yet.
