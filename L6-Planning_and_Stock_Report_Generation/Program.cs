using AutoGen.Core;
using AutoGen.OpenAI;
using AutoGen.OpenAI.Extension;
using System.Text;
using Util;

// Note
// This example is slightly different with the python ones when it comes to engineer agent
// due to the lacking built-in support of running python code in C# AutoGen.
// The engineer and executor agent is replaced with market watcher and data analyst agents to gather and plot the stock price data.

var openAIModel = "gpt-4o-mini";
var openaiClient = OpenAIClientProvider.Create();

// The Task!
var task = """
Write a blogpost about the stock price performance of Nvidia in the past month.
Today's date is 2024-04-23.
""";

// Build a group chat
// This group chat will include these agents:

// - User_proxy: to allow the user to comment on the report and ask the writer to refine it.
// - Planner: to determine relevant information needed to complete the task.
// - Market_Watcher: to gather the stock price data.
// - Data_Analyst: to plot the stock price data.
// - Writer: to write the report.

var user = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "user",
    modelName: openAIModel,
    systemMessage: """
    Give the task, and send instructions to writer to refine the blog post.
    """)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

var planner = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "Planner",
    modelName: openAIModel,
    systemMessage: """
    You are Planner.
    Given a task, please determine what step is needed to complete the task.
    Please only suggest steps that can be done by others.
    After each step is done by others, check the progress and instruct the remaining steps.
    If a step fails, try to work around it.
    If the task is completed, say 'task completed'.
    """)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

var workingDirectory = Path.Combine(Directory.GetCurrentDirectory(), "work");
if (!Directory.Exists(workingDirectory))
{
    Directory.CreateDirectory(workingDirectory);
}
var tools = new MarketTools(workingDirectory);
var marketWatcherFunctionCallMiddleware = new FunctionCallMiddleware(
    functions: [tools.RetrieveStockPriceFunctionContract],
    functionMap: new Dictionary<string, Func<string, Task<string>>>()
    {
        { tools.RetrieveStockPriceFunctionContract.Name!, tools.RetrieveStockPriceWrapper },
    });

var marketWatcher = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "Market_Watcher",
    modelName: openAIModel,
    systemMessage: """
    You are Market Watcher. You can gather stock price data.
    """)
    .RegisterMessageConnector()
    .RegisterStreamingMiddleware(marketWatcherFunctionCallMiddleware)
    .RegisterPrintMessage();

var dataAnalystFunctionCallMiddleware = new FunctionCallMiddleware(
    functions: [tools.PlotStockLineChartFunctionContract],
    functionMap: new Dictionary<string, Func<string, Task<string>>>()
    {
        { tools.PlotStockLineChartFunctionContract.Name!, tools.PlotStockLineChartWrapper },
    });

var dataAnalyst = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "Data_Analyst",
    modelName: openAIModel,
    systemMessage: """
    You are Data Analyst. You can plot stock price data.
    """)
    .RegisterMessageConnector()
    .RegisterStreamingMiddleware(dataAnalystFunctionCallMiddleware)
    .RegisterPrintMessage();

var writer = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "Writer",
    modelName: openAIModel,
    systemMessage: """
    Please write blogs in markdown format (with relevant titles) and put the content in pseudo ```md``` code block.
    You take feedback from the admin and refine your blog.
    """)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

// Define the group chat

// Create a group chat admin to orchestrate the group chat
// The admin will be responsible for selecting the next speaker
// It will not directly participate in the conversation
var groupChatAdminAgent = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "Admin",
    modelName: openAIModel)
    .RegisterMessageConnector();

// Add a speaker selection policy using workflow
var workflow = new Graph();

// user <=> planner
workflow.AddTransition(Transition.Create(user, planner));
workflow.AddTransition(Transition.Create(planner, user));

// planner <=> marketWatcher
workflow.AddTransition(Transition.Create(planner, marketWatcher));
workflow.AddTransition(Transition.Create(marketWatcher, planner));

// planner <=> dataAnalyst
workflow.AddTransition(Transition.Create(planner, dataAnalyst));
workflow.AddTransition(Transition.Create(dataAnalyst, planner));

// planner <=> writer
workflow.AddTransition(Transition.Create(planner, writer));
workflow.AddTransition(Transition.Create(writer, planner));

// Create a group chat
var groupChat = new GroupChat(
    admin: groupChatAdminAgent,
    workflow: workflow,
    members: [user, planner, marketWatcher, dataAnalyst, writer]);

// Add agent self-description as inital messages.
groupChat.AddInitializeMessage(new TextMessage(Role.Assistant, "I am market watcher. I can gather stock price data.", from: marketWatcher.Name));
groupChat.AddInitializeMessage(new TextMessage(Role.Assistant, "I am data analyst. I can plot stock price data.", from: dataAnalyst.Name));
groupChat.AddInitializeMessage(new TextMessage(Role.Assistant, "I am writer. I can write blogs in markdown format.", from: writer.Name));

// start the group chat!
var chatHistory = new List<IMessage>()
{
    new TextMessage(Role.User, task, from: user.Name),
};

await foreach (var reply in groupChat.SendAsync(chatHistory, maxRound: 20))
{
    chatHistory.Add(reply);

    if (reply.GetContent()?.ToLower().Contains("task completed") is true)
    {
        break;
    }
}

// retrieve the final blog post from writer and save it to a file
var finalBlogPath = Path.Combine(workingDirectory, "nvidia-blog.md");
var finalBlog = chatHistory.Where(c => c.From == writer.Name).Last().GetContent();
// retrieve the content between ```md``` code block
var blogContent = finalBlog!.Split("```md")[1].Split("```")[0].Trim();
await File.WriteAllTextAsync(finalBlogPath, blogContent);

// Print the final blog post path
Console.WriteLine($"The blog post has been written and saved to {finalBlogPath}");

public partial class MarketTools
{
    private string workDir;

    public MarketTools(string workDir)
    {
        this.workDir = workDir;
    }

    /// <summary>
    /// Retrieve stock price data for a given symbol and date range.
    /// </summary>
    /// <param name="symbol">stock symbol.</param>
    /// <param name="from">from date, in the format of YYYY-MM-dd</param>
    /// <param name="to">to date, in the format of YYYY-MM-dd</param>
    /// <returns></returns>
    [Function]
    public async Task<string> RetrieveStockPrice(string symbol, string from, string to)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Retrieving stock price data for {symbol} from {from} to {to}...");
        // Retrieve stock price data from the database
        var stockData = """
            2024-03-25    950.020020
            2024-03-26    925.609985
            2024-03-27    902.500000
            2024-03-28    903.559998
            2024-04-01    903.630005
            2024-04-02    894.520020
            2024-04-03    889.640015
            2024-04-04    859.049988
            2024-04-05    880.080017
            2024-04-08    871.330017
            2024-04-09    853.539978
            2024-04-10    870.390015
            2024-04-11    906.159973
            2024-04-12    881.859985
            2024-04-15    860.010010
            2024-04-16    874.150024
            2024-04-17    840.349976
            2024-04-18    846.710022
            2024-04-19    762.000000
            2024-04-22    795.179993
            """;

        sb.AppendLine(stockData);

        return sb.ToString();
    }

    /// <summary>
    /// Plot candle line chart for stock price data.
    /// </summary>
    /// <param name="symbol">stock symbol.</param>
    /// <param name="close">close price.</param>
    /// <param name="date">date, in the format of YYYY-MM-dd</param>
    [Function]
    public async Task<string> PlotStockLineChart(string symbol, float[] close, string[] date)
    {
        var title = $"Stock Price Performance of {symbol} from {date[0]} to {date[^1]}";

        // copy img/nvidia-plot.png to workDir/{title}.png
        var targetPath = Path.Combine(workDir, $"{title}.png");
        var imgPath = Path.Combine("img", "nvidia-plot.png");
        File.Copy(imgPath, targetPath, true);

        var reply = $"""
            Stock price data has been plotted.
            You can view the chart at {targetPath}
            """;

        return reply;
    }
}

