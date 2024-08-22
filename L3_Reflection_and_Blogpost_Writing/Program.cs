using AutoGen.Core;
using AutoGen.OpenAI;
using AutoGen.OpenAI.Extension;
using Azure.AI.OpenAI;

var openAIKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? throw new Exception("Please set the OPENAI_API_KEY environment variable.");
var openAIModel = "gpt-4o-mini";

var openaiClient = new OpenAIClient(openAIKey);

// The Task!
var task = """
    Write a concise but engaging blogpost about
    DeepLearning.AI. Make sure the blogpost is
    within 100 words.
    """;

// Create a writer agent
var writer = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "Writer",
    modelName: openAIModel,
    systemMessage: """
    You are a writer. You write engaging and concise blogpost (with title) on given topics.
    You must polish your writing based on the feedback you receive and give a refined version.
    Only return your final work without additional comments."
    """)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

// Write the blogpost
var reply = await writer.SendAsync(task);

// Adding reflection by creating a critic agent to reflect on the work of the writer agent.
var critic = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "Critic",
    modelName: openAIModel,
    systemMessage: """
    You are a critic. You review the work of the writer and provide constructive feedback to help improve the quality of the content."
    """)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

var conversation = await critic.SendAsync(
    receiver: writer,
    message: task,
    maxRound: 3)
    .ToListAsync();

// Use the last message in the conversation as the final result
var res = conversation.Last();

// Nested chat using middleware pattern
// Python AutoGen allows you to create a Json object task list and pass it to agent using `register_nested_chats`
// This is not supported in C# AutoGen yet.
// But you can achieve the same using the middleware pattern.

var SEOReviewer = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "SEO_Reviewer",
    modelName: openAIModel,
    systemMessage: """
    You are an SEO reviewer, known for your ability to optimize content for search engines, ensuring that it ranks well and attracts organic traffic.
    Make sure your suggestion is concise (within 3 bullet points), concrete and to the point.
    Begin the review by stating your role.
    """)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

var LegalReviewer = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "Legal_Reviewer",
    modelName: openAIModel,
    systemMessage: """
    You are a legal reviewer, known for your ability to ensure that content is legally compliant and free from any potential legal issues.
    Make sure your suggestion is concise (within 3 bullet points), concrete and to the point.
    Begin the review by stating your role.
    """)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

var EthicsReviewer = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "Ethics_Reviewer",
    modelName: openAIModel,
    systemMessage: """
    You are an ethics reviewer, known for your ability to ensure that content is ethically sound and free from any potential ethical issues.
    Make sure your suggestion is concise (within 3 bullet points), concrete and to the point.
    Begin the review by stating your role.
    """)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

var MetaReviewer = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "Meta_Reviewer",
    modelName: openAIModel,
    systemMessage: """
    You are a meta reviewer, you aggragate and review the work of other reviewers and give a final suggestion on the content."
    """)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

// Orchestrate the nested chats to solve the task
var middleware = new NestChatMiddleware(
    writer: writer,
    seo: SEOReviewer,
    legal: LegalReviewer,
    ethics: EthicsReviewer,
    meta: MetaReviewer);

var nestChatCritic = critic
    .RegisterMiddleware(middleware)
    .RegisterPrintMessage();

conversation = await nestChatCritic.SendAsync(
    message: task,
    receiver: writer,
    maxRound: 3)
    .ToListAsync();

var finalResult = conversation.Last();

class NestChatMiddleware : IMiddleware
{
    private readonly IAgent seo;
    private readonly IAgent legal;
    private readonly IAgent ethics;
    private readonly IAgent meta;
    private readonly IAgent writer;

    public NestChatMiddleware(IAgent writer, IAgent seo, IAgent legal, IAgent ethics, IAgent meta)
    {
        this.writer = writer;
        this.seo = seo;
        this.legal = legal;
        this.ethics = ethics;
        this.meta = meta;
    }

    public string? Name => nameof(NestChatMiddleware);

    public async Task<IMessage> InvokeAsync(MiddlewareContext context, IAgent critic, CancellationToken cancellationToken = default)
    {
        // trigger only when the last message is from writer
        if (context.Messages.Last().From != writer.Name)
        {
            return await critic.GenerateReplyAsync(context.Messages, context.Options, cancellationToken);
        }

        var messageToReview = context.Messages.Last();
        var reviewPrompt = $"""
            Review the following content.

            {messageToReview.GetContent()}
            """;
        // SEO Review
        var seoReview = await critic.SendAsync(
            receiver: seo,
            message: reviewPrompt,
            maxRound: 1)
            .ToListAsync();
        var legalReview = await critic.SendAsync(
            receiver: legal,
            message: reviewPrompt,
            maxRound: 1)
            .ToListAsync();

        var ethicsReview = await critic.SendAsync(
            receiver: ethics,
            message: reviewPrompt,
            maxRound: 1)
            .ToListAsync();

        var reviews = seoReview.Concat(legalReview).Concat(ethicsReview);

        var metaReview = await critic.SendAsync(
            receiver: meta,
            message: "Aggregrate feedback from all reviewers and give final suggestions on the writing.",
            chatHistory: reviews,
            maxRound: 1)
            .ToListAsync();

        var lastReview = metaReview.Last();
        lastReview.From = critic.Name;

        return lastReview;
    }
}