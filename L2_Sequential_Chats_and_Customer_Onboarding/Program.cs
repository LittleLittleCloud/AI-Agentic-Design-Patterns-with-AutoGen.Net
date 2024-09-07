using AutoGen.Core;
using AutoGen.OpenAI;
using AutoGen.OpenAI.Extension;
using Util;

var openAIModel = "gpt-4o-mini";
var openaiClient = OpenAIClientProvider.Create();

// Create the needed agents

var onboardingPersonalInformationAgent = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "Onboarding_Personal_Information_Agent",
    modelName: openAIModel,
    systemMessage: """
    You are a helpful customer onboarding agent,
    you are here to help new customers get started with our product.
    Your job is to gather customer's name and location.
    Do not ask for other information. Return 'TERMINATE' 
    when you have gathered all the information.
    """)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

var onboardingTopicPreferenceAgent = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "Onboarding_Topic_Preference_Agent",
    modelName: openAIModel,
    systemMessage: """
    You are a helpful customer onboarding agent,
    you are here to help new customers get started with our product.
    Your job is to gather customer's preferences on news topics.
    Do not ask for other information.
    Return 'TERMINATE' when you have gathered all the information.
    """)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

var customerEngagementAgent = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "Customer_Engagement_Agent",
    modelName: openAIModel,
    systemMessage: """
    You are a helpful customer service agent
    here to provide fun for the customer based on the user's
    personal information and topic preferences.
    This could include fun facts, jokes, or interesting stories.
    Make sure to make it engaging and fun!
    Return 'TERMINATE' when you are done.
    """)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

var summarizer = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "Summarizer",
    modelName: openAIModel,
    systemMessage: """
    You are a helpful summarizer agent.
    Your job is to summarize the conversation between the user and the customer service agent.
    Return 'TERMINATE' when you are done.
    """)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

var user = new OpenAIChatAgent(
    openAIClient: openaiClient,
    name: "User",
    modelName: openAIModel,
    systemMessage: """
    Your name is John and you live in New York.
    You are reaching out to customer service to find out something fun.
    """)
    .RegisterMessageConnector()
    .RegisterPrintMessage();

// Creating Tasks
// In python AutoGen, you can use initiate_chats to create and run a sequential of tasks in json object
// In dotnet AutoGen, however, that feature is not available, so you need to manually create these tasks using code.

// Task 1. Onboard customer by gathering name and location
// (onboard_personal_information_agent -> user .. (repeat less than two times)) -> summarizer
var greetingMessage = new TextMessage(Role.Assistant, """
    Hello, I'm here to help you get started with our product.
    Could you tell me your name and location?
    """, from: onboardingPersonalInformationAgent.Name);

var conversation = await onboardingPersonalInformationAgent.SendAsync(
    receiver: user,
    [greetingMessage],
    maxRound: 2)
    .ToListAsync();

var summarizePrompt = """
    Return the customer information into as JSON object only: {'name': '', 'location': ''}
    """;

var summary = await summarizer.SendAsync(summarizePrompt, conversation);

// Task 2. Gapther customer's preferences on news topics
// (onboarding_topic_preference_agent -> user .. (repeat one time)) -> summarizer
var topicPreferenceMessage = new TextMessage(Role.Assistant, """
    Great! Could you tell me what topics you are interested in reading about?
    """, from: onboardingTopicPreferenceAgent.Name);

conversation = await onboardingTopicPreferenceAgent.SendAsync(
    receiver: user,
    [topicPreferenceMessage],
    maxRound: 1)
    .ToListAsync();

// Keep summarizing
summary = await summarizer.SendAsync(chatHistory: new[] { summary }.Concat(conversation));

// Task 3. Engage the customer with fun facts, jokes, or interesting stories based on the user's personal information and topic preferences
// (user(find fun thing to read) -> customerEngagementAgent .. (repeat 1 time)) -> summarizer
var funFactMessage = new TextMessage(Role.User, """
    Let's find something fun to read.
    """, from: user.Name);

conversation = await user.SendAsync(
    receiver: customerEngagementAgent,
    chatHistory: conversation.Concat([funFactMessage]), // this time, we keep the previous conversation history
    maxRound: 1)
    .ToListAsync();

// Keep summarizing
summary = await summarizer.SendAsync(chatHistory: new[] { summary }.Concat(conversation));

