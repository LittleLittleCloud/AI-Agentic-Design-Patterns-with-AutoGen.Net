# AI-Agentic-Design-Patterns-with-AutoGen.Net

`AutoGen.Net` implementation of [`AI-Agentic-Design-Patterns-with-AutoGen`](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/)

> [!Note]
> Some examples are not exactly match what python AutoGen does. This is because some libraries are not available in .NET. For example, we use Tic Tac Toe game as a tool to demonstrate the concept of tool use where the python one use Chess game. This is because the `chess` library is not available in .NET. Another case is in Lesson 5, where python ones use financial analysis to demostrate code interpreter usage. In .Net AutoGen, however, I replace the financial analysis with math problem solving because of lacking library support in .Net eco-system.

## Console Apps
- [x] [Lesson 1: Multi-Agent Conversation and Stand-up Comedy](./L1_MultiAgent_Conversation_and_Standup_Comedy/)
- [x] [Lesson 2: Sequential Chats and Customer Onboarding](./L2_Sequential_Chats_and_Customer_Onboarding/)
- [x] [Lesson 3: Reflection_and_Blogpost_Writing](./L3_Reflection_and_Blogpost_Writing/)
- [x] [Lesson 4: Tool_Use_and_Conversational_Tic_Tac_Toe](./L4_Tool_Use_and_Conversational_Tic_Tac_Toe/)
- [x] [Lesson 5: Coding_and_Math_Problem_Solving](./L5_Coding_and_Math_Problem_Solving/)
- [x] [Lesson 6: L6-Planning_and_Stock_Report_Generation](./L6-Planning_and_Stock_Report_Generation/)

## Notebooks
- [x] [Lesson 1: Multi-Agent Conversation and Stand-up Comedy](./notebook/L1_MultiAgent_Conversation_and_Standup_Comedy.ipynb)
- [x] [Lesson 2: Sequential Chats and Customer Onboarding](./notebook/L2_Sequential_Chats_and_Customer_Onboarding.ipynb)
- [x] [Lesson 3: Reflection_and_Blogpost_Writing](./notebook/L3_Reflection_and_Blogpost_Writing.ipynb)
- [x] [Lesson 4: Tool_Use_and_Conversational_Tic_Tac_Toe](./notebook/L4_Tool_Use_and_Conversational_Tic_Tac_Toe.ipynb)
- [x] [Lesson 5: Coding_and_Math_Problem_Solving](./notebook/L5_Coding_and_Math_Problem_Solving.ipynb)
- [x] [Lesson 6: Planning_and_Stock_Report_Generation](./notebook/L6_Planning_and_Stock_Report_Generation.ipynb)

## Difference between Console Apps and Notebooks examples
- Notebook also contains the output result while Console app doesn't.
- The way to define a tool call is different between Console app and notebook examples. Console app uses `AutoGen.SourceGenerator` to create function definition directly from the structural comment while Notebook examples create function definition using semantic-kernel plugin style. This is because source generator is not available in Notebook use case.
- The code executor is different: In Console app examples, it needs to start a dotnet interactive instance to run C# code, which is not necessary in Notebook and we can simply execute the code using the running kernel.

