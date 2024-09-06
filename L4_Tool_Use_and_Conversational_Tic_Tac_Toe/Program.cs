using AutoGen.Core;
using AutoGen.OpenAI;
using AutoGen.OpenAI.Extension;
using Microsoft.SemanticKernel;
using OpenAI;
using System.ComponentModel;
using System.Text;

var openAIKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? throw new Exception("Please set the OPENAI_API_KEY environment variable.");
var openAIModel = "gpt-4o-mini";

var openaiClient = new OpenAIClient(openAIKey);
var chatClient = openaiClient.GetChatClient(openAIModel);

// Initialize the chess board [3, 3]
// 0: empty, 1: X, 2: O
// Define the needed tools
// 1. Tool for getting legal moves
// 2. Tool for making a move on the board

var board = new TicTacToe(new int[3, 3]);
var toolMiddleware = new FunctionCallMiddleware(
    functions:
    [
        board.DisplayBoardFunctionContract,
        board.MakeMoveFunctionContract,
        board.GetLegalMovesFunctionContract,
    ],
    functionMap: new Dictionary<string, Func<string, Task<string>>>
    {
        { board.DisplayBoardFunctionContract.Name!, board.DisplayBoardWrapper },
        { board.MakeMoveFunctionContract.Name!, board.MakeMoveWrapper },
        { board.GetLegalMovesFunctionContract.Name!, board.GetLegalMovesWrapper },
    });

// Create agents
// You will create the player agents for the tic-tac-toe game.
var nestMiddleware = new NestMiddleware();
var playerX = new OpenAIChatAgent(
    chatClient: chatClient,
    name: "Player_X",
    systemMessage: """
    You are Player X. You are playing Tic-Tac-Toe against Player O.
    You can make a move by providing the row and column number.
    The board is 3x3, and the row and column number should be between 0 and 2.
    First check the status, then get legal moves, and finally make a move.
    """)
    .RegisterMessageConnector()
    .RegisterMiddleware(toolMiddleware)
    .RegisterPrintMessage()
    .RegisterMiddleware(nestMiddleware);

var playerO = new OpenAIChatAgent(
    chatClient: chatClient,
    name: "Player_O",
    systemMessage: """
    You are Player O. You are playing Tic-Tac-Toe against Player X.
    You can make a move by providing the row and column number.
    The board is 3x3, and the row and column number should be between 0 and 2.
    First check the status, then get legal moves, and finally make a move.
    """)
    .RegisterMessageConnector()
    .RegisterMiddleware(toolMiddleware)
    .RegisterPrintMessage()
    .RegisterMiddleware(nestMiddleware);

// Start the game
var conversationHistory = new List<IMessage>()
{
    new TextMessage(Role.Assistant, "You start first", from: playerX.Name),
};

await foreach (var msg in playerX.SendAsync(receiver: playerO, chatHistory: conversationHistory, maxRound: 9))
{
    conversationHistory.Add(msg);

    // break if anyone wins
    if (board.CheckWin(1))
    {
        Console.WriteLine("Player X wins!");

        break;
    }
    else if (board.CheckWin(2))
    {
        Console.WriteLine("Player O wins!");

        break;
    }

    // print the board
    var displayBoard = await board.DisplayBoard();
    Console.WriteLine(displayBoard);
}

// check if it's a draw
if (!board.CheckWin(1) && !board.CheckWin(2))
{
    Console.WriteLine("It's a draw!");
}

public class NestMiddleware : IMiddleware
{
    public NestMiddleware()
    {
    }

    public string? Name => nameof(NestMiddleware);

    public async Task<IMessage> InvokeAsync(MiddlewareContext context, IAgent agent, CancellationToken cancellationToken = default)
    {
        // check status
        var checkStatusMessage = new TextMessage(Role.User, "check status");
        var status = await agent.SendAsync(chatHistory: [checkStatusMessage]);

        // get legal moves
        var legalMoves = await agent.SendAsync("get legal moves", chatHistory: [checkStatusMessage, status]);

        // make move
        var move = await agent.SendAsync("make move", chatHistory: [status, legalMoves]);

        return move;
    }
}


public partial class TicTacToe
{
    public int[,] board = new int[3, 3];

    public TicTacToe(int[,] board)
    {
        if (board.GetLength(0) != 3 || board.GetLength(1) != 3)
        {
            throw new ArgumentException("The board should be 3x3.");
        }

        this.board = board;
    }

    /// <summary>
    /// Get all legal moves on the board.
    /// </summary>
    [Function]
    [KernelFunction]
    [Description("Get all legal moves on the board.")]
    public async Task<string> GetLegalMoves()
    {
        var legalMoves = new List<int[]>();
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                if (board[i, j] == 0)
                {
                    legalMoves.Add([i, j]);
                }
            }
        }

        var sb = new StringBuilder();
        sb.AppendLine("Legal moves:");
        foreach (var move in legalMoves)
        {
            sb.AppendLine($"({move[0]}, {move[1]})");
        }

        return sb.ToString();
    }

    /// <summary>
    /// Display the current board.
    /// </summary>
    [Function]
    [KernelFunction]
    [Description("Display the current board.")]
    public Task<string> DisplayBoard()
    {
        var sb = new StringBuilder();
        sb.AppendLine("Current board:");
        var charMap = new Dictionary<int, string>
        {
            { 0, "0" },
            { 1, "X" },
            { 2, "O" },
        };
        for (var i = 0; i < 3; i++)
        {
            sb.AppendLine(string.Join(" | ", charMap[board[i, 0]], charMap[board[i, 1]], charMap[board[i, 2]]));
        }

        return Task.FromResult(sb.ToString());
    }

    /// <summary>
    /// Make a move on the board.
    /// </summary>
    /// <param name="player">The player making the move (1 for X, 2 for O).</param>
    /// <param name="row">The row to make the move.Must be between 0 and 2.</param>
    /// <param name="col">The column to make the move.Must be between 0 and 2.</param>
    /// <exception cref="ArgumentException"></exception>
    [Function]
    [KernelFunction]
    [Description("Make a move on the board.")]
    public async Task<string> MakeMove(
        [Description("The player making the move (1 for X, 2 for O).")]
        int player,
        [Description("The row to make the move. Must be between 0 and 2.")]
        int row,
        [Description("The column to make the move. Must be between 0 and 2.")]
        int col)
    {
        if (board[row, col] != 0)
        {
            return $"Invalid move. The cell ({row}, {col}) is already occupied.";
        }

        if (player != 1 && player != 2)
        {
            return "Invalid player. Player must be 1 or 2.";
        }

        board[row, col] = player;

        var sb = new StringBuilder();
        sb.AppendLine($"Player {player} made a move at ({row}, {col}).");

        return sb.ToString();
    }

    public bool CheckWin(int player)
    {
        // Check rows
        for (var i = 0; i < 3; i++)
        {
            if (board[i, 0] == player && board[i, 1] == player && board[i, 2] == player)
            {
                return true;
            }
        }

        // Check columns
        for (var i = 0; i < 3; i++)
        {
            if (board[0, i] == player && board[1, i] == player && board[2, i] == player)
            {
                return true;
            }
        }

        // Check diagonals
        if (board[0, 0] == player && board[1, 1] == player && board[2, 2] == player)
        {
            return true;
        }

        if (board[0, 2] == player && board[1, 1] == player && board[2, 0] == player)
        {
            return true;
        }

        return false;
    }
}

