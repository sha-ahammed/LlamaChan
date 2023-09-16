import os

import discord
from dotenv import load_dotenv
from discord.file import File
from discord.commands import Option

from model_inference import inference

load_dotenv()  # load all the variables from the env file
bot = discord.Bot()
llama = inference()


@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")


# @bot.slash_command(name="code", description="Instruct CodeLlama to write code for you.")
# async def code(ctx, prompt: str):
#     await ctx.defer()
#     response = await llama.generate_code(prompt)
#     await ctx.respond(response)


# @bot.slash_command(name="chat", description="Chat with Llama 2!")
# async def chat(
#     ctx,
#     prompt: Option(str, "Query goes here", required=True),
# ):
#     await ctx.defer()
#     response = await llama.chat(prompt)
#     await ctx.respond(response)


@bot.slash_command(name="chat_gpt", description="Chat with GPT-3.5!")
async def chat_gpt(ctx, prompt: Option(str, "Query goes here", required=True)):
    await ctx.defer()
    response = await llama.ChatGPT(prompt)
    await ctx.respond(response)


@bot.slash_command(name="llama_2", description="Chat with Llama 2!")
async def llama_two(ctx, prompt: Option(str, "Query goes here", required=True)):
    await ctx.defer()
    response = await llama.Llama(prompt)
    await ctx.respond(response)


bot.run(os.getenv("TOKEN"))  # run the bot with the token
