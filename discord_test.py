import discord

import logging

# logging.basicConfig(level=logging.INFO)

from discord.ext import commands

bot = commands.Bot(command_prefix='/')

@bot.command()
async def caps(ctx, *params):
    text_caps = str(list(params)).upper()
    await ctx.send(text_caps)

#class MyClient(discord.Client):
#    async def on_ready(self):
#        print('Logged on as {0}!'.format(self.user))
#
#    async def on_message(self, message):
#        print('Message from {0.author}: {0.content}'.format(message))
#        if message.author != self.user:
#            if message.content.lower().startswith("hello"):
#                await message.channel.send("Hello ! :3")

#client = MyClient()
#client = discord.Client()

#@client.event
#async def on_ready()
#   print('Logged on as {0}!'.format(client.user))

@bot.event
async def on_ready():
   print('Logged on as {0}!'.format(bot.user))

@bot.event
async def on_message(message):
    print('Message from {0.author}: {0.content}'.format(message))
    if message.author != bot.user:
        if message.content.lower().startswith("hello"):
            await message.channel.send("Hello ! :3")
    await bot.process_commands(message)

from discord_settings_local import TOKEN
bot.run(TOKEN)
#client.run(TOKEN)
