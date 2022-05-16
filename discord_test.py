import discord

class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged on as {0}!'.format(self.user))

    async def on_message(self, message):
        print('Message from {0.author}: {0.content}'.format(message))
        if message.author != client.user:
            if message.content.lower().startswith("hello"):
                await message.channel.send("Hello ! :3")

client = MyClient()

from discord_settings_local import TOKEN
client.run(TOKEN)
