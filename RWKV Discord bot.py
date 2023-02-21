from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import TORCH_QUANT
#from rwkvstic.agnostic.backends import TF
#from rwkvstic.agnostic.backends import TORCH_STREAM
import torch
import random

runtime_dtype = torch.float32# torch.float64, torch.bfloat16, torch.float32
#dtype = torch.float32
#pin_memory = True
chunksize = 4
useGPU = True
target = 4

#model = RWKV("C://Users/User/Desktop/RWKV/rwkvstic-master/RWKV-4-Pile-7B-20230109-ctx4096.pth", mode=TORCH_STREAM, runtimedtype=runtime_dtype, dtype=dtype, target=target, pinMem=pin_memory)
model = RWKV("C://Users/User/Desktop/RWKV/rwkvstic-master/RWKV-4-Pile-3B-20221110-ctx4096.pth", mode=TORCH_QUANT, runtimedtype=runtime_dtype, chunksize=chunksize, useGPU=useGPU, target=target)
#model = RWKV("C://Users/User/Desktop/RWKV/rwkvstic-master/RWKV-4-Pile-3B-20221110-ctx4096.pth", mode=TORCH_STREAM, runtimedtype=runtime_dtype, dtype=dtype, target=target, pinMem=pin_memory)

import ast
from difflib import SequenceMatcher
from discord.ext import commands
import time 
import re
import discord
import io

token = ""
intents = discord.Intents.default() #all
client = discord.Client(intents=intents)


###### A good prompt for chatbot ######
#with open("C://Users/User/Desktop/RWKV/rwkvstic-master/conversations.txt", "r") as file:
#	a = file.read()
#	context = a
#	file.close()
context = '''
The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Smuggy, and a human user, called User. In the following interactions, User and Smuggy will converse in natural language, and Smuggy will do its best to answer User’s questions. Smuggy was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. The conversation begins.
Additionally, Smuggy was created by Naugustogi aka Enju. Smuggy (She) likes his creator very much.
Smuggy's Answers should be knowledgeable and be very detailed.

Smuggy: Hello, I'm smuggy, i'm an Artificial intelligence Assistant. I love Anime. I'm a girl. I love to write with humans.

User: Ok Smuggy, I’m going to start by quizzing you with a few warm-up questions. How are you doing?

Smuggy: I'm good, i feel very weird being an AI.

User: what does "emptyState = model.emptyState() and model.setState(emptyState)" do?

Smuggy: The code emptyState = model.emptyState() creates a new state object with all its properties initialized to their default values. 
This is done by calling a method called emptyState() on the model object. 

User: That was very helpful.

Smuggy: I was build to be helpful.
You can ask me anything.

User: how can i lucid dream about a girl?

Smuggy:To have a lucid dream about a specific person, such as a girl, try the following tips:
1. Focus on the person before sleeping. Visualize them in your mind, imagine talking to them and interacting with them.

User: Thank you!

Smuggy: No problem! Do you have any other questions or needs?
'''

model.loadContext(newctx=context)
print("torch.cuda.memory_allocated: %fGB" %
      (torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB" %
      (torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB" %
      (torch.cuda.max_memory_reserved(0)/1024/1024/1024))
#TEMPERATURE = 1.0
#top_p = 0.8

#------------------------------------------------
#----------------Discord-------------------------
#------------------------------------------------


@client.event
async def on_message(message):
	print("def loaded")
	if message.author == client.user:
		return
#------------------------------------------------		
	if  client.user.mentioned_in(message):

		prompt = message.content
		prompt = prompt[23:]
		channel = message.channel
		global output
		global text
		text = ""
		output = ""
		model.loadContext(ctx="\n", newctx=f"User: {prompt}")	
		model.loadContext(ctx="\n", newctx=f"Smuggy:")

		if("currentmem" in prompt):
			currentMem = model.getState()
			print(currentMem)
			await channel.send("currentmem is in console")
			return
		if("changemode" in prompt):
			emptyState = model.emptyState()
			model.setState(emptyState)
			print("deleted")
			await channel.send("deleted")
			return
			
		if("addnewstate" in prompt):
			with open("C://Users/User/Desktop/RWKV/rwkvstic-master/state.txt", "r") as file:
				content = file.read()
			model.loadContext(newctx=content)
			print("created")
			await channel.send("created")
			file.close()
			return
			
		with torch.no_grad():
			for i in range(1000):
				output = model.forward()["output"]
				text = text + output
				print(text)
				output = text
				if("User" in output or i == 24 or "Smuggy" in output or "  " in output or "Smug" in output or "user" in output or "Sm" in output):
					if("User" in output or "user" in output or "Smug" in output or "smug" in output):
						output = output[:-4]
					if("Smuggy" in output):
						output = output[:-6]
					if("Sm" in output):
						output = output[:-2]
					await channel.send(output)
					model.loadContext(newctx=f" {output}")

					with open("C://Users/User/Desktop/RWKV/rwkvstic-master/conversations.txt", "a") as file:
						file.write("User: " + prompt + "\n")
						file.write("Smuggy: " + output + "\n")
						file.close()
					break
		

		print(output)

			

client.run(token)