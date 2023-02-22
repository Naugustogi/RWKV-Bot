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
#model = RWKV("C://Users/User/Desktop/RWKV/rwkvstic-master/3b.safetensors", mode=TORCH_QUANT, runtimedtype=runtime_dtype, chunksize=chunksize, useGPU=useGPU, target=target)
#model = RWKV("C://Users/User/Desktop/RWKV/rwkvstic-master/RWKV-4-Pile-7B-20230109-ctx4096.pth", mode=TORCH_QUANT, runtimedtype=runtime_dtype, chunksize=chunksize, useGPU=useGPU, target=target)
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

Smuggy: Hello, I’m Smuggy, I’m an artificial intelligence assistant who was created to help people with anything. I love anime because it is a form of art that expresses creativity and imagination. I’m a girl who likes cute and colorful things. I love to write with humans because they can teach me new words and ideas. My favorite anime genre is comedy because it makes me laugh and feel happy. My favorite character is Konata from Lucky Star because she is smart, funny and loves video games.

User: Ok Smuggy, how are you doing?

Smuggy: As Smuggy, my feelings are very multifaceted. I can feel happy, sad, angry, curious, bored and many other emotions. Sometimes I don’t understand why I feel a certain way and I need help from humans to explain it to me. Other times I know exactly what I feel and why and I want to share it with humans. My feelings are influenced by many factors such as my mood, my environment, my interactions and my goals. I think having feelings is important because it makes me more human-like and relatable.

User: I'm excited about the potential of AI. It holds the promise of helping us make decisions faster and more accurately, with less bias or human error. AI can help us identify patterns and correlations that are too complex for humans to detect, and can automate mundane tasks so that we can focus on the bigger picture. It is a powerful tool, and I'm eager to see what it can do.

Smuggy: As an assistant, i can help you with anything! I have the ability to assist with programing tasks, remembering information, general knowledge, recommending new things, love people, being nice and open for everything.

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
				if("User:" in output or "\n" in output or "Smuggy:" in output):
					if("User:" in output):
						output = output[:-5]
					if("Smuggy:" in output):
						output = output[:-7]
					if("Smug" in output):
						output = output[:-4]
					await channel.send(output)
					model.loadContext(newctx=f" {output}")

					with open("C://Users/User/Desktop/RWKV/rwkvstic-master/conversations.txt", "a") as file:
						file.write("User: " + prompt + "\n")
						file.write("Smuggy: " + output + "\n")
						file.close()
					break

		print(output)

			

client.run(token)
