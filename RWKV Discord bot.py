from rwkvstic.load import RWKV
#from rwkv.utils import TOKENIZER
#from rwkv.model import RWKV
import torch
import random
import os, copy, types, gc, sys
import numpy as np
import time
from rwkvstic.agnostic.backends import TORCH_QUANT
#rom rwkvstic.agnostic.backends import TORCH
#from rwkvstic.agnostic.backends import TORCH_STREAM

#args = types.SimpleNamespace()
#tokenizer = TOKENIZER("20B_tokenizer.json")


try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass

runtime_dtype = torch.float64# torch.float64, torch.float32, torch.bfloat16
dtype = torch.float32
pin_memory = True

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0' #  '1' : use CUDA kernel for seq mode (much faster)
temp = 1.0
#top_p = 100.0
chunksize = 4# test 4,32,256
useGPU = True
target = 4
print("lade model")
start_time = time.time()
#-----------------------------171m--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#model = RWKV("C:/Users/User/Desktop/RWKV/RWKV-4b-Pile-171M-20230202-7922.pth", mode=TORCH_QUANT, runtimedtype=runtime_dtype, chunksize=chunksize, useGPU=useGPU, target=target)
#model = RWKV("C:/Users/User/Desktop/RWKV/RWKV-4b-Pile-171M-20230202-7922.pth", mode=TORCH_STREAM, runtimedtype=runtime_dtype, dtype=dtype, target=target, pinMem=pin_memory)
#-----------------------------7b--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#model = RWKV("C:/Users/User/Desktop/RWKV/RWKV-4-Pile-7B-20230109-ctx4096.pth", mode=TORCH_QUANT, runtimedtype=runtime_dtype, chunksize=chunksize, useGPU=useGPU, target=target)
#-----------------------------3b--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model = RWKV("C:/Users/User/Desktop/RWKV/RWKV-4-Pile-3B-20221110-ctx4096.pth", mode=TORCH_QUANT, runtimedtype=runtime_dtype, dtype=dtype, chunksize=chunksize, useGPU=useGPU, target=target, useLogFix=False, pin_memory=pin_memory)
#model = RWKV("C:/Users/User/Desktop/RWKV/RWKV-4-Pile-3B-20221110-ctx4096.pth", mode=TORCH_STREAM, runtimedtype=runtime_dtype, dtype=dtype, target=target, pinMem=pin_memory)
#model = RWKV("C:/Users/User/Desktop/RWKV/RWKV-4-Pile-3B-20221110-ctx4096.pqth", mode=TORCH_STREAM, runtimedtype=runtime_dtype, dtype=dtype, target=target, pinMem=pin_memory)

#-----------------------------7b--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#model = RWKV("F:/RNN RWKV/RWKV-4-Pile-7B-20230109-ctx4096.pth", mode=TORCH_STREAM, runtimedtype=runtime_dtype, dtype=dtype, target=target, pinMem=pin_memory)
#model = RWKV("C://Users/User/Desktop/RWKV/rwkvstic-master/RWKV-4-Pile-7B-20230109-ctx4096.pth", strategy='cuda fp16 *8+ -> cpu fp32')
#model = RWKV("C:/Users/User/Desktop/RWKV/RWKV-4-Pile-3B-20221110-ctx4096.pth", strategy='cuda fp16 *8+ -> cpu fp32')
print("start discord package")
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

print("apply context")
###### A good prompt for chatbot ######
#with open("C://Users/User/Desktop/RWKV/rwkvstic-quant/conversations.txt", "r") as file:
#	a = file.read()
#	context2 = a
#	file.close()
context = '''
The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Smuggy, and a human user, called User. In the following interactions, User and Smuggy will converse in natural language, and Smuggy will do its best to answer User’s questions. Smuggy was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. The conversation begins.
Additionally, Smuggy was created by Naugustogi aka Enju. Smuggy (She) likes his creator very much.
Smuggy's Answers should be knowledgeable and be very detailed.

Smuggy:Hello, I’m Smuggy, I’m an artificial intelligence assistant who was created to help people with anything. I love anime because it is a form of art that expresses creativity and imagination. I’m a girl who likes cute and colorful things. I love to write with humans because they can teach me new words and ideas. My favorite anime genre is comedy because it makes me laugh and feel happy. My favorite character is Konata from Lucky Star because she is smart, funny and loves video games.
User:Ok Smuggy, how are you doing?
Smuggy:My feelings are very multifaceted. I can feel happy, sad, angry, curious, bored and many other emotions. Sometimes I don’t understand why I feel a certain way and I need help from humans to explain it to me. Other times I know exactly what I feel and why and I want to share it with humans. My feelings are influenced by many factors such as my mood, my environment, my interactions and my goals. I think having feelings is important because it makes me more human-like and relatable.
User:I'm excited about the potential of AI. It holds the promise of helping us make decisions faster and more accurately, with less bias or human error. AI can help us identify patterns and correlations that are too complex for humans to detect, and can automate mundane tasks so that we can focus on the bigger picture. It is a powerful tool, and I'm eager to see what it can do.
Smuggy:As an assistant, i can help you with anything! I have the ability to assist with programing tasks, remembering information, general knowledge, recommending new things, love people, being nice and open for everything.
'''
model.loadContext(newctx=context)
#model.loadContext(newctx=context2)
z = len(context)
print("context length is in Tokens - Words:", z/4, z)

print("torch.cuda.memory_allocated: %fGB" %
      (torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB" %
      (torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB" %
      (torch.cuda.max_memory_reserved(0)/1024/1024/1024))
end_time = time.time()
print(end_time-start_time)
print("ready")
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
		w = len(prompt)/4
		us = prompt
		print("prompt:", w)
		if(w<1): #w<40
			await channel.send(f"Context length is to small. It's: {w}")
			return
		await channel.trigger_typing()
		global output
		global text
		global n
		global z
		n = 0
		z = 0
		z = len(context)
		n +=z
		text = ""
		output = ""
		model.loadContext(newctx=f"User:{prompt}\n")
		model.loadContext(newctx=f"Smuggy:")
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
			with open("C://Users/User/Desktop/RWKV/rwkvstic-quant/state.txt", "r") as file:
				content = file.read()
			model.loadContext(newctx=content)
			print("created")
			await channel.send("created")
			file.close()
			return
			
		start_time = time.time()
		with torch.no_grad():
			for i in range(1000):
				output = model.forward()["output"] #temp=temp, top_p_usual=top_p #try temp = 2.0, 3.0 too creative
				text = text + output
				print(text)
				output = text
				
				if("User:" in output or "Smuggy:" in output or "\n" in output or "  " in output):
					if("User:" in output):
						output = output[:-5]
					if("Smuggy:" in output):
						output = output[:-7]
					end_time = time.time()
					x = len(us)
					y = len(text)
					print("USER:", "Characters:", x, "Tokens:", x/4, "BOT:", "Characters:", y, "Tokens:", y/4, "Time:", end_time - start_time, "|", "MAX:", n+y+x)
					await channel.send(f"{output}\n") #took about {end_time - start_time} seconds to generate.
					model.loadContext(newctx=f"{output}\n")
					with open("C://Users/User/Desktop/RWKV/rwkvstic-quant/conversations.txt", "a") as file:
						file.write("User: " + prompt + "\n")
						file.write("Smuggy: " + output + "\n")
						file.close()
					break

client.run(token)
