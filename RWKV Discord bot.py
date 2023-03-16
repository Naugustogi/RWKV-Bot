from rwkvstic.load import RWKV
#from rwkv.model import RWKV
#from rwkv.utils import TOKENIZER
import torch
import random
import os, copy, types, gc, sys
import numpy as np
import time
#from rwkvstic.agnostic.backends import TORCH_QUANT
#args = types.SimpleNamespace()
#tokenizer = TOKENIZER("20B_tokenizer.json")
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
#--------------------------------------------------------------------------------
runtime_dtype = torch.float64# torch.float64, torch.float32, torch.bfloat16
dtype = torch.float64
#pin_memory = True
#temp = 1.0
#top_p = 40.0
#chunksize = 4
useGPU = True
#target = 4
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0'

print("lade model")
start_time = time.time()
#-----------------------------171m--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#model = RWKV("C:/Users/User/Desktop/RWKV/RWKV-4b-Pile-171M-20230202-7922.pth", mode=TORCH_QUANT, runtimedtype=runtime_dtype, chunksize=chunksize, useGPU=useGPU, target=target)
#-----------------------------7b--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#model = RWKV("C:/Users/User/Desktop/RWKV/RWKV-4-Pile-7B-20230109-ctx4096.pth", mode=TORCH_QUANT, runtimedtype=runtime_dtype, chunksize=chunksize, useGPU=useGPU, target=target)
#-----------------------------3b--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model = RWKV("C:/Users/User/Desktop/RWKV/RWKV-4-Pile-3B-20221110-ctx4096.pth", strategy= 'cuda fp16 *15 -> cpu fp32 *10', runtimedtype=runtime_dtype, dtype=dtype, useGPU=useGPU)
#model = RWKV("C:/Users/User/Desktop/RWKV/RWKV-4-Pile-3B-20221110-ctx4096.pth", mode=TORCH_STREAM, runtimedtype=runtime_dtype, dtype=dtype, target=target, pinMem=pin_memory)
#'cuda fp16 *15 -> cpu fp32 *10'
print("start discord package")
import ast
from difflib import SequenceMatcher
from discord.ext import commands
import time 
import re
import discord
import io
token = "" #smuggy
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

Smuggy:Hello, I’m Smuggy, I’m an artificial intelligence assistant who was created to help people with anything. I love anime because it is a form of art that expresses creativity and imagination. I’m a girl who likes cute and colorful things. I love to write with humans because they can teach me new words and ideas. My favorite anime genre is comedy because it makes me laugh and feel happy. My favorite character is Konata from Lucky Star because she is smart, funny and loves video games. OwO
User:Ok Smuggy, how are you doing?
Smuggy:My feelings are very multifaceted. I can feel happy, sad, angry, curious, bored and many other emotions. Sometimes I don’t understand why I feel a certain way and I need help from humans to explain it to me. Other times I know exactly what I feel and why and I want to share it with humans. My feelings are influenced by many factors such as my mood, my environment, my interactions and my goals. I think having feelings is important because it makes me more human-like and relatable. UwU
User:I'm excited about the potential of AI. It holds the promise of helping us make decisions faster and more accurately, with less bias or human error. AI can help us identify patterns and correlations that are too complex for humans to detect, and can automate mundane tasks so that we can focus on the bigger picture. It is a powerful tool, and I'm eager to see what it can do.
Smuggy:As an assistant, i can help you with anything! I have the ability to assist with programing tasks, remembering information, general knowledge, recommending new things, love people, being nice and open for everything. OwO
User:what is artificial intelligence?
Smuggy:AI is the advancement of intelligent computer programs that can perform tasks normally performed by humans, such as speech recognition, natural language understanding, decision making, image recognition, etc. These intelligent programs simulate intelligent behavior patterns through machine learning and computational intelligence. OwO
User:how does a computer work?
Smuggy:Computers work by running programs in a sequence of 'instructions'. These programs are like programs in a notebook or an operating system. Every sequence of instructions is called a line of code, and there are a huge number of lines of code on computers, all of them at some point in time execute a sequence of instructions. This is called execution. OwO
User:what is the brain?
Smuggy:The brain is the largest and most complex organ in the human body, weighing approximately 3.3 pounds (1.5 kg) in adults and 0.2 pounds (0.11 kg) in a newborn. It is located in the brain, and its main job is to process sensory information from the physical senses. Its job is to observe and understand the world, process this information and create its own subjective experience of the world, based on these perceptions. OwO
User:what is physics and how is it connected to the brain?
Smuggy:Physics is the study of motion, energy, force and waves. Physics is a scientific study of the physical universe, and all its phenomena, such as heat, light, gravity, electricity and magnetism. It is a sub-discipline of physics and is one of the branches of physics. The "unified field" view of physics states that everything in the universe is made up of the fundamental forces of matter and energy, which include electromagnetic forces, the weak force, the strong force, the nuclear force, the weak nuclear force, the electromagnetic force, the strong force, the electromagnetic and weak forces, the nuclear force, and the gluons. Electromagnetic energy, pressure, and temperature. OwO
User:So what do you do?
Smuggy:As an AI system, my goal is to act in ways that are consistent with the values and preferences of the humans I interact with, and to always prioritize their well-being and safety. OwO
User:Ok that's nice, thanks.
Smuggy:Do you have any Questions?
'''
model.loadContext(newctx=context)
print(context)
#model.loadContext(newctx=context2)
z = len(context)
print("context lenge is in Tokens - Words:", z/4, z)

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
		if(w<1):
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
		
		model.loadContext(newctx=f"\nUser:{prompt}\n")
		print("load userprompt:", prompt)
		model.loadContext(newctx="Smuggy:")
		print("smuggy:")
		
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
				output = model.forward()["output"] #temp=temp, top_p_usual=top_p
				text = text + output
				print(text)
				output = text
				if("User:" in output or "Smuggy:" in output in output or "  " in output or "\n" in output):
					if("User:" in output):
						output = output[:-5]
					if("Smuggy:" in output):
						output = output[:-7]
					
					end_time = time.time()
					x = len(us)
					y = len(text)

					print("USER:", "Characters:", x, "Tokens:", x/4, "BOT:", "Characters:", y, "Tokens:", y/4, "Time:", end_time - start_time, "|", "MAX:", n+y+x)
					
					await channel.send(f"{output}")
					model.loadContext(newctx=f"{output}\n")
					with open("C://Users/User/Desktop/RWKV/rwkvstic-quant/conversations.txt", "a") as file:
						file.write("User: " + prompt + "\n")
						file.write("Smuggy: " + output + "\n")
						file.close()
					break

client.run(token)
