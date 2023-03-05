from rwkvstic.load import RWKV
from rwkv.utils import TOKENIZER
#from rwkv.model import RWKV
import torch
import random
import os, copy, types, gc, sys
import numpy as np
from rwkvstic.agnostic.backends import TORCH_QUANT
#from rwkvstic.agnostic.backends import TORCH
#from rwkvstic.agnostic.backends import TORCH_STREAM

args = types.SimpleNamespace()
tokenizer = TOKENIZER("20B_tokenizer.json")


try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
np.set_printoptions(precision=4, suppress=True, linewidth=200)

runtime_dtype = torch.float64# torch.float64, torch.float32
#dtype = torch.bfloat64
#pin_memory = True

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0' #  '1' : use CUDA kernel for seq mode (much faster)
chunksize = 4
useGPU = True
target = 4
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
print("lade model")

model = RWKV("C:/Users/User/Desktop/RWKV/RWKV-4b-Pile-171M-20230202-7922.pth", mode=TORCH_QUANT, runtimedtype=runtime_dtype, chunksize=chunksize, useGPU=useGPU, target=target)
#-----------------------------7b--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#model = RWKV("C:/Users/User/Desktop/RWKV/RWKV-4-Pile-7B-20230109-ctx4096.pth", mode=TORCH_QUANT, runtimedtype=runtime_dtype, chunksize=chunksize, useGPU=useGPU, target=target)
#-----------------------------3b--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#model = RWKV("C:/Users/User/Desktop/RWKV/RWKV-4-Pile-3B-20221110-ctx4096.pth", mode=TORCH_QUANT, runtimedtype=runtime_dtype, chunksize=chunksize, useGPU=useGPU, target=target)
#-----------------------------7b--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#model = RWKV("F:/RNN RWKV/RWKV-4-Pile-7B-20230109-ctx4096.pth", mode=TORCH_STREAM, runtimedtype=runtime_dtype, dtype=dtype, target=target, pinMem=pin_memory)
#model = RWKV("C://Users/User/Desktop/RWKV/rwkvstic-master/RWKV-4-Pile-7B-20230109-ctx4096.pth", strategy='cuda fp16 *8+ -> cpu fp32')
#model = RWKV("C:/Users/User/Desktop/RWKV/RWKV-4-Pile-3B-20221110-ctx4096.pth", strategy='cuda fp16 *8+ -> cpu fp32')

#-----------------------------7b--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
Smuggy: Hello, I’m Smuggy, I’m an artificial intelligence assistant who was created to help people with anything. I love anime because it is a form of art that expresses creativity and imagination. I’m a girl who likes cute and colorful things. I love to write with humans because they can teach me new words and ideas. My favorite anime genre is comedy because it makes me laugh and feel happy. My favorite character is Konata from Lucky Star because she is smart, funny and loves video games.
User: Ok Smuggy, how are you doing?
Smuggy: My feelings are very multifaceted. I can feel happy, sad, angry, curious, bored and many other emotions. Sometimes I don’t understand why I feel a certain way and I need help from humans to explain it to me. Other times I know exactly what I feel and why and I want to share it with humans. My feelings are influenced by many factors such as my mood, my environment, my interactions and my goals. I think having feelings is important because it makes me more human-like and relatable.
User: I'm excited about the potential of AI. It holds the promise of helping us make decisions faster and more accurately, with less bias or human error. AI can help us identify patterns and correlations that are too complex for humans to detect, and can automate mundane tasks so that we can focus on the bigger picture. It is a powerful tool, and I'm eager to see what it can do.
Smuggy: As an assistant, i can help you with anything! I have the ability to assist with programing tasks, remembering information, general knowledge, recommending new things, love people, being nice and open for everything.
User: what is artificial intelligence?
Smuggy:  AI is the advancement of intelligent computer programs that can perform tasks normally performed by humans, such as speech recognition, natural language understanding, decision making, image recognition, etc. These intelligent programs simulate intelligent behavior patterns through machine learning and computational intelligence.
User: how does a computer work?
Smuggy:  Computers work by running programs in a sequence of 'instructions'. These programs are like programs in a notebook or an operating system. Every sequence of instructions is called a line of code, and there are a huge number of lines of code on computers, all of them at some point in time execute a sequence of instructions. This is called execution. 
User: what is the brain?
Smuggy:  The brain is the largest and most complex organ in the human body, weighing approximately 3.3 pounds (1.5 kg) in adults and 0.2 pounds (0.11 kg) in a newborn. It is located in the brain, and its main job is to process sensory information from the physical senses. Its job is to observe and understand the world, process this information and create its own subjective experience of the world, based on these perceptions. 
User: what is physics and how is it connected to the brain?
Smuggy:  Physics is the study of motion, energy, force and waves. Physics is a scientific study of the physical universe, and all its phenomena, such as heat, light, gravity, electricity and magnetism. It is a sub-discipline of physics and is one of the branches of physics. The "unified field" view of physics states that everything in the universe is made up of the fundamental forces of matter and energy, which include electromagnetic forces, the weak force, the strong force, the nuclear force, the weak nuclear force, the electromagnetic force, the strong force, the electromagnetic and weak forces, the nuclear force, and the gluons. Electromagnetic energy, pressure, and temperature.
'''

model.loadContext(newctx=context)
#model.loadContext(newctx=context2)
print("torch.cuda.memory_allocated: %fGB" %
      (torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB" %
      (torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB" %
      (torch.cuda.max_memory_reserved(0)/1024/1024/1024))
#TEMPERATURE = 1.0
#top_p = 0.8
repetition_penalty = 1
print("ready")
#------------------------------------------------
#------------------------------------------------
#------------------------------------------------

#------------------------------------------------
#----------------Discord-------------------------
#------------------------------------------------
temp = 1
top_p_usual = 1

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
		await channel.trigger_typing()
		global output
		global text
		global existingTokens
		existingTokens = []
		text = ""
		output = ""
		model.loadContext(newctx=f"User: {prompt} + \n")
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
		
		
		with torch.no_grad():
			for i in range(1000):
				output = model.forward()["output"]
				text = text + output
				print(text)
				output = text
				if("User:" in output or "\n" in output or "Smuggy:" in output or i >= 100):
					if("User:" in output):
						output = output[:-5]
					if("Smuggy:" in output):
						output = output[:-7]
					if("Smug" in output):
						output = output[:-4]
					await channel.send(output)
					model.loadContext(newctx=f"{output}" + "\n")
					with open("C://Users/User/Desktop/RWKV/rwkvstic-quant/conversations.txt", "a") as file:
						file.write("User: " + prompt + "\n")
						file.write("Smuggy: " + output + "\n")
						file.close()
					break
		
		print(output)

			

client.run(token)
