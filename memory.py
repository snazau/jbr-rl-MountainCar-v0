import random


class Memory:
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, element):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = element
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return list(zip(*random.sample(self.memory, batch_size)))
