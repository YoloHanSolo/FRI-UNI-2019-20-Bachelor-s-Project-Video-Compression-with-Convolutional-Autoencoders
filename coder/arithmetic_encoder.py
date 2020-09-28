# 
# Reference arithmetic coding
# Copyright (c) Project Nayuki
# 
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
# 

# Modified by Jan Pelicon

import contextlib, sys, os, torch
import coder.arithmeticcoding as arithmeticcoding
import numpy as np

class ArithmeticCoder():
	def __init__(self, tensor_shape, output_name, adaptive=False, stats=0):
		self.tensor_shape = tensor_shape
		self.output_name = output_name
		self.adaptive = adaptive
		self.resolution = 256 * 256
		self.size = tensor_shape[0] * tensor_shape[1] * tensor_shape[2] + stats

		self.filesize = 0
		self.filesize_avg = 0
		self.files = 0
		self.bpp = 0

		self._code = None
		print("Init = Arithmetic Encoder\n\tAdaptive = {}\n\tSize = {}\n\tTensor Shape = {}\n".format(self.adaptive, self.size, self.tensor_shape))

	def Encode(self, index, component):
		data_in = component._code
		file_name = "{}_{}.code".format(self.output_name, index)
		
		if self.adaptive:
			with contextlib.closing(arithmeticcoding.BitOutputStream(open(file_name, "wb"))) as bitout:
				self.adaptive_compress(data_in, bitout)
		else:
			freqs = self.get_frequencies(data_in)
			freqs.increment(256)		
			with contextlib.closing(arithmeticcoding.BitOutputStream(open(file_name, "wb"))) as bitout:
				self.write_frequencies(bitout, freqs)
				self.compress(freqs, data_in, bitout)		
		self.filesize = os.path.getsize(file_name)
		self.filesize_avg += self.filesize
		self.files += 1

	def Decode(self, index):
		data_out = None
		file_name = "{}_{}.code".format(self.output_name, index)
		if self.adaptive:
			with open(file_name, "rb") as inp:
				bitin = arithmeticcoding.BitInputStream(inp)
				data_out = self.adaptive_decompress(bitin)
		else:
			with open(file_name, "rb") as inp:
				bitin = arithmeticcoding.BitInputStream(inp)
				freqs = self.read_frequencies(bitin)
				data_out = self.decompress(freqs, bitin)
		self._code = data_out

	def AvgFileSize(self):
		self.bpp = (self.filesize_avg*8/self.files)/self.resolution
		print("Average file size = {:.0f} B".format(self.filesize_avg/self.files))
		print("Bits per pixel = {:.3f} bpp".format(self.bpp))
		print("Number of Files = {}".format(self.files))

	def get_frequencies(self, vector):
		freqs = arithmeticcoding.SimpleFrequencyTable([0] * 257)	
		for index in range(len(vector)):
			freqs.increment(vector[index])
		return freqs

	def write_frequencies(self, bitout, freqs):
		for i in range(256):
			self.write_int(bitout, 32, freqs.get(i))

	def read_frequencies(self, bitin):
		def read_int(n):
			result = 0
			for _ in range(n):
				result = (result << 1) | bitin.read_no_eof()  # Big endian
			return result
		
		freqs = [read_int(32) for _ in range(256)]
		freqs.append(1)  # EOF symbol
		return arithmeticcoding.SimpleFrequencyTable(freqs)

	def write_int(self, bitout, numbits, value):
		for i in reversed(range(numbits)):
			bitout.write((value >> i) & 1)

	"""
	DEFAULT VERSION
	"""

	def compress(self, freqs, vector, bitout):
		enc = arithmeticcoding.ArithmeticEncoder(32, bitout)

		for index in range(len(vector)):
			enc.write(freqs, vector[index])
		enc.write(freqs, 256)
		enc.finish()

	def decompress(self, freqs, bitin):
		dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
		output_array = np.ndarray((self.size)).astype(np.uint8)
		index = 0
		while True:
			symbol = dec.read(freqs)
			if symbol == 256:  # EOF symbol
				break
			output_array[index] = symbol
			index += 1
		return output_array
	
	"""
	ADAPTIVE VERSION
	"""
	def adaptive_compress(self, input_vector, bitout):
		initfreqs = arithmeticcoding.FlatFrequencyTable(257)
		freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
		enc = arithmeticcoding.ArithmeticEncoder(32, bitout)

		for index in range(len(input_vector)):
			enc.write(freqs, input_vector[index])
			freqs.increment(input_vector[index])
		enc.write(freqs, 256)
		enc.finish()

	def adaptive_decompress(self, bitin):
		initfreqs = arithmeticcoding.FlatFrequencyTable(257)
		freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
		dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
		output_array = np.ndarray((self.size)).astype(np.uint8)	
		index = 0	
		while True:
			# Decode and write one byte
			symbol = dec.read(freqs)
			if symbol == 256:  # EOF symbol
				break
			output_array[index] = symbol
			index += 1
			freqs.increment(symbol)
		return output_array

		
