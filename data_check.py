import pandas as pd
import numpy as np
import glob
import os

def main():
	# Get subdirectories
	sub_dir = [x[0] for x in os.walk('./')]
	print(sub_dir)



if __name__ == "__main__":
	main()