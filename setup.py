from setuptools import setup, Extension

with open("README.md", "r") as fh:
	long_description = fh.read()

setup(
	name='pyaroma', # name of pip install
	version='0.1', # version number of unstable package
	description="A Framework to for developing Deep Leanring models", # describe your package
	py_modules=["nn/activations",
				"nn/backpropagation",
				"nn/forward",
				"nn/layers",
				"nn/losses",
				"nn/model",
				"nn/parameters",
				"optim/optimizers",
				"eval/evaluation",
				"utils/dataloader",
				"utils/process_tensor",
				"utils/transforms",
				"viz/visualization"], # list the files of all modules
	package_dir={'': 'src'}, # dir modules
	classifiers=[
				"Programming Language :: Python :: 3",
				"Programming Language :: Python :: 3.6",
				"Programming Language :: Python :: 3.7",
				"Operating System :: OS Independent",
				],
	long_description=long_description,
	long_description_content_type="text/markdown",
	install_requires=[
					 "numpy ~= 1.19",
					 "matplotlib ~= 3.3",
					 "pandas ~= 1.1",
					 "kaggle ~= 1.5",
					 "seaborn ~= 0.11",
					 ],
	extras_require= {
					"dev": [
						"tqdm>=4.50",
						],
					},
	)