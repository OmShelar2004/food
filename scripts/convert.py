"""Convert a Keras HDF5 model (.h5) to the Keras v3 folder/file format (.keras).

Usage:
  python convert.py [input.h5] [output.keras]

If no args provided the script uses:
  input: efficientnetb0_food101.h5
  output: efficientnetb0_food101.keras
"""
import os
import sys

try:
	# Local import - will raise if tensorflow not available
	from tensorflow.keras.models import load_model
except Exception as e:
	print("Error: TensorFlow/Keras not available in this environment:\n", e)
	sys.exit(1)


def convert(in_path: str, out_path: str) -> int:
	if not os.path.exists(in_path):
		print(f"Input file not found: {in_path}")
		return 2

	print(f"Loading model from: {in_path}")
	try:
		model = load_model(in_path, compile=False)
	except Exception as e:
		print("Failed to load .h5 model:\n", e)
		return 3

	print(f"Saving model to: {out_path}")
	try:
		# Keras will infer format from extension; using .keras writes the new Keras format
		model.save(out_path)
	except Exception as e:
		print("Failed to save model as .keras:\n", e)
		return 4

	print("Conversion complete.")
	return 0


def main():
	in_file = sys.argv[1] if len(sys.argv) > 1 else "efficientnetb0_food101.h5"
	out_file = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(in_file)[0] + ".keras"

	rc = convert(in_file, out_file)
	sys.exit(rc)


if __name__ == "__main__":
	main()
