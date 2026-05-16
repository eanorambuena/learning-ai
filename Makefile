.PHONY: install run i

i:
	conda run -n learning-ai pip install -r requirements.txt

run:
	conda run -n learning-ai python $(file)
