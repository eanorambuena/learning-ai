.PHONY: install run i

i:
	conda run -n tfenv pip install -r requirements.txt

run:
	conda run -n tfenv python $(file)
