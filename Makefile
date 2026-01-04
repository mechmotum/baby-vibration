FIRST_DIFF_TAG = v8
NETWORKDIR := $(shell grep -A3 'network-data-directory:' config.yml | tail -n1 | cut -c 25-)
rsync:
	# --archive, -a            archive mode is -rlptgoD (no -A,-X,-U,-N,-H)
	# --verbose, -v            increase verbosity
	# --compress, -z           compress file data during the transfer
	# --human-readable, -h     output numbers in a human-readable format
	# --delete                 delete extraneous files from dest dirs
	# --exclude=PATTERN
	#  trailing slash on source means direcotry
	#  Trillingen-Project_Dec2024/raw_data/* will match baby-vibration-data/*
	rsync -avzh --delete --delete-excluded "$(NETWORKDIR)" /home/moorepants/Data/baby-vibration-data
pushwebsite:
	mkdir website
	cp index.html website/
	cp data/*.xlsx website/
	cp -r fig website/
	ghp-import --no-jekyll --message="Update results site" --no-history --push website
	rm -r website/
trackchanges:
	git checkout $(FIRST_DIFF_TAG)
	cp main.tex $(FIRST_DIFF_TAG).tex
	git checkout master
	latexdiff $(FIRST_DIFF_TAG).tex main.tex > diff-master_$(FIRST_DIFF_TAG).tex
	rm $(FIRST_DIFF_TAG).tex
	git checkout -- main.tex
	pdflatex -interaction nonstopmode diff-master_$(FIRST_DIFF_TAG).tex
	bibtex diff-master_$(FIRST_DIFF_TAG).aux
	pdflatex -interaction nonstopmode diff-master_$(FIRST_DIFF_TAG).tex
	pdflatex -interaction nonstopmode diff-master_$(FIRST_DIFF_TAG).tex
