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
	cp data/all-reps.xlsx website/
	cp -r fig website/
	ghp-import --no-jekyll --message="Update results site" --no-history --push website
	rm -r website/
