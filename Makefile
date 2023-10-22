print_make_commands:
	@echo log:  - CLI to add a row to a user-selected file ending in .log
	@echo install:  - install dependencies
	@echo clean:  - remove build artifacts

log:  # CLI to add a row to a user-selected file ending in .log
	./tools/logger.sh

install:
	python -m pip install -r requirements.txt
	pip install -e .

clean:
	rm -f dbchat/src/dbchat.egg-info