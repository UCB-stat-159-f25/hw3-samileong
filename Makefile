ENV_NAME = ligo

ENV_FILE = environment.yml

.PHONY : env html clean

env :
	@if conda env list | grep -q "$(ENV_NAME)"; then \
		conda env update -f $(ENV_FILE) -n $(ENV_NAME); \
	else \
		conda env create -f $(ENV_FILE) -n $(ENV_NAME); \
	fi
    
html :
	# build the html rendering of MyST site 
	myst build --html 

clean :
	# clean up the figures, audio and _build folders 
	rm -rf figures/* audio/* _build/*