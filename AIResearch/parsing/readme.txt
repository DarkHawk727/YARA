Purpose of parse is to take local and/or online files and parse them into smaller chunks of token for
GPT's learning purpose. an optimizer (not yet deployed) will try to boost utility rate in each input batch
by boosting input closer to max. token input limit of a given GPT model. 


1. download python 3 
2. download dependency:
pip3 install -r requirements.txt

3. to run the code:
quick demo: 
currently using one file (see dir at current level) as input, hard-coded.
later will allow for Command like tool input and batch input (i.e. input of a list of files)

Work in progress:

Optimizer
web crawler 
multiple format parser
Comand line tool allowing user input

(...and more if feedback arises).