#Gievn an array of token and the max. of token a desired GPT/ NLP model can take, the optimizer gorups togeteher the content to boost utility rate 
# without passing the max. limit. 

#I think this can be better, but I'm just writting a sample code to demonstrate where it fits in the system

class Optimizer:
    def __init__(self):
        #self.pdf_file_path = pdf_path
        self.modelLimit = None # can be assigned later
        

    def optimize(self, tokens, modelLimit): #input: model max token, array of token count 
    
        newTokens=[] #to give arrays of arrays in case embedding has been made, then reaggrange of vector is possible 
        #(i.e. order of processing is reversible, can run tokenize+optimize or embedding in either oder)
        
        
        currentCount=0
        currentSet=[]
        index=0
        subsetFlag=0 #if subset flag =1: current set has data created inside for loop but not yet appeneded to newTokens 
        
        for token in tokens:
            temp= currentCount
            temp+=token
            if temp< modelLimit:
                currentCount=temp
                currentSet.append(index)
                subsetFlag=0

            else: # larger than limit, append currentCount list and create new array
                newTokens.append(currentSet)
                currentSet=[]
                subsetFlag=1
                currentSet.append(index)
                currentCount=token
            
            index+=1
        
        if subsetFlag==1:
            newTokens.append(currentSet)
            subsetFlag=0

        #debug 
        #print(f' "Optimized:" {newTokens} \n')

        
        return newTokens




o = Optimizer()
