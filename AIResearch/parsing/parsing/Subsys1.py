
"""
Subsys1 command:

python Subsys1.py path_to_file1.pdf path_to_file2.pdf 

python Subsys1.py --debug path_to_files

M:\gpt_rl\parsing\parsing\pedpolicies-storm-drainage-policy.pdf

Idea, once parsed, write it to file

Subsys 1 allows batch sources file processing. 
use fileName2023_11_14[1] as fileName2023_11_14 chunk 1 data
"""

import os
from PDFparser import PDFparser
from Optimizer import Optimizer
from Token_count import Token_count


class Subsys1:
    def __init__(self, path, debug=False, token_limit=3600):
        self.path = path
        self.debug = debug
        self.model_token_limit=token_limit
        self.token=[]
        self.files_doc=[]
        self.splitContent=[]
        self.splitContentOptimized=[] # to fill after optimizer performed and gives how chunks should fit together making a larger chunk to boost utility rate
    



    def check_file_or_directory(self):
        result = None
        if os.path.isfile(self.path):
            result = f"{self.path} is a file."
            self.files_doc.append(self.path)
        elif os.path.isdir(self.path):
            result = f"{self.path} is a directory."
        else:
            result = f"{self.path} is neither a file nor a directory."

        if self.debug:
            print(result)
        #return result

    def list_files_in_directory(self): #if it's a directory
        print("list_files_in_directory")
        files = None
        if os.path.isdir(self.path):
            files = os.listdir(self.path)
            print(f"files= {files} /n")
            for file in files:
                print("adding /n")
                self.files_doc.append((os.path.join(self.path, file))) # add dir file to list
            if self.debug: #printing only
                if files:
                    print(f"Files in {self.path}:")
                    for file in files:
                        print(os.path.join(self.path, file))
                else:
                    print(f"{self.path} is an empty directory.")
        #return files

    def build_file_doc(self):
        self.check_file_or_directory()
        self.list_files_in_directory()
    
    def sys1(self):
        

        file_path_pdf =self.path

        pdf_parser=PDFparser()
        token_counter=Token_count()
        optimizer= Optimizer()

        tokens=[]

        chunk_index=0
        for file_path_pdf in self.files_doc:
         
            modified_pdf = file_path_pdf.replace('\\', '\\\\')
            print("modified_pdf")
            print(modified_pdf)
            print(self.files_doc)
            pdf_paragraphs=pdf_parser.extract_paragraphs_2(modified_pdf) 
            for pdf_para in pdf_paragraphs:
                tokens.append(token_counter.num_tokens_from_string(pdf_para))
                self.splitContent.append(pdf_para) #assume splitted content is smallest unit of content, <<< model token limit
            
            self.token.append(tokens) #append function token to class token 

        #Debug     
        #print tokens and their index
        for (i, item) in enumerate(tokens, start=0):
            print (i, item)

        subsets=optimizer.optimize(tokens, self.model_token_limit) #grouping of split content index
        print(f' "subset from optimize" {subsets}')

        for s in subsets:
            if isinstance(s, list): #s is a list
                for s1 in s:
                    self.splitContentOptimized.append(self.splitContent[s1])
                    #print("split content")
                    #print(s1)  
                    #print(self.splitContentOptimized)
            else:
                self.splitContentOptimized.append(self.splitContent[s])

                    

        sum=[]

        for i in subsets:
            currentSum=0
            for j in i:
                currentSum+=tokens[j]
            sum.append(currentSum)


            
        print(f' "Optimized result: " {sum}')

        #calculation of utility rate: 
        ur=0
        for u in sum:
            ur+= u/self.model_token_limit

        ur=ur/len(sum)
        print( f' "utility rate = " {ur}')

    def write_tokens_to_file(self, filename):
        # Construct the file path in the current directory
        current_directory = os.getcwd()
        file_path = os.path.join(current_directory, filename)

        with open(file_path, 'w') as file:
            #file.write("[")
            for token_list in self.splitContentOptimized:
                modified_token_list = token_list.replace("\n", " ")  # Assign the modified string to a new variable
                file.write(  modified_token_list + "\n\n\n")
            #file.write("]")










if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check if a path is a file or directory.")
    parser.add_argument("path", help="Path to check")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    subsys1 = Subsys1(args.path, debug=args.debug)
    subsys1.list_files_in_directory()
    result = subsys1.check_file_or_directory()
   
    subsys1.sys1()
    # Write the tokens to a file in the current directory
    subsys1.write_tokens_to_file("tokens.txt")



   




