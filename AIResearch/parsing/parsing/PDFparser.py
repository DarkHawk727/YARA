
"""
instantiate:
pdf1=PDFParser(pdf_path)
pdf1=PDFParser(pdf_path).extract_paragraphs(); 

calling extract_paragraph to do other pdfs: 
pdf1=PDFParser.extract_paragraphs(filepath) 
    pdf_path = pdf_path if pdf_path is not None else self.pdf_path
"""


import pdfplumber

class PDFparser:
    def __init__(self, optional_input=None):
        #self.pdf_file_path = pdf_path
        self.optional_input = optional_input
        

    def extract_paragraphs_2(self, pdf_path ): #optional argument, if not provided then use constructor argument.  
       # pass
        self.optional_input=pdf_path
        with pdfplumber.open(pdf_path) as pdf:
            paragraphs = []
            for page in pdf.pages:
                text = page.extract_text()
                paragraphs.extend(text.split('\n\n'))  # Assuming paragraphs are separated by two newlines

        return paragraphs

# for debug purpose: unit testing
"""
pdf_path = "C:\\Users\\evliu\\Documents\\parsing\\parsing\\pedpolicies-storm-drainage-policy.pdf"

pdfparser = PDFparser()
paragraphs=pdfparser.extract_paragraphs_2(pdf_path)
for paragraph in paragraphs:
    print(paragraph)

print(pdf_path)
"""





