import aspose.pdf as apdf
import sys
from os import path

if len(sys.argv) < 2:
    print("Usage: python pdf_pptx_converter.py <input.pdf> [output.pptx]")
    sys.exit(1)

infile = sys.argv[1]
outfile = sys.argv[2] if len(sys.argv) > 2 else infile.replace('.pdf', '.pptx')

document = apdf.Document(infile)
save_options = apdf.PptxSaveOptions()
document.save(outfile, save_options)

print(f"{infile} converted into {outfile}")
