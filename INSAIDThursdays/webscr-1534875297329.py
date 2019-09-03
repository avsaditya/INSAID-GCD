import urllib.request,re

f = urllib.request.urlopen("https://www.bc.edu/a-z/directories/contact/quicknos.html")
#("http://www.thehotelwindsor.com.au/")
s = f.read()

print(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}",str(s)))
print(re.findall(r"((?:\d{3}|\(\d{3}\))?(?:\s|-|\.)?\d{3}(?:\s|-|\.)\d{4})",str(s)))