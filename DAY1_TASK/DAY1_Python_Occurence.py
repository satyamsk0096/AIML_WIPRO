import re

def find_python_occurrences(text):
    pattern = r'Python'
    matches = re.findall(pattern, text)
    return matches

text = "Python is a popular programming language. Python is widely used for web development, data science, and machine learning."

occurrences = find_python_occurrences(text)
print("Occurrences of 'Python':", occurrences)
