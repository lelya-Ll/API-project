from tkinter import filedialog, Tk
import requests

Tk().withdraw()
filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])

if not filename:
    exit()

methods = {
    "1": "tf-idf",
    "2": "bag-of-words",
    "3": "tokenize",
    "4": "stemming",
    "5": "lemmatize",
    "6": "pos",
    "7": "ner",
    "8": "lsa",
    "9": "word2vec"
}

print("Выберите метод:")
for num, name in methods.items():
    print(f"{num}. {name}")

choice = input("Введите цифру (по умолчанию 1)): ").strip()
if not choice:
    choice = "1"

method = methods.get(choice, "tf-idf")

print(f"Файл: {filename}")
print(f"Метод: {method}")

try:
    with open(filename, 'rb') as f:
        response = requests.post(
            f"http://localhost:8000/upload-and-process?method={method}",
            files={"file": f}
        )

    if response.ok:
        result = response.json()
        print("\nРезультат:", result)
    else:
        print(f"Ошибка: {response.status_code}")

except Exception as e:
    print(f"Ошибка: {e}")